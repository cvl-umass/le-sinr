import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import datetime
from h3.unstable import vect

class CoordEncoder:

    def __init__(self, input_enc, raster=None, input_dim=0):
        self.input_enc = input_enc
        self.raster = raster
        self.input_dim = input_dim

    def encode(self, locs, normalize=True):
        if self.raster is not None:
            self.raster = self.raster.to(locs.device)
        # assumes lon, lat in range [-180, 180] and [-90, 90]
        if normalize:
            locs = normalize_coords(locs)
        if self.input_enc == 'sin_cos': # sinusoidal encoding
            loc_feats = encode_loc(locs, input_dim=self.input_dim)
        elif self.input_enc == 'env': # bioclim variables
            loc_feats = bilinear_interpolate(locs, self.raster)
        elif self.input_enc == 'sin_cos_env': # sinusoidal encoding & bioclim variables
            loc_feats = encode_loc(locs, input_dim=self.input_dim)
            context_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = torch.cat((loc_feats, context_feats), 1)
        elif self.input_enc == 'satclip': #SatClip Embedding
            if not hasattr(self, 'model'):
                import sys
                sys.path.append('./satclip/satclip')
                from satclip.satclip.load import get_satclip
                self.model = get_satclip('satclip/satclip-vit16-l10.ckpt', device="cpu")
                self.model.eval()
            self.model = self.model.to(locs.device)
            locs = locs*torch.tensor([[180.0, 90.0]], device=locs.device)
            max_batch = 1000000
            loc_feats = torch.empty(locs.shape[0], 256, device=locs.device)
            with torch.no_grad():
                for i in range(0, locs.shape[0], max_batch):
                    loc_feats[i:i+max_batch] = self.model(locs[i:i+max_batch].double()).float()
        elif self.input_enc == 'latlon':
            return locs[:,[1,0]] * torch.tensor([[90.0, 180.0]], device=locs.device)
        elif self.input_enc == 'lonlat':
            return locs * torch.tensor([[180.0, 90.0]], device=locs.device)
        elif self.input_enc == 'lonlat_env':
            loc_feats = locs * torch.tensor([[180.0, 90.0]], device=locs.device)
            context_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = torch.cat((loc_feats, context_feats), 1)
            return loc_feats
        elif self.input_enc == 'sh':
            if not hasattr(self, 'model'):
                import sys
                sys.path.append('./SH')
                from locationencoder import LocationEncoder

                hparams = dict(
                    legendre_polys=10,
                    dim_hidden=256,
                    num_layers=1,
                    optimizer=dict(lr=1e-4, wd=1e-3),
                    num_classes=256
                )
                # Pytorch Lightning Model
                self.model = LocationEncoder("sphericalharmonics", "siren", hparams).positional_encoder
            loc_feats = self.model((locs * torch.tensor([[180.0, 90.0]], device=locs.device)).double()).float()
        elif self.input_enc == 'sh_env':
            if not hasattr(self, 'model'):
                import sys
                sys.path.append('./SH')
                from locationencoder import LocationEncoder

                hparams = dict(
                    legendre_polys=10,
                    dim_hidden=256,
                    num_layers=1,
                    optimizer=dict(lr=1e-4, wd=1e-3),
                    num_classes=256
                )
                # Pytorch Lightning Model
                self.model = LocationEncoder("sphericalharmonics", "siren", hparams).positional_encoder
            context_feats = bilinear_interpolate(locs, self.raster)
            loc_feats = self.model((locs*torch.tensor([[180.0, 90.0]], device=locs.device)).double()).float()
            loc_feats = torch.cat((loc_feats, context_feats), 1)
        else:
            raise NotImplementedError('Unknown input encoding.')
        return loc_feats

    def encode_fast(self, loc: list[float], normalize=True):
        assert not normalize
        if self.input_enc == 'sin_cos':
            loc_feats = encode_loc_fast(loc, input_dim=self.input_dim)
        else:
            raise NotImplementedError('Unknown input encoding.')
        return loc_feats


def normalize_coords(locs):
    # locs is in lon {-180, 180}, lat {90, -90}
    # output is in the range [-1, 1]

    locs[:,0] /= 180.0
    locs[:,1] /= 90.0

    return locs

def encode_loc(loc_ip, concat_dim=1, input_dim=0):
    # assumes inputs location are in range -1 to 1
    # location is lon, lat
    encs = []
    for i in range(input_dim//4):
        encs.append(torch.sin(math.pi*(2**i)*loc_ip))
        encs.append(torch.cos(math.pi*(2**i)*loc_ip))
    feats = torch.cat(encs, concat_dim)
    return feats


def encode_loc_fast(loc_ip: list[float], input_dim=0):
    # assumes inputs location are in range -1 to 1
    # location is lon, lat
    input_dim //= 2 # needed to make it compatible with encode_loc
    feats = [(math.sin if i%(2*len(loc_ip))<len(loc_ip) else math.cos)(math.pi*(2**(i//(2*len(loc_ip))))*loc_ip[i%len(loc_ip)]) for i in range(input_dim)]
    return feats


def bilinear_interpolate(loc_ip, data, remove_nans_raster=True):
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    assert data is not None

    # map to [0,1], then scale to data size
    loc = (loc_ip.clone() + 1) / 2.0
    loc[:,1] = 1 - loc[:,1] # this is because latitude goes from +90 on top to bottom while
                            # longitude goes from -90 to 90 left to right

    assert not torch.any(torch.isnan(loc))

    if remove_nans_raster:
        data[torch.isnan(data)] = 0.0 # replace with mean value (0 is mean post-normalization)

    # cast locations into pixel space
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long()  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val

def rand_samples(batch_size, device, rand_type='uniform'):
    # randomly sample background locations

    if rand_type == 'spherical':
        rand_loc = torch.rand(batch_size, 2).to(device)
        theta1 = 2.0*math.pi*rand_loc[:, 0]
        theta2 = torch.acos(2.0*rand_loc[:, 1] - 1.0)
        lat = 1.0 - 2.0*theta2/math.pi
        lon = (theta1/math.pi) - 1.0
        rand_loc = torch.cat((lon.unsqueeze(1), lat.unsqueeze(1)), 1)

    elif rand_type == 'uniform':
        rand_loc = torch.rand(batch_size, 2).to(device)*2.0 - 1.0

    return rand_loc

def get_time_stamp():
    cur_time = str(datetime.datetime.now())
    date, time = cur_time.split(' ')
    h, m, s = time.split(':')
    s = s.split('.')[0]
    time_stamp = '{}-{}-{}-{}'.format(date, h, m, s)
    return time_stamp

def coord_grid(grid_size, split_ids=None, split_of_interest=None):
    # generate a grid of locations spaced evenly in coordinate space

    feats = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
    mg = np.meshgrid(np.linspace(-180, 180, feats.shape[1]), np.linspace(90, -90, feats.shape[0]))
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]
    if split_ids is None or split_of_interest is None:
        # return feats for all locations
        # this will be an N x 2 array
        return feats.reshape(feats.shape[0]*feats.shape[1], 2)
    else:
        # only select a subset of locations
        ind_y, ind_x = np.where(split_ids==split_of_interest)

        # these will be N_subset x 2 in size
        return feats[ind_y, ind_x, :]

def create_spatial_split(raster, mask, train_amt=1.0, cell_size=25):
    # generates a checkerboard style train test split
    # 0 is invalid, 1 is train, and 2 is test
    # c_size is units of pixels
    split_ids = np.ones((raster.shape[0], raster.shape[1]))
    start = cell_size
    for ii in np.arange(0, split_ids.shape[0], cell_size):
        if start == 0:
            start = cell_size
        else:
            start = 0
        for jj in np.arange(start, split_ids.shape[1], cell_size*2):
            split_ids[ii:ii+cell_size, jj:jj+cell_size] = 2
    split_ids = split_ids*mask
    if train_amt < 1.0:
        # take a subset of the data
        tr_y, tr_x = np.where(split_ids==1)
        inds = np.random.choice(len(tr_y), int(len(tr_y)*(1.0-train_amt)), replace=False)
        split_ids[tr_y[inds], tr_x[inds]] = 0
    return split_ids

def average_precision_score_faster(y_true, y_scores):
    # drop in replacement for sklearn's average_precision_score
    # comparable up to floating point differences
    num_positives = y_true.sum()
    inds = np.argsort(y_scores)[::-1]
    y_true_s = y_true[inds]

    false_pos_c = np.cumsum(1.0 - y_true_s)
    true_pos_c = np.cumsum(y_true_s)
    recall = true_pos_c / num_positives
    false_neg = np.maximum(true_pos_c + false_pos_c, np.finfo(np.float32).eps)
    precision = true_pos_c / false_neg

    recall_e = np.hstack((0, recall, 1))
    recall_e = (recall_e[1:] - recall_e[:-1])[:-1]
    map_score = (recall_e*precision).sum()
    return map_score


def average_precision_score_fasterer(y_true, y_scores):
    # drop in replacement for sklearn's average_precision_score
    # comparable up to floating point differences
    num_positives = y_true.sum()
    inds = torch.argsort(y_scores, descending=True)
    y_true_s = y_true[inds]

    false_pos_c = torch.cumsum(1.0 - y_true_s, dim=0)
    true_pos_c = torch.cumsum(y_true_s, dim=0)
    recall = true_pos_c / num_positives
    false_neg = (true_pos_c + false_pos_c).clip(min=np.finfo(np.float32).eps)
    precision = true_pos_c / false_neg

    recall_e = torch.cat([torch.zeros(1, device=recall.device), recall, torch.ones(1, device=recall.device)])
    recall_e = (recall_e[1:] - recall_e[:-1])[:-1]
    map_score = (recall_e*precision).sum()
    return map_score


def lonlat2cartesian(lonlat_array):
    """
    Convert from longitude and latitude to 3D Cartesian coordinates for an array of points.

    Parameters:
        lonlat_array (np.ndarray): Array of shape (N, 2) where each row is [lon, lat] in degrees.

    Returns:
        np.ndarray: Array of shape (N, 3) where each row is [x, y, z].
    """
    # Convert degrees to radians
    lon_rad = np.radians(lonlat_array[:, 0])  # Longitude
    lat_rad = np.radians(lonlat_array[:, 1])  # Latitude

    # Calculate Cartesian coordinates
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack((x, y, z))


def cartesian2lonlat(cartesian_array):
    """
    Convert from 3D Cartesian coordinates to longitude and latitude for an array of points.

    Parameters:
        cartesian_array (np.ndarray): Array of shape (N, 3) where each row is [x, y, z].

    Returns:
        np.ndarray: Array of shape (N, 2) where each row is [lon, lat] in degrees.
    """
    x = cartesian_array[:, 0]
    y = cartesian_array[:, 1]
    z = cartesian_array[:, 2]

    # Calculate latitude and longitude in radians
    lat_rad = np.arcsin(z)
    lon_rad = np.arctan2(y, x)

    # Convert radians to degrees
    lon = np.degrees(lon_rad)
    lat = np.degrees(lat_rad)

    return np.column_stack((lon, lat))
