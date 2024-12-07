import os
import numpy as np
import json
import pandas as pd
from calendar import monthrange
import torch
import utils
import random
from h3.unstable import vect
import h3.api.numpy_int as h3

class LocationDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, input_dim=4):

        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env', 'sh_env', 'lonlat_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster, input_dim=input_dim)

        # define some properties:
        self.locs = locs
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = labels
        self.classes = classes
        self.class_to_taxa = class_to_taxa

        # useful numbers:
        self.num_classes = len(np.unique(labels))
        self.input_dim = input_dim

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        class_id  = self.labels[index]

        return loc_feat, loc, class_id

def load_env():
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    raster = load_context_feats(os.path.join(paths['env'],'bioclim_elevation_scaled.npy'))
    return raster

def load_context_feats(data_path):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats = torch.from_numpy(context_feats)
    return context_feats

_file_cache = {}
def load_inat_data(ip_file, taxa_of_interest=None):
    if os.path.exists('.datacache.pt'):
        print('\nLoading cached data')
        if '.datacache.pt' not in _file_cache:
            # If not in the cache, read the file and store its content in the cache
            _file_cache['.datacache.pt'] = torch.load('.datacache.pt')
        locs, taxa, users, dates, years, obs_ids = _file_cache['.datacache.pt']
    else:
        print('\nLoading  ' + ip_file)
        data = pd.read_csv(ip_file)

        # remove outliers
        num_obs = data.shape[0]
        data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
        if (num_obs - data.shape[0]) > 0:
            print(num_obs - data.shape[0], 'items filtered due to invalid locations')

        if 'accuracy' in data.columns:
            data.drop(['accuracy'], axis=1, inplace=True)

        if 'positional_accuracy' in data.columns:
            data.drop(['positional_accuracy'], axis=1, inplace=True)

        if 'geoprivacy' in data.columns:
            data.drop(['geoprivacy'], axis=1, inplace=True)

        if 'observed_on' in data.columns:
            data.rename(columns = {'observed_on':'date'}, inplace=True)

        num_obs_orig = data.shape[0]
        data = data.dropna()
        size_diff = num_obs_orig - data.shape[0]
        if size_diff > 0:
            print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

        # keep only taxa of interest:
        if taxa_of_interest is not None:
            num_obs_orig = data.shape[0]
            data = data[data['taxon_id'].isin(taxa_of_interest)]
            print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

        print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

        locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
        taxa = data['taxon_id'].values.astype(np.int64)

        if 'user_id' in data.columns:
            users = data['user_id'].values.astype(np.int64)
            _, users = np.unique(users, return_inverse=True)
        elif 'observer_id' in data.columns:
            users = data['observer_id'].values.astype(np.int64)
            _, users = np.unique(users, return_inverse=True)
        else:
            users = np.ones(taxa.shape[0], dtype=np.int64)*-1

        # Note - assumes that dates are in format YYYY-MM-DD
        temp = np.array(data['date'], dtype='S10')
        temp = temp.view('S1').reshape((temp.size, -1))
        years = temp[:,:4].view('S4').astype(int)[:,0]
        months = temp[:,5:7].view('S2').astype(int)[:,0]
        days = temp[:,8:10].view('S2').astype(int)[:,0]
        days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
        dates  = days_per_month[months-1] + days-1
        dates  = np.round((dates) / 364.0, 4).astype(np.float32)
        if 'id' in data.columns:
            obs_ids = data['id'].values
        elif 'observation_uuid' in data.columns:
            obs_ids = data['observation_uuid'].values
        torch.save((locs, taxa, users, dates, years, obs_ids), '.datacache.pt')

    return locs, taxa, users, dates, years, obs_ids

def choose_aux_species(current_species, num_aux_species, aux_species_seed, taxa_file):
    if num_aux_species == 0:
        return []
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    taxa_file = os.path.join(data_dir, taxa_file)
    with open(taxa_file, 'r') as f:
        inat_large_metadata = json.load(f)
    aux_species_candidates = [x['taxon_id'] for x in inat_large_metadata]
    aux_species_candidates = np.setdiff1d(aux_species_candidates, current_species)
    print(f'choosing {num_aux_species} species to add from {len(aux_species_candidates)} candidates')
    rng = np.random.default_rng(aux_species_seed)
    idx_rand_aux_species = rng.permutation(len(aux_species_candidates))
    aux_species = list(aux_species_candidates[idx_rand_aux_species[:num_aux_species]])
    return aux_species

def get_taxa_of_interest(species_set='all', num_aux_species=0, aux_species_seed=123, taxa_file=None, taxa_file_snt=None):
    if species_set == 'all':
        return None
    if species_set == 'snt_birds':
        assert taxa_file_snt is not None
        with open(taxa_file_snt, 'r') as f: #
            taxa_subsets = json.load(f)
        taxa_of_interest = list(taxa_subsets['snt_birds'])
    else:
        raise NotImplementedError
    # optionally add some other species back in:
    aux_species = choose_aux_species(taxa_of_interest, num_aux_species, aux_species_seed, taxa_file)
    taxa_of_interest.extend(aux_species)
    return taxa_of_interest

def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123, subset=None, subset_cap=-1):
    if hard_cap == -1:
        if subset_cap != -1:
            raise NotImplementedError('subset_cap set but not hard_cap')
        return np.arange(len(labels))
    print(f'subsampling (up to) {hard_cap} per class for the training set')
    ids, counts = np.unique(labels, return_counts=True)
    count_ind = np.cumsum(counts)
    count_ind[1:] = count_ind[:-1]
    count_ind[0] = 0
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))

    ordered_inds = np.argsort(labels[idx_rand], kind='stable')
    caps = hard_cap + np.zeros_like(counts)
    if subset is not None and subset_cap != -1:
        caps[subset] = subset_cap
    idx_ss = idx_rand[np.concatenate([ordered_inds[i:i+min(limit, cap)] for i, limit, cap in zip(count_ind, counts, caps)])]
    print(f'final training set size: {len(idx_ss)}')
    return idx_ss

def uniform_sample_h3(cells, low, high):
    '''uniformly sample points in a batch of h3 cells'''
    out = np.empty((len(cells), 2))
    invalid_mask = np.arange(len(cells))
    cell_ids_buffer = np.empty(len(cells), dtype='uint64')
    while len(invalid_mask) > 0:
        #print(len(invalid_mask))
        pts = np.random.random((len(invalid_mask), 2))
        pts = high + pts*(low - high)

        vect._vect.geo_to_h3_vect(pts[:,0], pts[:,1], 5, cell_ids_buffer)

        valid_mask = (cell_ids_buffer[:len(cells)] == cells)
        out[invalid_mask[valid_mask]] = pts[valid_mask]
        neg_mask = ~valid_mask
        invalid_mask = invalid_mask[neg_mask]
        low = low[neg_mask]
        high = high[neg_mask]
        cells = cells[neg_mask]

    return out


def get_train_data(params):
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    obs_file  = os.path.join(data_dir, params['obs_file'])
    taxa_file = os.path.join(data_dir, params['taxa_file'])
    taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

    taxa_of_interest = get_taxa_of_interest(params['species_set'], params['num_aux_species'], params['aux_species_seed'], params['taxa_file'], taxa_file_snt)

    locs, labels, _, dates, _, _ = load_inat_data(obs_file, taxa_of_interest)
    if params['zero_shot']:
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
        D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
        D = D.item()
        taxa_snt = D['taxa'].tolist()
        taxa = [int(tt) for tt in data['taxa_presence'].keys()]
        taxa = list(set(taxa + taxa_snt))
        mask = labels != taxa[0]
        for i in range(1, len(taxa)):
            mask &= (labels != taxa[i])
        locs = locs[mask]
        dates = dates[mask]
        labels = labels[mask]
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    # load class names
    class_info_file = json.load(open(taxa_file, 'r'))
    class_names_file = [cc['latin_name'] for cc in class_info_file]
    taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    classes = dict(zip(taxa_ids_file, class_names_file))

    subset = None
    if params['subset_cap_name'] is not None:
        if params['subset_cap_name'] == 'iucn':
            with open('paths.json', 'r') as f:
                paths = json.load(f)
            with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
            taxa = [int(tt) for tt in data['taxa_presence'].keys()]
            # get classes to eval
            subset = np.zeros((len(taxa),), dtype=int)
            for tt_id, tt in enumerate(taxa):
                class_of_interest = np.where(np.array(class_to_taxa) == tt)[0]
                if len(class_of_interest) != 0:
                    subset[tt_id] = class_of_interest
        else:
            raise NotImplementedError(f'Uknown subset name: {params["subset_cap_name"]}')

    idx_ss = get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'], subset, params['subset_cap_num_per_class'])

    locs = torch.from_numpy(np.array(locs)[idx_ss]) # convert to Tensor

    labels = torch.from_numpy(np.array(class_ids)[idx_ss])

    ds = LocationDataset(locs, labels, classes, class_to_taxa, input_enc=params['input_enc'], device=params['device'], input_dim=params['input_dim'])

    return ds

