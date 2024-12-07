import torch
import torch.utils.data
import torch.nn as nn
import math
import csv
import numpy as np
import json
import os


def get_model(params, inference_only=False):
    if params['model'] == 'ResidualFCNet':
        return ResidualFCNet(params['input_dim'] + (20 if 'env' in params['input_enc'] and 'contrastive' not in params['input_enc'] else 0), params['num_classes'] + (20 if 'env' in params['loss'] else 0), params['num_filts'], params['depth'])
    elif params['model'] == 'LinNet':
        return LinNet(params['input_dim'] + (20 if 'env' in params['input_enc'] else 0), params['num_classes'])
    elif params['model'] == 'HyperNet':
        return HyperNet(params, params['input_dim'] + (20 if 'env' in params['input_enc'] else 0), params['num_classes'], params['num_filts'], params['depth'],
                                params['species_dim'], params['species_enc_depth'], params['species_filts'], params['species_enc'], params['pos_enc'], inference_only=inference_only)
    else:
        raise NotImplementedError('Invalid model specified.')


class ResLayer(nn.Module):
    def __init__(self, linear_size, activation=nn.ReLU):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = activation()
        self.nonlin2 = activation()
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class ResidualFCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, depth=4, nonlin='relu', lowrank=0):
        super(ResidualFCNet, self).__init__()
        self.inc_bias = False
        if lowrank < num_filts and lowrank != 0:
            l1 = nn.Linear(num_filts if depth != -1 else num_inputs, lowrank, bias=self.inc_bias)
            l2 = nn.Linear(lowrank, num_classes, bias=self.inc_bias)
            self.class_emb = nn.Sequential(l1, l2)
        else:
            self.class_emb = nn.Linear(num_filts if depth != -1 else num_inputs, num_classes, bias=self.inc_bias)
        if nonlin == 'relu':
            activation = nn.ReLU
        elif nonlin == 'silu':
            activation = nn.SiLU
        else:
            raise NotImplementedError('Invalid nonlinearity specified.')
        layers = []
        if depth != -1:
            layers.append(nn.Linear(num_inputs, num_filts))
            layers.append(activation())
            for i in range(depth):
                layers.append(ResLayer(num_filts, activation=activation))
        else:
            layers.append(nn.Identity())
        self.feats = torch.nn.Sequential(*layers)

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest), self.eval_single_class(loc_emb, -1)
            return torch.sigmoid(class_pred[0]), torch.sigmoid(class_pred[1])
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]


class SimpleFCNet(ResidualFCNet):
    def forward(self, x, return_feats=True):
        assert return_feats
        loc_emb = self.feats(x)
        class_pred = self.class_emb(loc_emb)
        return class_pred



class LinNet(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(LinNet, self).__init__()
        self.num_layers = 0
        self.inc_bias = False
        self.class_emb = nn.Linear(num_inputs, num_classes, bias=self.inc_bias)
        self.feats = nn.Identity()  # does not do anything

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]


class ParallelMulti(torch.nn.Module):
    def __init__(self, x: list[torch.nn.Module]):
        super(ParallelMulti, self).__init__()
        self.layers = nn.ModuleList(x)

    def forward(self, xs, **kwargs):
        out = torch.cat([self.layers[i](x, **kwargs) for i,x in enumerate(xs)], dim=1)
        return out


class SequentialMulti(torch.nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs, **kwargs)
            else:
                inputs = module(inputs)
        return inputs


class SHEnv(nn.Module):
    def __init__(self, hparams):
        super(SHEnv, self).__init__()
        import sys
        sys.path.append('./SH')
        from locationencoder import LocationEncoder
        from locationencoder.locationencoder import get_neural_network

        # Pytorch Lightning Model
        model = LocationEncoder("sphericalharmonics", "siren", hparams)
        self.encoder = model.positional_encoder
        self.network = get_neural_network('siren', hparams['legendre_polys']**2 + 20, hparams=hparams)

    def forward(self, x):
        pos_enc = self.encoder(x[:,:2].double()).float()
        pos_enc = torch.cat([pos_enc, x[:,2:]], dim=1)
        return self.network(pos_enc)


class HyperNet(nn.Module):
    '''
    :param asdf
    '''
    def __init__(self, params, num_inputs, num_classes, num_filts, pos_enc_depth, species_dim, species_enc_depth, species_filts, species_enc='embed', pos_enc='FCNet', inference_only=False):
        super(HyperNet, self).__init__()
        if species_enc == 'embed':
            self.species_emb = nn.Embedding(num_classes, species_dim)
            self.species_emb.weight.data *= 0.01
        elif species_enc == 'wiki':
            self.species_emb = WikiEncoder(params, params['text_emb_path'], species_dim, inference_only=inference_only)

        if species_enc_depth == -1:
            self.species_enc = nn.Identity()
        elif species_enc_depth == 0:
            self.species_enc = nn.Linear(species_dim, num_filts+1)
        else:
            self.species_enc = SimpleFCNet(species_dim, num_filts+1, species_filts, depth=species_enc_depth)

        if 'geoprior' in params['loss']:
            self.species_params = nn.Parameter(torch.randn(num_classes, species_dim))
            self.species_params.data *= 0.0386

        if 'contrastive' in params['loss']:
            self.temp = nn.Parameter(2.65926003693*torch.ones(1))

        if pos_enc == 'FCNet':
            self.pos_enc = SimpleFCNet(num_inputs, num_filts, num_filts, depth=pos_enc_depth)
        elif pos_enc == 'GeoCLIP_pretrained':
            from geoclip import LocationEncoder
            self.pos_enc = LocationEncoder()
            if num_filts != 512:
                self.pos_enc = nn.Sequential(self.pos_enc, nn.Linear(512, num_filts))
        elif pos_enc == 'GeoCLIP':
            from geoclip import LocationEncoder
            self.pos_enc = LocationEncoder(from_pretrained=False)
            if num_filts != 512:
                self.pos_enc = nn.Sequential(self.pos_enc, nn.Linear(512, num_filts))
        elif pos_enc == 'SH':
            import sys
            sys.path.append('./SH')
            from locationencoder import LocationEncoder

            hparams = dict(
                legendre_polys=10,
                dim_hidden=num_filts,
                num_layers=pos_enc_depth,
                optimizer=dict(lr=1e-4, wd=1e-3),
                num_classes=num_filts
            )
            # Pytorch Lightning Model
            self.pos_enc = LocationEncoder("sphericalharmonics", "siren", hparams)
        elif pos_enc == 'SHEnv':
            hparams = dict(
                legendre_polys=40,
                dim_hidden=num_filts,
                num_layers=pos_enc_depth,
                optimizer=dict(lr=1e-4, wd=1e-3),
                num_classes=num_filts
            )
            self.pos_enc = SHEnv(hparams)
        elif pos_enc == 'SirenNet':
            import sys
            sys.path.append('./SH')
            from locationencoder.locationencoder import get_neural_network

            hparams = dict(
                legendre_polys=10,
                dim_hidden=num_filts,
                num_layers=pos_enc_depth,
                num_classes=num_filts
            )
            self.pos_enc = get_neural_network('siren', 100 + (20 if 'env' in params['input_enc'] else 0), hparams=hparams)

    def forward(self, x, y):
        ys, indmap = torch.unique(y, return_inverse=True)
        species = self.species_enc(self.species_emb(ys))
        species_w, _ = species[...,:-1], species[...,-1:]
        pos = self.pos_enc(x)
        out = torch.bmm(species_w[indmap],pos[...,None])
        out = out.squeeze(-1)
        if hasattr(self, 'species_params'):
            out2 = torch.bmm(self.species_params[ys][indmap],pos[...,None])
            out2 = out2.squeeze(-1)
            out3 = (species_w, self.species_params[ys], ys)
            return out, out2, out3
        else:
            return out

    def zero_shot(self, x, species_emb):
        species = self.species_enc(self.species_emb.zero_shot(species_emb))
        species_w, _ = species[...,:-1], species[...,-1:]
        pos = self.pos_enc(x)
        out = pos @ species_w.T
        return out


class WikiEncoder(nn.Module):
    def __init__(self, params, path, embedding_dim, inference_only=False):
        super(WikiEncoder, self).__init__()
        self.path = path
        if not inference_only:
            import datasets
            with open('paths.json', 'r') as f:
                paths = json.load(f)
            data_dir = paths['train']
            obs_file = os.path.join(data_dir, params['obs_file'])
            taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

            taxa_of_interest = datasets.get_taxa_of_interest(params['species_set'], params['num_aux_species'],
                                                    params['aux_species_seed'], params['taxa_file'], taxa_file_snt)

            locs, labels, _, dates, _, _ = datasets.load_inat_data(obs_file, taxa_of_interest)
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
                labels = labels[mask]
            unique_taxa, class_ids = np.unique(labels, return_inverse=True)
            class_to_taxa = unique_taxa.tolist()

            embs = torch.load(path)
            ids = embs['taxon_id'].tolist()
            if 'keys' in embs:
                taxa_counts = torch.zeros(len(ids), dtype=torch.int32)
                for i,k in embs['keys']:
                    taxa_counts[i] += 1
            else:
                taxa_counts = torch.ones(len(ids), dtype=torch.int32)
            count_sum = torch.cumsum(taxa_counts, dim=0) - taxa_counts
            embs = embs['data']

            self.taxa2row = {taxaid:i for i, taxaid in enumerate(ids)}
            indmap = -1+torch.zeros(len(class_to_taxa), dtype=torch.int)
            countmap = torch.zeros(len(class_to_taxa), dtype=torch.int)
            self.species_emb = nn.Embedding(len(class_to_taxa), embedding_dim)
            self.species_emb.weight.data *= 0.01

            for i in range(len(class_to_taxa)):
                if class_to_taxa[i] in ids:
                    i2 = ids.index(class_to_taxa[i])
                    indmap[i] = count_sum[i2]
                    countmap[i] = taxa_counts[i2]

            self.register_buffer('indmap', indmap, persistent=False)
            self.register_buffer('countmap', countmap, persistent=False)
            self.register_buffer('embs', embs, persistent=False)
            assert embs.shape[1] == 4096
        self.scale = nn.Parameter(torch.zeros(1))
        if params['text_hidden_dim'] == 0:
            self.linear1 = nn.Linear(4096, embedding_dim)
        else:
            self.linear1 = nn.Linear(4096, params['text_hidden_dim'])
            for l in range(params['text_num_layers']-1):
                setattr(self, f'linear{l+2}', nn.Linear(params['text_hidden_dim'], params['text_hidden_dim']))
            setattr(self, f'linear{params["text_num_layers"]+1}', nn.Linear(params['text_hidden_dim'], embedding_dim))
            self.act = nn.SiLU()

    def forward(self, x):
        inds = self.indmap[x] + (torch.rand(x.shape,device=x.device)*self.countmap[x]).floor().int()
        out = self.embs[inds]
        out = self.linear1(out)
        if hasattr(self, 'linear2'):
            out = self.act(out)
        i = 2
        while hasattr(self, f'linear{i}'):
            if hasattr(self, f'linear{i}'):
                out = self.act(getattr(self, f'linear{i}')(out))
            i += 1
        #out = self.scale * (out / (out.std(dim=1)[:, None]))
        out2 = self.species_emb(x)
        chosen = torch.ones((out.shape[0],), device=x.device)
        chosen[inds == -1] = 0
        out = chosen[:,None] * out + (1-chosen[:,None])*out2
        return out


    def zero_shot(self, species_emb):
        out = species_emb
        out = self.linear1(out)
        if hasattr(self, 'linear2'):
            out = self.act(out)
        i = 2
        while hasattr(self, f'linear{i}'):
            if hasattr(self, f'linear{i}'):
                out = self.act(getattr(self, f'linear{i}')(out))
            i += 1
        return out