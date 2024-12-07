import collections
import sys

import numpy as np
import pandas as pd
import random
import torch
import time
import os
import copy
import json
import tifffile
import h3
import setup

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import utils
import models
import datasets
from calendar import monthrange
from torch.nn.functional import logsigmoid, softmax
from tqdm import tqdm

class EvaluatorSNT:
    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
        D = D.item()
        self.loc_indices_per_species = D['loc_indices_per_species']
        self.labels_per_species = D['labels_per_species']
        self.taxa = D['taxa']
        self.obs_locs = D['obs_locs']
        self.obs_locs_idx = D['obs_locs_idx']

    def get_labels(self, species):
        species = str(species)
        lat = []
        lon = []
        gt = []
        for hx in self.data:
            cur_lat, cur_lon = h3.h3_to_geo(hx)
            if species in self.data[hx]:
                cur_label = int(len(self.data[hx][species]) > 0)
                gt.append(cur_label)
                lat.append(cur_lat)
                lon.append(cur_lon)
        lat = np.array(lat).astype(np.float32)
        lon = np.array(lon).astype(np.float32)
        obs_locs = np.vstack((lon, lat)).T
        gt = np.array(gt).astype(np.float32)
        return obs_locs, gt

    @torch.no_grad()
    def run_evaluation(self, model, enc, extra_input=None):
        results = {}

        # set seeds:
        np.random.seed(self.eval_params['seed'])
        random.seed(self.eval_params['seed'])

        # evaluate the geo model for each taxon
        results['per_species_average_precision_all'] = np.zeros((len(self.taxa)), dtype=np.float32)

        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = torch.cat([enc.encode(obs_locs), extra_input.expand(obs_locs.shape[0], -1)], dim=1) if extra_input is not None else enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        if self.eval_params['extract_pos']:
            assert 'HyperNet' in self.train_params['model']
            model = model.pos_enc
            self.train_params['model'] = 'ResidualFCNet'

        if self.train_params['model'] == 'ResidualFCNet':
            feature_extractor = lambda x: model(x, return_feats=True)
        elif 'HyperNet' in self.train_params['model']:
            feature_extractor = lambda x: model.pos_enc(x)
        else:
            raise NotImplementedError(f'Unknown model type, {self.train_params["model"]}')

        if 'HyperNet' not in self.train_params['model'] and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
            # generate model predictions for classes of interest at eval locations
            loc_emb = feature_extractor(loc_feat)
            wt = model.class_emb.weight[classes_of_interest, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1))
        elif self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0:
            import datasets
            with open('paths.json', 'r') as f:
                paths = json.load(f)
            data_dir = paths['train']
            obs_file = os.path.join(data_dir, self.train_params['obs_file'])
            taxa_file = os.path.join(data_dir, self.train_params['taxa_file'])
            taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

            taxa_of_interest = datasets.get_taxa_of_interest(self.train_params['species_set'],
                                                             self.train_params['num_aux_species'],
                                                             self.train_params['aux_species_seed'],
                                                             self.train_params['taxa_file'], taxa_file_snt)

            locs, labels, _, dates, _, _ = datasets.load_inat_data(obs_file, taxa_of_interest)
            unique_taxa, class_ids = np.unique(labels, return_inverse=True)
            class_to_taxa = unique_taxa.tolist()
            if self.eval_params['num_samples'] > 0:
                idx_ss = datasets.get_idx_subsample_observations(labels, self.eval_params['num_samples'],
                                                                 random.randint(0, 2 ** 32), None, -1)
                locs = torch.from_numpy(np.array(locs)[idx_ss])
                labels = torch.from_numpy(np.array(class_ids)[idx_ss])
                pos_examples = {}
                for tt in self.taxa:
                    c = class_to_taxa.index(tt)
                    pos_examples[tt] = locs[labels == c]
                    pos_examples[tt] = feature_extractor(enc.encode(pos_examples[tt].to(self.eval_params['device']))).cpu()
                neg_examples = utils.rand_samples(10000, self.eval_params['device'], rand_type='spherical')
                if extra_input is not None:
                    raise NotImplementedError('extra_input provided')
                neg_examples = feature_extractor(torch.cat([enc.encode(neg_examples, normalize=False), enc.encode(locs[torch.randperm(locs.shape[0], device=locs.device)[:10000]].clone().to(self.eval_params['device']), normalize=True)])).cpu()
            loc_emb = feature_extractor(loc_feat)

            if self.train_params['model'] == 'HyperNet':
                embs = torch.load('./data/eval/gpt_data.pt')
                emb_ids = embs['taxon_id'].tolist()
                keys = embs['keys']
                embs = embs['data']
        if self.eval_params['num_samples'] == -1:
            loc_emb = feature_extractor(loc_feat)
        split_rng = np.random.default_rng(self.eval_params['split_seed'])
        for tt_id, tt in enumerate(self.taxa):

            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) == 0 and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
                continue
            # generate ground truth labels for current taxa
            cur_loc_indices = np.array(self.loc_indices_per_species[tt_id])
            cur_labels = np.array(self.labels_per_species[tt_id])

            # apply per-species split:
            assert self.eval_params['split'] in ['all', 'val', 'test']
            if self.eval_params['split'] != 'all':
                num_val = np.floor(len(cur_labels) * self.eval_params['val_frac']).astype(int)
                idx_rand = split_rng.permutation(len(cur_labels))
                if self.eval_params['split'] == 'val':
                    idx_sel = idx_rand[:num_val]
                elif self.eval_params['split'] == 'test':
                    idx_sel = idx_rand[num_val:]
                cur_loc_indices = cur_loc_indices[idx_sel]
                cur_labels = cur_labels[idx_sel]
            cur_labels = (torch.from_numpy(cur_labels).to(self.eval_params['device']) > 0).float()

            ##########################################################################################
            #
            ##########################################################################################
            if self.eval_params['num_samples'] == -1:
                species_w = model.species_params[self.train_params['class_to_taxa'].index(tt)]
                preds = loc_emb @ species_w.detach()
                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(cur_labels,
                                                                                                             preds[cur_loc_indices]).item()
                continue

            if 'HyperNet' not in self.train_params['model'] and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                # extract model predictions for current taxa from prediction matrix
                pred = pred_mtx[cur_loc_indices, tt_id]
            elif self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0:
                if self.eval_params['num_samples'] > 0:
                    X = torch.cat([pos_examples[tt], neg_examples], dim=0).to(self.eval_params['device'])
                    y = torch.zeros(X.shape[0], dtype=torch.long).to(self.eval_params['device'])
                    y[:pos_examples[tt].shape[0]] = 1
                C = 0.05
                crit = torch.nn.BCEWithLogitsLoss()
                crit2 = torch.nn.MSELoss()

                if self.train_params['model'] == 'ResidualFCNet':
                    w = torch.nn.Parameter(torch.zeros(X.shape[1], 1))
                    opt = torch.optim.Rprop([w], lr=0.001)
                    with torch.set_grad_enabled(True):
                        for i in range(40):
                            opt.zero_grad()
                            output = X @ w
                            yhat = y.float()[:, None]
                            loss = 0.5 * crit(output[yhat == 0], yhat[yhat == 0]) + 0.5 * crit(output[yhat == 1],
                                                                                               yhat[
                                                                                                   yhat == 1]) + 1 / (
                                               C * len(pos_examples[tt])) * crit2(w, 0 * w)
                            loss.backward()
                            opt.step()
                    pred = torch.sigmoid(((loc_emb @ w.cuda())))[cur_loc_indices].flatten()

                elif self.train_params['model'] == 'HyperNet':
                    if tt not in emb_ids:
                        results['per_species_average_precision_all'][tt_id] = 0.0
                        continue
                    sec_ind = emb_ids.index(tt)
                    sections = [i for i,x in enumerate(keys) if x[0] == sec_ind]

                    def get_feat(x):
                        species = model.species_enc(model.species_emb.zero_shot(x))
                        species_w, species_b = species[..., :-1], species[..., -1:]
                        if self.eval_params['num_samples'] == 0:
                            out = loc_emb @ (species_w.detach()).T
                            return out
                        w = torch.nn.Parameter(torch.zeros_like(species_w))
                        opt = torch.optim.Rprop([w], lr=0.001)
                        with torch.set_grad_enabled(True):
                            for i in range(40):
                                opt.zero_grad()
                                output = (X @ (w + species_w.detach()).T)
                                yhat = y.float()[:, None].repeat(1, w.shape[0])
                                loss = 0.5*crit(output[yhat == 0], yhat[yhat == 0]) + 0.5*crit(output[yhat == 1], yhat[yhat == 1]) + \
                                    1/(C*len(pos_examples[tt])) * crit2(w, 0 * w)
                                loss.backward()
                                opt.step()
                        out = loc_emb @ (w.data + species_w.detach()).T
                        return out
                    # average precision score:
                    yfeats = torch.cat([embs[section][None].to(self.eval_params['device']) for section in sections])
                    preds = get_feat(yfeats)
                    if len(sections) > 1:#'habitat', 'overview_summary'
                        kws = [self.eval_params['text_section']]
                        best_sections = [i for i,s in enumerate(sections) if any((x in keys[s][1].lower() for x in kws))]
                        results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(cur_labels, preds[cur_loc_indices][:,best_sections].mean(dim=1)).item()
                    else:
                        results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(cur_labels, preds[cur_loc_indices][:,0].mean(dim=1)).item()
                    continue
            else:
                raise NotImplementedError('Eval not implemented')
            # compute the AP for each taxa
            results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(cur_labels, pred).item()

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)

        return results

    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')

class EvaluatorIUCN:

    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            self.data = json.load(f)
        self.obs_locs = np.array(self.data['locs'], dtype=np.float32)
        self.taxa = [int(tt) for tt in self.data['taxa_presence'].keys()]

    @torch.no_grad()
    def run_evaluation(self, model, enc, extra_input=None):
        results = {}
        #self.train_params['model'] = 'ResidualFCNet'
        #m = model
        #model = lambda x, return_feats=True: m.pos_enc(x)
        results['per_species_average_precision_all'] = np.zeros(len(self.taxa), dtype=np.float32)
        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = torch.cat([enc.encode(obs_locs), extra_input.expand(obs_locs.shape[0], -1)], dim=1) if extra_input is not None else enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        if self.eval_params['extract_pos']:
            assert 'HyperNet' in self.train_params['model']
            model = model.pos_enc
            self.train_params['model'] = 'ResidualFCNet'

        if 'HyperNet' not in self.train_params['model'] and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
            # generate model predictions for classes of interest at eval locations
            loc_emb = model(loc_feat, return_feats=True)
            wt = model.class_emb.weight[classes_of_interest, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1))
        elif (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
            if self.train_params['model'] == 'ResidualFCNet':
                import datasets
                from sklearn.linear_model import LogisticRegression
                with open('paths.json', 'r') as f:
                    paths = json.load(f)
                data_dir = paths['train']
                obs_file = os.path.join(data_dir, self.train_params['obs_file'])
                taxa_file = os.path.join(data_dir, self.train_params['taxa_file'])
                taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

                taxa_of_interest = datasets.get_taxa_of_interest(self.train_params['species_set'], self.train_params['num_aux_species'],
                                                        self.train_params['aux_species_seed'], self.train_params['taxa_file'], taxa_file_snt)

                locs, labels, _, dates, _, _ = datasets.load_inat_data(obs_file, taxa_of_interest)
                unique_taxa, class_ids = np.unique(labels, return_inverse=True)
                class_to_taxa = unique_taxa.tolist()
                idx_ss = datasets.get_idx_subsample_observations(labels, self.eval_params['num_samples'], random.randint(0,2**32), None, -1)
                locs = torch.from_numpy(np.array(locs))
                labels = torch.from_numpy(np.array(class_ids))
                locs = locs[idx_ss]
                labels = labels[idx_ss]
                pos_examples = {}
                for tt in self.taxa:
                    c = class_to_taxa.index(tt)
                    pos_examples[tt] = locs[labels == c]
                    pos_examples[tt] = model(enc.encode(pos_examples[tt].to(self.eval_params['device'])), return_feats=True).cpu()
                neg_examples = utils.rand_samples(10000, self.eval_params['device'], rand_type='spherical')
                if extra_input is not None:
                    raise NotImplementedError('extra_input provided')
                neg_examples = model(torch.cat([enc.encode(neg_examples, normalize=False), enc.encode(locs[torch.randperm(locs.shape[0], device=locs.device)[:10000]].clone().to(self.eval_params['device']), normalize=True)]), return_feats=True).cpu()
                loc_emb = model(loc_feat, return_feats=True)
            elif self.train_params['model'] == 'HyperNet':
                import datasets
                from sklearn.linear_model import LogisticRegression
                with open('paths.json', 'r') as f:
                    paths = json.load(f)
                data_dir = paths['train']
                obs_file = os.path.join(data_dir, self.train_params['obs_file'])
                taxa_file = os.path.join(data_dir, self.train_params['taxa_file'])
                taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

                taxa_of_interest = datasets.get_taxa_of_interest(self.train_params['species_set'], self.train_params['num_aux_species'],
                                                        self.train_params['aux_species_seed'], self.train_params['taxa_file'], taxa_file_snt)
                locs, labels, _, dates, _, _ = datasets.load_inat_data(obs_file, taxa_of_interest)
                unique_taxa, class_ids, class_counts = np.unique(labels, return_inverse=True, return_counts=True)
                class_counts = class_counts.clip(max=1000)
                if self.eval_params['num_samples'] > 0:
                    class_to_taxa = unique_taxa.tolist()
                    idx_ss = datasets.get_idx_subsample_observations(labels, self.eval_params['num_samples'], random.randint(0,2**32), None, -1)
                    locs = torch.from_numpy(np.array(locs))
                    labels = torch.from_numpy(np.array(class_ids))
                    locs = locs[idx_ss]
                    labels = labels[idx_ss]
                    pos_examples = {}
                    for tt in self.taxa:
                        c = class_to_taxa.index(tt)
                        pos_examples[tt] = locs[labels == c]
                        pos_examples[tt] = model.pos_enc(enc.encode(pos_examples[tt].to(self.eval_params['device']))).cpu()
                    neg_examples = utils.rand_samples(10000, self.eval_params['device'], rand_type='spherical')
                    if extra_input is not None:
                        raise NotImplementedError('extra_input provided')
                    neg_examples = model.pos_enc(torch.cat([enc.encode(neg_examples, normalize=False), enc.encode(locs[torch.randperm(locs.shape[0], device=locs.device)[:10000]].clone().to(self.eval_params['device']), normalize=True)])).cpu()

                embs = torch.load('./data/eval/gpt_data.pt')
                emb_ids = embs['taxon_id'].tolist()
                keys = embs['keys']
                embs = embs['data']
                loc_emb = model.pos_enc(loc_feat)
            else:
                raise NotImplementedError('Eval for zero-shot not implemented')
        if self.eval_params['num_samples'] == -1:
            loc_emb = model.pos_enc(loc_feat)
        for tt_id, tt in tqdm(enumerate(self.taxa)):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) == 0 and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
            else:
                if self.eval_params['num_samples'] == -1:
                    gt = torch.zeros(obs_locs.shape[0], dtype=torch.float32, device=self.eval_params['device'])
                    gt[self.data['taxa_presence'][str(tt)]] = 1.0
                    species_w = model.species_params[self.train_params['class_to_taxa'].index(tt)]
                    preds = loc_emb @ species_w.detach()
                    results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt, preds).item()
                    continue
                # extract model predictions for current taxa from prediction matrix
                if 'HyperNet' not in self.train_params['model'] and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                    pred = pred_mtx[:, tt_id]
                elif (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                    if self.train_params['model'] == 'ResidualFCNet':
                        X = torch.cat([pos_examples[tt], neg_examples], dim=0)
                        y = torch.zeros(X.shape[0], dtype=torch.long)
                        y[:pos_examples[tt].shape[0]] = 1

                        #clf = LogisticRegression(class_weight='balanced', fit_intercept=False, C=0.05, max_iter=200, random_state=0).fit(X.numpy(), y.numpy())

                        C = 0.05
                        w = torch.nn.Parameter(torch.zeros(X.shape[1], 1))
                        opt = torch.optim.Rprop([w], lr=0.001)
                        crit = torch.nn.BCEWithLogitsLoss()
                        crit2 = torch.nn.MSELoss()
                        with torch.set_grad_enabled(True):
                            for i in range(40):
                                opt.zero_grad()
                                output = X @ w
                                yhat = y.float()[:, None]
                                loss = 0.5 * crit(output[yhat == 0], yhat[yhat == 0]) + 0.5 * crit(output[yhat == 1],
                                                                                                   yhat[
                                                                                                       yhat == 1]) + 1 / (
                                                   C * len(pos_examples[tt])) * crit2(w, 0 * w)
                                loss.backward()
                                opt.step()

                        pred = torch.sigmoid(((loc_emb @ w.cuda()))).flatten()
                        #pred = torch.from_numpy(clf.predict_proba(loc_emb.cpu()))[:,1]
                        #pred = torch.sigmoid(((loc_emb @ (torch.from_numpy(clf.coef_).cuda().float().T)) + torch.from_numpy(clf.intercept_).cuda().float()).squeeze(-1))
                        #locs = torch.from_numpy(utils.coord_grid((1000,2000))).to(self.eval_params['device'])
                        #locs = model(enc.encode(locs), return_feats=True)
                        #img = torch.sigmoid(((locs @ (torch.from_numpy(clf.coef_).cuda().float().T)) + torch.from_numpy(clf.intercept_).cuda().float()).squeeze(-1))
                        #plt.imshow(img.detach().cpu())
                    elif self.train_params['model'] == 'HyperNet':
                        if tt not in emb_ids:
                            results['per_species_average_precision_all'][tt_id] = 0.0
                            continue
                        gt = torch.zeros(obs_locs.shape[0], dtype=torch.float32, device=self.eval_params['device'])
                        gt[self.data['taxa_presence'][str(tt)]] = 1.0
                        if self.eval_params['num_samples'] == -1:
                            species_w = model.species_params[self.train_params['class_to_taxa'].index(tt)]
                            preds = loc_emb @ species_w.detach()
                            results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt,preds).item()
                            continue
                        with torch.no_grad():
                            if tt in emb_ids:
                                em = embs
                                emi = emb_ids
                                ky = keys
                            else:
                                results['per_species_average_precision_all'][tt_id] = 0.0
                                continue
                            sec_ind = emi.index(tt)
                            sections = [i for i,x in enumerate(ky) if x[0] == sec_ind]
                            order = ['distribution', 'range', 'text']
                            best_section = None
                            order_ind = 0
                            while best_section is None and order_ind < len(order):
                                for section in sections:
                                    if order[order_ind] in ky[section][1].lower():
                                        best_section = section
                                        break
                                order_ind += 1
                            gt = torch.zeros(obs_locs.shape[0], dtype=torch.float32, device=self.eval_params['device'])
                            gt[self.data['taxa_presence'][str(tt)]] = 1.0
                            def get_feat(x):
                                species = model.species_enc(model.species_emb.zero_shot(x))
                                species_w, species_b = species[..., :-1], species[..., -1:]
                                if self.eval_params['num_samples'] == 0:
                                    out = loc_emb @ (species_w.detach()).T
                                    return out

                                X = torch.cat([pos_examples[tt], neg_examples], dim=0).to(self.eval_params['device'])
                                y = torch.zeros(X.shape[0], dtype=torch.long).to(self.eval_params['device'])
                                y[:pos_examples[tt].shape[0]] = 1
                                C = 0.05

                                w = torch.nn.Parameter(torch.zeros_like(species_w))
                                opt = torch.optim.Rprop([w], lr=0.001)
                                crit = torch.nn.BCEWithLogitsLoss()
                                crit2 = torch.nn.MSELoss()
                                with torch.set_grad_enabled(True):
                                    for i in range(40):
                                        opt.zero_grad()
                                        output = (X @ (w + species_w.detach()).T) + 0*species_b.squeeze(-1)
                                        yhat = y.float()[:, None].repeat(1, w.shape[0])
                                        loss = 0.5 * crit(output[yhat == 0], yhat[yhat == 0]) + 0.5 * crit(
                                            output[yhat == 1], yhat[yhat == 1]) + 1 / (
                                                           C * len(pos_examples[tt])) * crit2(w, 0 * w)

                                        loss.backward()
                                        opt.step()
                                        '''out = loc_emb @ (w.data + species_w.detach()).T
                                        gt = torch.zeros(out.shape[0], dtype=torch.float32,
                                                         device=self.eval_params['device'])
                                        gt[self.data['taxa_presence'][str(tt)]] = 1.0
                                        print(utils.average_precision_score_fasterer(gt, out[:, 0]).item())'''

                                out = loc_emb @ (w.data + species_w.detach()).T
                                out = (out + 0*species_b.squeeze(-1))
                                return out
                            # average precision score:
                            yfeats = torch.cat([em[section][None].to(self.eval_params['device']) for section in sections])
                            preds = get_feat(yfeats)
                            if len(sections) > 1:#'habitat', 'overview_summary'
                                kws = [self.eval_params['text_section']] if len(ky) == len(keys) else ['text', 'range','distribution','habitat']
                                best_sections = [i for i,s in enumerate(sections) if any((x in ky[s][1].lower() for x in kws))]
                                #yfeats2 = torch.cat(
                                #    [em[section][None].to(self.eval_params['device']) for section in best_sections]).mean(dim=0, keepdim=True)
                                #pred2 = get_feat(yfeats2)
                                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt, preds[:, best_sections].mean(dim=1)).item()
                            else:
                                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt, preds[:, 0]).item()
                            continue
                else:
                    if tt_id % 32 == 0:
                        preds = torch.empty(loc_feat.shape[0], classes_of_interest[tt_id:tt_id+32].shape[0], device=self.eval_params['device'])
                        for i in range(0,preds.shape[0],50000):
                            xbatch = loc_feat[i:i+50000]
                            ybatch = classes_of_interest[tt_id:tt_id+32].to(self.eval_params['device']).expand(xbatch.shape[0], -1)
                            preds[i:i+50000] = model(xbatch, ybatch)
                    pred = preds[:,tt_id%32]
                gt = torch.zeros(obs_locs.shape[0], dtype=torch.float32, device=self.eval_params['device'])
                gt[self.data['taxa_presence'][str(tt)]] = 1.0
                # average precision score:
                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt, pred).item()

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)
        return results

    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')


class EvaluatorGeoPrior:

    def __init__(self, train_params, eval_params):
        # store parameters:
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        # load vision model predictions:
        self.data = np.load(os.path.join(paths['geo_prior'], 'geo_prior_model_preds.npz'))
        print(self.data['probs'].shape[0], 'total test observations')
        # load locations:
        meta = pd.read_csv(os.path.join(paths['geo_prior'], 'geo_prior_model_meta.csv'))
        self.obs_locs  = np.vstack((meta['longitude'].values, meta['latitude'].values)).T.astype(np.float32)
        temp = np.array(meta['observed_on'].values, dtype='S10')
        temp = temp.view('S1').reshape((temp.size, -1))
        years = temp[:, :4].view('S4').astype(int)[:, 0]
        months = temp[:, 5:7].view('S2').astype(int)[:, 0]
        days = temp[:, 8:10].view('S2').astype(int)[:, 0]
        days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
        dates = days_per_month[months - 1] + days - 1
        self.dates = np.round((dates) / 365.0, 4).astype(np.float32)
        # taxonomic mapping:
        self.taxon_map = self.find_mapping_between_models(self.data['model_to_taxa'], self.train_params['class_to_taxa'])
        print(self.taxon_map.shape[0], 'out of', len(self.data['model_to_taxa']), 'taxa in both vision and geo models')

        cs = torch.load('class_counts.pt')
        cs = cs.sum() / cs
        cs = cs.to(self.eval_params['device'])
        self.C = cs[None]
        self.pdf = utils.DataPDFH3(device=self.eval_params['device'])

    def find_mapping_between_models(self, vision_taxa, geo_taxa):
        # this will output an array of size N_overlap X 2
        # the first column will be the indices of the vision model, and the second is their
        # corresponding index in the geo model
        taxon_map = np.ones((vision_taxa.shape[0], 2), dtype=np.int32)*-1
        taxon_map[:, 0] = np.arange(vision_taxa.shape[0])
        geo_taxa_arr = np.array(geo_taxa)
        for tt_id, tt in enumerate(vision_taxa):
            ind = np.where(geo_taxa_arr==tt)[0]
            if len(ind) > 0:
                taxon_map[tt_id, 1] = ind[0]
        inds = np.where(taxon_map[:, 1]>-1)[0]
        taxon_map = taxon_map[inds, :]
        return taxon_map

    def convert_to_inat_vision_order(self, geo_pred_ip, vision_top_k_prob, vision_top_k_inds, vision_taxa, taxon_map, k=1.0):
        # this is slow as we turn the sparse input back into the same size as the dense one
        vision_pred = np.zeros((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        geo_pred = k*np.ones((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        vision_pred[np.arange(vision_pred.shape[0])[..., np.newaxis], vision_top_k_inds] = vision_top_k_prob

        geo_pred[:, taxon_map[:, 0]] = geo_pred_ip[:, taxon_map[:, 1]]

        return geo_pred, vision_pred

    def run_evaluation(self, model, enc, extra_input=None):
        results = {}

        # loop over in batches
        batch_start = np.hstack((np.arange(0, self.data['probs'].shape[0], self.eval_params['batch_size']), self.data['probs'].shape[0]))
        correct_pred = np.zeros(self.data['probs'].shape[0])
        from tqdm import tqdm
        for bb_id, bb in tqdm(enumerate(range(len(batch_start)-1))):
            batch_inds = np.arange(batch_start[bb], batch_start[bb+1])

            vision_probs = self.data['probs'][batch_inds, :]
            vision_inds = self.data['inds'][batch_inds, :]
            gt = self.data['labels'][batch_inds]
            dates = torch.from_numpy(self.dates[batch_inds])

            obs_locs_batch = torch.from_numpy(self.obs_locs[batch_inds, :]).to(self.eval_params['device'])
            noise_level = 1.0
            loc_feat = torch.cat([enc.encode(obs_locs_batch), extra_input], 1) if extra_input is not None else enc.encode(obs_locs_batch)

            with torch.no_grad():
                geo_pred = model(loc_feat).cpu().numpy()

            geo_pred, vision_pred = self.convert_to_inat_vision_order(geo_pred, vision_probs, vision_inds,
                                                                       self.data['model_to_taxa'], self.taxon_map, k=1.0)
            #geo_pred = softmax(torch.from_numpy(geo_pred), dim=1).numpy()
            comb_pred = np.argmax(vision_pred*geo_pred, 1)
            comb_pred = (comb_pred==gt)
            correct_pred[batch_inds] = comb_pred
        accuracy_by_taxa = np.zeros(len(self.data['model_to_taxa']))
        for tt_id, tt in enumerate(self.data['model_to_taxa']):
            inds = np.where(self.data['labels'] == tt)[0]
            accuracy_by_taxa[tt_id] = float((correct_pred[inds].mean()))
        torch.save(correct_pred, f'correct_{noise_level}.pt')
        torch.save(accuracy_by_taxa, f'abt_{noise_level}.pt')
        results['vision_only_top_1'] = float((self.data['inds'][:, -1] == self.data['labels']).mean())
        results['vision_geo_top_1'] = float(correct_pred.mean())
        return results

    def report(self, results):
        print('Overall accuracy vision only model', round(results['vision_only_top_1'], 3))
        print('Overall accuracy of geo model     ', round(results['vision_geo_top_1'], 3))
        print('Gain                              ', round(results['vision_geo_top_1'] - results['vision_only_top_1'], 3))

class EvaluatorGeoFeature:

    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        self.data_path = paths['geo_feature']
        self.country_mask = tifffile.imread(os.path.join(paths['masks'], 'USA_MASK.tif')) == 1
        self.raster_names = ['ABOVE_GROUND_CARBON', 'ELEVATION', 'LEAF_AREA_INDEX', 'NON_TREE_VEGITATED', 'NOT_VEGITATED', 'POPULATION_DENSITY', 'SNOW_COVER', 'SOIL_MOISTURE', 'TREE_COVER']
        self.raster_names_log_transform = ['POPULATION_DENSITY']

    def load_raster(self, raster_name, log_transform=False):
        raster = tifffile.imread(os.path.join(self.data_path, raster_name + '.tif')).astype(np.float32)
        valid_mask = ~np.isnan(raster).copy() & self.country_mask
        # log scaling:
        if log_transform:
            raster[valid_mask] = np.log1p(raster[valid_mask] - raster[valid_mask].min())
        # 0/1 scaling:
        raster[valid_mask] -= raster[valid_mask].min()
        raster[valid_mask] /= raster[valid_mask].max()

        return raster, valid_mask

    def get_split_labels(self, raster, split_ids, split_of_interest):
        # get the GT labels for a subset
        inds_y, inds_x = np.where(split_ids==split_of_interest)
        return raster[inds_y, inds_x]

    def get_split_feats(self, model, enc, split_ids, split_of_interest, extra_input=None):
        locs = utils.coord_grid(self.country_mask.shape, split_ids=split_ids, split_of_interest=split_of_interest)
        locs = torch.from_numpy(locs).to(self.eval_params['device'])
        locs_enc = torch.cat([enc.encode(locs), extra_input.expand(locs.shape[0], -1)], 1) if extra_input is not None else enc.encode(locs)
        with torch.no_grad():
            feats = model(locs_enc, return_feats=True).cpu().numpy()
        return feats

    def run_evaluation(self, model2, enc, extra_input=None):
        if self.train_params['model'] == 'ResidualFCNet':
            model = model2
        elif self.train_params['model'] == 'HyperNet':
            model = lambda x, return_feats=True: model2.pos_enc(x)
        else:
            raise NotImplementedError()
        results = {}
        for raster_name in self.raster_names:
            do_log_transform = raster_name in self.raster_names_log_transform
            raster, valid_mask = self.load_raster(raster_name, do_log_transform)
            split_ids = utils.create_spatial_split(raster, valid_mask, cell_size=self.eval_params['cell_size'])
            feats_train = self.get_split_feats(model, enc, split_ids=split_ids, split_of_interest=1, extra_input=extra_input)
            feats_test = self.get_split_feats(model, enc, split_ids=split_ids, split_of_interest=2, extra_input=extra_input)
            labels_train = self.get_split_labels(raster, split_ids, 1)
            labels_test = self.get_split_labels(raster, split_ids, 2)
            scaler = MinMaxScaler()
            feats_train_scaled = scaler.fit_transform(feats_train)
            feats_test_scaled = scaler.transform(feats_test)
            clf = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10, fit_intercept=True, scoring='r2').fit(feats_train_scaled, labels_train)
            train_score = clf.score(feats_train_scaled, labels_train)
            test_score = clf.score(feats_test_scaled, labels_test)
            results[f'train_r2_{raster_name}'] = float(train_score)
            results[f'test_r2_{raster_name}'] = float(test_score)
            results[f'alpha_{raster_name}'] = float(clf.alpha_)
        return results

    def report(self, results):
        report_fields = [x for x in results if 'test_r2' in x]
        for field in report_fields:
            print(f'{field}: {results[field]}')
        print(np.mean([results[field] for field in report_fields]))

def launch_eval_run(overrides):

    eval_params = setup.get_default_params_eval(overrides)

    # set up model:
    eval_params['model_path'] = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], eval_params['ckp_name'])
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    default_params = setup.get_default_params_train()
    for key in default_params:
        if key not in train_params['params']:
            train_params['params'][key] = default_params[key]
    model = models.get_model(train_params['params'], inference_only=True)
    model.load_state_dict(train_params['state_dict'], strict=False)
    model = model.to(eval_params['device'])
    model.eval()

    # create input encoder:
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env', 'sh_env', 'lonlat_env']:
        raster = datasets.load_env().to(eval_params['device'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster, input_dim=train_params['params']['input_dim'])


    print('\n' + eval_params['eval_type'])
    t = time.time()
    if eval_params['eval_type'] == 'snt':
        eval_params['split'] = 'test' # val, test, all
        eval_params['val_frac'] = 0.50
        eval_params['split_seed'] = 7499
        evaluator = EvaluatorSNT(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'iucn':
        evaluator = EvaluatorIUCN(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'geo_prior':
        evaluator = EvaluatorGeoPrior(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'geo_feature':
        evaluator = EvaluatorGeoFeature(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    else:
        raise NotImplementedError('Eval type not implemented.')
    print(f'evaluation completed in {np.around((time.time()-t)/60, 1)} min')
    return results