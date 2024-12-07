import copy
import os
import shutil

import numpy as np
import torch
import setup
import losses
import models
import datasets
import utils
from tqdm import tqdm

class Trainer():

    def __init__(self, model, train_loader, params):

        self.params = params

        # define loaders:
        self.train_loader = train_loader

        # define model:
        self.model = model

        # define important objects:
        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = self.train_loader.dataset.enc.encode

        # define optimization objects:
        if params['finetune_path'] != '':
            ckpt = torch.load(os.path.join(params['save_base'], params['finetune_path']))['state_dict']
            del ckpt['species_params']
            del ckpt['species_emb.species_emb.weight']
            self.model.load_state_dict(ckpt, strict=False)
            for k,v in self.model.named_parameters():
                if k != 'species_params':
                    v.requires_grad = False
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], params['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])

    def train_one_epoch(self):

        self.model.train()
        # initialise run stats
        running_loss = 0.0
        samples_processed = 0
        steps_trained = 0
        for _, batch in tqdm(enumerate(self.train_loader)):
            # reset gradients:
            self.optimizer.zero_grad()
            # compute loss:
            batch_loss = self.compute_loss(batch, self.model, self.params, self.encode_location)
            # backwards pass:
            batch_loss.backward()
            # update parameters:
            self.optimizer.step()
            # track and report:
            running_loss += float(batch_loss.item())
            steps_trained += 1
            samples_processed += batch[0].shape[0]
            if steps_trained % self.params['log_frequency'] == 0:
                print(f'[{samples_processed}/{len(self.train_loader.dataset)}] loss: {np.around(running_loss / self.params["log_frequency"], 4)}')
                running_loss = 0.0
        # update learning rate according to schedule:
        self.lr_scheduler.step()

    def save_model(self, postfix=''):
        save_path = os.path.join(self.params['save_path'], f'model{postfix}.pt')
        op_state = {'state_dict': self.model.state_dict(), 'params' : self.params}
        torch.save(op_state, save_path)


class ConstrastiveTrainer():
    def __init__(self, model, train_loader, params):
        self.params = params

        # define loaders:
        self.train_loader = train_loader

        # define model:
        self.model = model
        self.ema_model = copy.deepcopy(model)

        # define important objects:
        loss_dict = {'clip': losses.clip_loss}
        self.compute_loss = loss_dict[params['loss']]
        self.encode_location = self.train_loader.dataset.enc.encode

        # define optimization objects:
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": params['weight_decay'],
                },
            ],
            lr=params['lr']
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])


    def update_ema_variables(self, ema_decay):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.copy_(ema_param.data * ema_decay + (1 - ema_decay) * param.data)

    def eval(self, use_ema=False):
        if not hasattr(self, 'eval_data'):
            import json
            # load h3 data:
            with open('paths.json', 'r') as f:
                paths = json.load(f)
            with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
            device = self.train_loader.dataset.enc.raster.device
            self.eval_data = torch.from_numpy(np.array(data['locs'], dtype=np.float32)).to(device)
            self.eval_data_encoded = self.encode_location(self.eval_data)
            mask = torch.sum(self.eval_data_encoded == 0, dim=1) == 0
            self.eval_data = self.eval_data[mask]
            self.eval_data_encoded = self.eval_data_encoded[mask]
            perm = torch.randperm(self.eval_data.shape[0], generator=torch.Generator().manual_seed(42))
            self.eval_data = self.eval_data[perm]
            self.eval_data_encoded = self.eval_data_encoded[perm]
        model = self.model if not use_ema else self.ema_model
        model.eval()
        with torch.no_grad():
            loss = 0
            for i in range(0, self.eval_data.shape[0], 2048):
                ed = self.eval_data[i:i+2048]
                ede = self.eval_data_encoded[i:i+2048]
                loss += ed.shape[0]*self.compute_loss((ede, ed, None), model, self.params)
            loss /= self.eval_data.shape[0]
        return loss.item()

    def train_one_epoch(self):
        self.model.train()
        # initialise run stats
        running_loss = 0.0
        samples_processed = 0
        steps_trained = 0
        for batch in tqdm(self.train_loader):
            # reset gradients:
            self.optimizer.zero_grad()
            # compute loss:
            batch_loss = self.compute_loss(batch, self.model, self.params)
            # backwards pass:
            batch_loss.backward()
            # update parameters:
            self.optimizer.step()
            self.update_ema_variables(self.params['ema'])
            # track and report:
            running_loss += float(batch_loss.item())
            steps_trained += 1
            samples_processed += batch[0].shape[0]
            if steps_trained % self.params['log_frequency'] == 0:
                print(f'[{samples_processed}/{len(self.train_loader.dataset)}] loss: {np.around(running_loss / self.params["log_frequency"], 4)}')
                running_loss = 0.0
        # update learning rate according to schedule:
        self.lr_scheduler.step()

    def save_model(self, postfix=''):
        save_path = os.path.join(self.params['save_path'], f'model{postfix}.pt')
        op_state = {'state_dict': self.model.state_dict(), 'params' : self.params}
        torch.save(op_state, save_path)


def launch_training_run(ovr):
    # setup:
    params = setup.get_default_params_train(ovr)
    params['save_path'] = os.path.join(params['save_base'], params['experiment_name'])
    if params['timestamp']:
        params['save_path'] = params['save_path'] + '_' + utils.get_time_stamp()
    try:
        os.makedirs(params['save_path'], exist_ok=False)
    except:
        print('Experiment already exists')
        exit(0)
        shutil.rmtree(params['save_path'])
        os.makedirs(params['save_path'], exist_ok=False)

    # data:
    train_dataset = datasets.get_train_data(params)
    params['input_dim'] = train_dataset.input_dim
    params['num_classes'] = train_dataset.num_classes
    params['class_to_taxa'] = train_dataset.class_to_taxa
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=3,
        collate_fn=getattr(train_dataset, 'collate_fn', None))

    # model:
    model = models.get_model(params)
    model = model.to(params['device'])

    # train:
    trainer = Trainer(model, train_loader, params)
    for epoch in range(0, params['num_epochs']):
        print(f'epoch {epoch+1}')
        if epoch > 0 and epoch % params['save_frequency'] == 0:
            trainer.save_model(postfix=f'_{epoch}')
        trainer.train_one_epoch()
    trainer.save_model()


def launch_contrastive_training_run(ovr):
    from viz_ica import viz_ica
    # setup:
    params = setup.get_default_params_contrastive(ovr)
    params['save_path'] = os.path.join(params['save_base'], params['experiment_name'])
    if params['timestamp']:
        params['save_path'] = params['save_path'] + '_' + utils.get_time_stamp()
    try:
        os.makedirs(params['save_path'], exist_ok=False)
    except:
        print('Already Exists')

    # data:
    train_dataset = datasets.get_train_data(params)
    params['input_dim'] = train_dataset.input_dim
    params['num_classes'] = train_dataset.num_classes
    params['class_to_taxa'] = train_dataset.class_to_taxa
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=getattr(train_dataset, 'collate_fn', None))

    # model:
    pos_enc = utils.CoordEncoder('sin_cos', input_dim=params['input_dim'])
    model = models.ClipModel(lambda x: train_dataset.enc.encode(x, normalize=False),
                             lambda x: pos_enc.encode(x, normalize=False),
                             params['input_dim'], num_classes=params['proj_dim'], num_hidden=params['num_filts'],
                             depth=params['depth'], depth_feat=params['depth_feat'])
    model = model.to(params['device'])

    # train:
    trainer = ConstrastiveTrainer(model, train_loader, params)
    performance = []
    performance2 = []
    performance.append(trainer.eval())
    performance2.append(trainer.eval(use_ema=True))
    for epoch in range(0, params['num_epochs']):
        print(performance[-1], performance2[-1])
        #viz_ica(model.pos_model, input_dim=8)
        print(f'epoch {epoch+1}')
        trainer.train_one_epoch()
        performance.append(trainer.eval())
        performance2.append(trainer.eval(use_ema=True))
        if min(performance) == performance[-1]:
            trainer.save_model(postfix='')
    return performance2