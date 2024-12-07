import os
import numpy as np
import torch

import train
import eval

train_params = {}

train_params['experiment_name'] = f'lesinr_posenv' # This will be the name of the directory where results for this run are saved.

'''
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds'
'''
train_params['species_set'] = 'all'

'''
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
'''
train_params['hard_cap_num_per_class'] = 1000

'''
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
'''
train_params['num_aux_species'] = 0

'''
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env'
'''
train_params['input_enc'] = 'sin_cos_env'
train_params['input_dim'] = 4
train_params['depth'] = 4

train_params['lr'] = 0.0005
train_params['model'] = 'HyperNet'
train_params['species_dim'] = 256
train_params['species_enc_depth'] = 0
train_params['species_filts'] = 0
train_params['pos_enc'] = 'FCNet'
train_params['species_enc'] = 'wiki'
train_params['text_emb_path'] = './data/train/wiki_data.pt'
train_params['text_hidden_dim'] = 512
train_params['zero_shot'] = True
train_params['text_num_layers'] = 1
train_params['num_epochs'] = 10
train_params['geoprior_temp'] = 0.0

'''
loss
- Which loss to use for training.
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me', 'an_full_hypernet'
'''
train_params['loss'] = 'an_full_hypernet_geoprior'

# train:
train.launch_training_run(train_params)

# evaluate:
for eval_type in ['iucn', 'snt']:
    for text_section in ['habitat', 'range']:
        for num_samp in [0,1,2,3,5,10,100,1000]:
            eval_params = {}
            eval_params['exp_base'] = './experiments'
            eval_params['experiment_name'] = train_params['experiment_name']
            eval_params['eval_type'] = eval_type
            eval_params['input_dim'] = train_params['input_dim']
            eval_params['ckp_name'] = 'model.pt'
            eval_params['num_samples'] = num_samp
            eval_params['text_section'] = text_section
            eval_params['extract_pos'] = False
            cur_results = eval.launch_eval_run(eval_params)
            np.save(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)

'''
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
'''
