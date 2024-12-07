import torch
import utils
from torch.nn.functional import logsigmoid


def get_loss_function(params):
    if params['loss'] == 'an_full':
        return an_full
    elif params['loss'] == 'an_slds':
        return an_slds
    elif params['loss'] == 'an_ssdl':
        return an_ssdl
    elif params['loss'] == 'an_full_me':
        return an_full_me
    elif params['loss'] == 'an_slds_me':
        return an_slds_me
    elif params['loss'] == 'an_ssdl_me':
        return an_ssdl_me
    elif params['loss'] == 'an_full_hypernet':
        return an_full_hypernet
    elif params['loss'] == 'an_full_hypernet_geoprior':
        return an_full_hypernet_geoprior
    elif params['loss'] == 'contrastive_hypernet_geoprior':
        return contrastive_hypernet_geoprior
    elif params['loss'] == 'contrastive_hypernet_geoprior_v2':
        return contrastive_hypernet_geoprior_v2


def neg_log(x):
    return -torch.log(x + 1e-5)

def neg_log_sig(x):
    return -torch.nn.functional.logsigmoid(x)


def bernoulli_entropy(p):
    entropy = p * neg_log(p) + (1 - p) * neg_log(1 - p)
    return entropy


def an_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id])  # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id])  # entropy
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_slds(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    loc_emb = model(loc_feat, return_feats=True)

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))

    num_classes = loc_pred.shape[1]
    bg_class = torch.randint(low=0, high=num_classes - 1, size=(batch_size,), device=params['device'])
    bg_class[bg_class >= class_id[:batch_size]] += 1

    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred[inds[:batch_size], bg_class])  # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class])  # entropy
    else:
        raise NotImplementedError

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_full(batch, model, params, loc_to_feats, neg_type='hard'):
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    # get predictions for locations and background locations
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))[:params['num_classes']]
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))[:params['num_classes']]

    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log(1.0 - loc_pred)  # assume negative
        loss_bg = neg_log(1.0 - loc_pred_rand)  # assume negative
    elif neg_type == 'entropy':
        loss_pos = -1 * bernoulli_entropy(1.0 - loc_pred)  # entropy
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand)  # entropy
    else:
        raise NotImplementedError
    loss_pos[inds[:batch_size], class_id] = params['pos_weight'] * neg_log(loc_pred[inds[:batch_size], class_id])

    # total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss


def an_full_me(batch, model, params, loc_to_feats):
    return an_full(batch, model, params, loc_to_feats, neg_type='entropy')


def an_ssdl_me(batch, model, params, loc_to_feats):
    return an_ssdl(batch, model, params, loc_to_feats, neg_type='entropy')


def an_slds_me(batch, model, params, loc_to_feats):
    return an_slds(batch, model, params, loc_to_feats, neg_type='entropy')


def an_full_hypernet(batch, model, params, loc_to_feats, neg_type='hard', class_samples=192):
    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    batch_size = loc_feat.shape[0]

    class_id_real = torch.randint(0,params['num_classes']-1, size=(batch_size, class_samples-1), device=params['device'])
    class_id_real[class_id_real >= class_id[:,None]] += 1
    class_id_real = torch.cat([class_id[:,None], class_id_real], dim=1)
    class_id_fake = torch.randint(0,params['num_classes'], size=(batch_size, class_samples), device=params['device'])
    class_id_cat = torch.cat([class_id_real, class_id_fake], 0)

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, class_id_cat)
    loc_pred = loc_emb_cat[:batch_size, :]
    loc_pred_rand = loc_emb_cat[batch_size:, :]

    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log_sig(-loc_pred)*(params['num_classes']-1)/(class_samples-1)  # assume negative
        loss_bg = neg_log_sig(-loc_pred_rand)*(params['num_classes']/class_samples)  # assume negative
    else:
        raise NotImplementedError
    loss_pos[:, 0] = params['pos_weight'] * neg_log_sig(loc_pred[:, 0])

    # total loss
    loss = loss_pos.sum()/params['num_classes']/loss_pos.shape[0] + loss_bg.sum()/params['num_classes']/loss_bg.shape[0]

    return loss


def an_full_hypernet_geoprior(batch, model, params, loc_to_feats, neg_type='hard', class_samples=192):
    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    batch_size = loc_feat.shape[0]

    class_id_real = torch.randint(0,params['num_classes']-1, size=(batch_size, class_samples-1), device=params['device'])
    class_id_real[class_id_real >= class_id[:,None]] += 1
    class_id_real = torch.cat([class_id[:,None], class_id_real], dim=1)
    class_id_fake = torch.randint(0,params['num_classes'], size=(batch_size, class_samples), device=params['device'])
    class_id_cat = torch.cat([class_id_real, class_id_fake], 0)

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat, loc_emb_cat2, species_sim = model(loc_cat, class_id_cat)
    s1, s2, uniq = species_sim
    s1 = s1/torch.norm(s1, dim=1, keepdim=True)
    s2 = s2/torch.norm(s2, dim=1, keepdim=True)
    imap = torch.zeros(uniq.max()+1, dtype=int, device=uniq.device)
    imap[uniq] = torch.arange(len(uniq), device=uniq.device)
    loc_pred = loc_emb_cat[:batch_size, :]
    loc_pred_rand = loc_emb_cat[batch_size:, :]
    loc_pred2 = loc_emb_cat2[:batch_size, :]
    loc_pred_rand2 = loc_emb_cat2[batch_size:, :]

    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log_sig(-loc_pred)*(params['num_classes']-1)/(class_samples-1)  # assume negative
        loss_bg = neg_log_sig(-loc_pred_rand)*(params['num_classes']/class_samples)  # assume negative
        loss_pos2 = neg_log_sig(-loc_pred2)*(params['num_classes']-1)/(class_samples-1)  # assume negative
        loss_bg2 = neg_log_sig(-loc_pred_rand2)*(params['num_classes']/class_samples)  # assume negative

        loss3 = 0
        for x,y in [(s1, s2), (s2, s1)]:
            l3 = (x[imap[class_id_real[:,:1]]] * y[imap[class_id_real]]).sum(dim=-1)
            l3[:,0] *= -1
            l3 = neg_log_sig(-params['geoprior_temp']*l3)
            loss3 += 0.5*(l3[:,0].mean() + l3[:,1:].mean())
    else:
        raise NotImplementedError
    loss_pos[:, 0] = params['pos_weight'] * neg_log_sig(loc_pred[:, 0])
    loss_pos2[:, 0] = params['pos_weight'] * neg_log_sig(loc_pred2[:, 0])
    # total loss
    loss = loss_pos.sum()/params['num_classes']/loss_pos.shape[0] + loss_bg.sum()/params['num_classes']/loss_bg.shape[0]
    loss2 = loss_pos2.sum()/params['num_classes']/loss_pos2.shape[0] + loss_bg2.sum()/params['num_classes']/loss_bg2.shape[0]

    return loss+loss2+loss3


def contrastive_hypernet_geoprior(batch, model, params, loc_to_feats):
    crit = torch.nn.CrossEntropyLoss()
    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    batch_size = loc_feat.shape[0]

    ys, indmap = torch.unique(class_id, return_inverse=True)
    species = model.species_enc(model.species_emb(ys))
    species_w, species_b = species[..., :-1], species[..., -1:]
    species_w = species_w/species_w.norm(dim=1, keepdim=True)
    species_w2 = model.species_params[ys][indmap]
    pos = model.pos_enc(loc_feat)
    pos = pos/(pos.norm(dim=1, keepdim=True))
    out1 = model.temp.exp() * species_w[indmap] @ pos.T
    out2 = out1.T
    out3 = model.temp.exp() * species_w2[indmap] @ pos.T
    out4 = out3.T
    labels = torch.arange(0, batch_size, device=params['device'])

    # total loss
    loss = 0.5*(crit(out1, labels) + crit(out2, labels))
    loss2 = 0.5*(crit(out3, labels) + crit(out4, labels))
    return loss+loss2


def contrastive_hypernet_geoprior_v2(batch, model, params, loc_to_feats):
    crit = torch.nn.CrossEntropyLoss()
    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    batch_size = loc_feat.shape[0]

    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    ys, indmap = torch.unique(class_id, return_inverse=True)
    species = model.species_enc(model.species_emb(ys))
    species_w, species_b = species[..., :-1], species[..., -1:]
    species_w = species_w/species_w.norm(dim=1, keepdim=True)
    species_w2 = model.species_params[ys][indmap]
    pos = model.pos_enc(loc_feat)
    pos = pos/(pos.norm(dim=1, keepdim=True))
    pos_rand = model.pos_enc(rand_feat)
    pos_rand = pos_rand/(pos_rand.norm(dim=1, keepdim=True))
    pos_all = torch.cat([pos, pos_rand], dim=0)
    out1 = model.temp.exp() * species_w[indmap] @ pos_all.T
    out2 = out1.T[:batch_size]
    out3 = model.temp.exp() * species_w2[indmap] @ pos_all.T
    out4 = out3.T[:batch_size]
    labels = torch.arange(0, batch_size, device=params['device'])

    # total loss
    loss = 0.5*(crit(out1, labels) + crit(out2, labels))
    loss2 = 0.5*(crit(out3, labels) + crit(out4, labels))
    return loss+loss2