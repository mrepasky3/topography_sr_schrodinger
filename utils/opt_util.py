import ml_collections

import torch
import torch.nn.functional as F

from bridge.schrodinger_bridge import compute_label


def save_checkpoint(
	network,
	params,
	epoch,
	savepath
	):

	state_dict = network.state_dict()
	for i, (name, _value) in enumerate(network.named_parameters()):
		assert name in state_dict
		state_dict[name] = params[i]
	filename = "net_ckpt_epoch{}".format(epoch)
	torch.save(state_dict, savepath+"/"+filename)


def get_batch(
	batch,
	downsampler,
	upsampler,
	vae,
	device,
	height_scaling=1000.,
	):

	batch_dict = ml_collections.ConfigDict()

	batch_dict.dems_batch = (batch['high_res_dem'].unsqueeze(1)/height_scaling).to(device)
	batch_dict.latent_dems_batch = vae.module.encode_sample(batch_dict.dems_batch)

	batch_dict.cond_dems_batch = downsampler(batch_dict.dems_batch)
	batch_dict.cond_dems_batch_upsampled = upsampler(batch_dict.cond_dems_batch)
	batch_dict.latent_cond_dems_batch = vae.module.encode_sample(batch_dict.cond_dems_batch_upsampled)

	imgs_batch = (batch['imgs'].unsqueeze(2)).to(device)
	batch_size, set_size, _, _, _ = imgs_batch.shape
	batch_dict.masks_batch = ((imgs_batch.view(batch_size, set_size, -1).max(axis=-1).values - imgs_batch.view(batch_size, set_size, -1).min(axis=-1).values) > 0).float()
	batch_dict.imgs_batch = imgs_batch

	return batch_dict


def get_repeated_batch(
	n_repeat,
	batch,
	downsampler,
	upsampler,
	vae,
	device,
	bounded=False,
	img_bound=None,
	height_scaling=1000.
	):

	batch_dict = ml_collections.ConfigDict()

	batch_dict.dems_batch = (batch['high_res_dem'].unsqueeze(1)/height_scaling).to(device)
	batch_dict.latent_dems_batch = vae.module.encode_sample(batch_dict.dems_batch)

	batch_dict.cond_dems_batch = downsampler(batch_dict.dems_batch)
	batch_dict.cond_dems_batch_upsampled = upsampler(batch_dict.cond_dems_batch)
	batch_dict.latent_cond_dems_batch = vae.module.encode_sample(batch_dict.cond_dems_batch_upsampled)

	imgs_batch = (batch['imgs'].unsqueeze(2)).to(device)
	batch_size, set_size, _, _, _ = imgs_batch.shape
	batch_dict.masks_batch = ((imgs_batch.view(batch_size, set_size, -1).max(axis=-1).values - imgs_batch.view(batch_size, set_size, -1).min(axis=-1).values) > 0).float()
	batch_dict.imgs_batch = imgs_batch
	
	if bounded:
		maskmask = torch.zeros_like(batch_dict.masks_batch)
		maskmask_idx = torch.cat([torch.stack([k*torch.ones(img_bound,dtype=int).view(-1,1), torch.randperm(batch_dict.masks_batch.shape[-1],dtype=int)[:img_bound].view(-1,1)], dim=1).squeeze() for k in range(batch_dict.masks_batch.shape[0])])
		maskmask[maskmask_idx[:,0], maskmask_idx[:,1]] = 1
		batch_dict.masks_batch *= maskmask

	if n_repeat > 1:
		batch_dict.cond_dems_batch = batch_dict.cond_dems_batch.repeat_interleave(n_repeat, dim=0)
		batch_dict.latent_cond_dems_batch = batch_dict.latent_cond_dems_batch.repeat_interleave(n_repeat, dim=0)
		batch_dict.masks_batch = batch_dict.masks_batch.repeat_interleave(n_repeat, dim=0)
		batch_dict.imgs_batch = batch_dict.imgs_batch.repeat_interleave(n_repeat, dim=0)

	return batch_dict


def get_batch_vae(
	batch,
	device,
	height_scaling=1000.,
	):

	batch_dict = ml_collections.ConfigDict()

	batch_dict.dems_batch = (batch['high_res_dem'].unsqueeze(1)/height_scaling).to(device)

	return batch_dict


def compute_loss(
	method_config,
	network,
	batch_dict
	):

	step = torch.randint(0, method_config.interval, (batch_dict.dems_batch.shape[0],))
	
	xt = method_config.diffusion.q_sample(
		step,
		batch_dict.latent_dems_batch,
		batch_dict.latent_cond_dems_batch
	)
	
	label = compute_label(
		method_config.diffusion,
		step,
		batch_dict.latent_dems_batch,
		xt
	)

	pred = network(
		xt,
		step,
		batch_dict.cond_dems_batch,
		batch_dict.imgs_batch,
		batch_dict.masks_batch
	)

	assert xt.shape == label.shape == pred.shape

	return F.mse_loss(pred, label)