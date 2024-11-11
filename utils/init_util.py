import os
import numpy as np
import ml_collections

import torch
import torch.nn as nn
import torch.optim as optim

from .config_util import load_config
from configs.configs import get_vae_config, get_set_dit_config
from datasets.patches_dataset import PolarPatchDataset, PolarPatchRealisticSun

from latent_diffusion.autoencoder import AutoencoderKL
from diffusion_transformer.models import SetDiT

from bridge.schrodinger_bridge import Diffusion, make_beta_schedule

def get_path(args, create=False):
	
	res_range = "{}-{}MPP".format(args.resolution, args.resolution*args.downsample)
	dirname = "bridge_{}_{:03d}".format(res_range, args.model_id)
	if args.dem_dataset == "polar":
		dirname = "polar_" + dirname
	elif args.dem_dataset == "bettersun":
		dirname = "bettersun_" + dirname
	path = "results/{}".format(dirname)
	
	if create:
		if not os.path.exists("results"):
			os.mkdir("results")
		if not os.path.exists(path):
			os.mkdir(path)
		else:
			print("{} Already Exists!".format(path))
			assert False
	return path


def get_uq_path(savepath, args):
	uq_savepath = savepath+"/uq"
	if args.dataset != "test":
		uq_savepath += "_" + args.dataset
	if not os.path.exists(uq_savepath):
		os.mkdir(uq_savepath)
	return uq_savepath


def get_recon_path(savepath, args):
	recon_savepath = savepath+"/recon"
	if args.dataset != "test":
		recon_savepath += "_" + args.dataset
	if not os.path.exists(recon_savepath):
		os.mkdir(recon_savepath)
	return recon_savepath


def load_dataset(args, p_flip=0., dem_only=False):

	if args.dem_dataset == "polar":
		patch_dataset_raw = PolarPatchDataset(p_flip=p_flip)
	elif args.dem_dataset == "bettersun":
		
		if dem_only:
			num_img_vals = np.arange(5,6,1)
			missing_prob = 0.
			img_selection = "replacement"
		else:
			num_img_vals = np.arange(args.num_imgs_lower, args.num_imgs_upper+1, 1)
			missing_prob = args.missing_prob
			img_selection = args.img_selection
		
		patch_dataset_raw = PolarPatchRealisticSun(
			p_flip=p_flip,
			num_img_vals=num_img_vals,
			missing_prob=missing_prob,
			img_selection=img_selection,
			dem_only=dem_only
		)
	else:
		raise NotImplementedError("dem_dataset must be in [polar, bettersun]")

	return patch_dataset_raw


def split_dataset(patch_dataset_raw, args):

	if args.dem_dataset == "polar" or args.dem_dataset == "bettersun":
		train_idx = np.load("process_data/thresh2242_train_idx.npy")
		val_idx = np.load("process_data/thresh2242_val_idx.npy")
		test_idx = np.load("process_data/thresh2242_test_idx.npy")

		patch_dataset_train = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(train_idx))
		patch_dataset_val = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(val_idx))
		patch_dataset_test = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(test_idx))
	else:
		raise NotImplementedError("dem_dataset must be in [polar, bettersun]")

	return patch_dataset_train, patch_dataset_val, patch_dataset_test


def init_optim(model_params, config):
	optimizer = optim.AdamW(model_params, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
	return optimizer


def init_up_down_samplers(downsample, mode, inp_size):
	downsampler = nn.AvgPool2d(downsample)
	upsampler = nn.Upsample(inp_size, mode=mode)
	return downsampler, upsampler


def disabled_train(self, mode=True):
		"""Overwrite model.train with this function to make sure train/eval mode
		does not change anymore."""
		return self


def load_vae(args, device, requires_grad=False):
	# initialize the trained encoder

	vae_config = get_vae_config()

	vae = AutoencoderKL(
		in_channels=1,
		out_ch=1,
		ch=vae_config.model.base_channels,
		ch_mult=vae_config.model.channel_mult,
		num_res_blocks=vae_config.model.num_res_blocks,
		attn_resolutions=vae_config.model.attention_resolutions,
		resolution=96,
		dropout=vae_config.model.dropout,
		z_channels=vae_config.model.z_channels,
		embed_dim=vae_config.model.embed_dim
	).to(device)
	
	if args.dem_dataset == "polar" or args.dem_dataset == "bettersun":
		vae.load_state_dict(torch.load("results/polar_vae_{}MPP_{:03d}/vae_ckpt_iter{}".format(args.resolution, args.encoder_id, args.encoder_iter), map_location=device))
	else:
		raise NotImplementedError("dem_dataset must be in [polar, bettersun]")
	
	vae = nn.DataParallel(vae).eval()

	vae.train = disabled_train
	if not requires_grad:
		for param in vae.parameters():
			param.requires_grad = False

	return vae


def init_model_and_method(args, savepath, device, load_epoch=None):
	vae_config = get_vae_config()
	encoded_compression = 2 ** (len(vae_config.model.channel_mult)-1)
	encoded_channels = vae_config.model.z_channels

	network, model_config = init_model(
		downsample=args.downsample,
		encoded_channels=encoded_channels,
		encoded_compression=encoded_compression,
		savepath=savepath,
		device=device,
		load_epoch=load_epoch
	)

	method_config = init_method(
		config=model_config,
		device=device,
	)

	return network, model_config, method_config


def init_model(
	downsample=None,
	encoded_channels=None,
	encoded_compression=None,
	savepath=None,
	device='cpu',
	load_epoch=None,
	):

	config = load_config(savepath, get_set_dit_config())

	network = SetDiT(
		in_channels=encoded_channels,
		hidden_size=config.model.hidden_size,
		learn_sigma=False,
		input_size=96//encoded_compression,
		patch_size=config.model.patch_size,
		depth=config.model.depth,
		num_heads=config.model.num_heads,
		
		cond_input_size=96//downsample,
		cond_patch_size=config.model.cond_patch_size,
		cond_in_channels=1,
		cond_depth=config.model.cond_depth,
		cond_num_heads=config.model.cond_num_heads,

		set_input_size=96,
		set_patch_size=config.model.set_patch_size,
		set_in_channels=1,
		set_depth=config.model.set_depth,
		set_num_heads=config.model.set_num_heads,
		set_conv_model_channels=config.model.set_conv_model_channels,
		set_conv_out_channels=config.model.set_conv_out_channels,
		set_conv_num_res_blocks=config.model.set_conv_num_res_blocks,
		set_conv_channel_mult=config.model.set_conv_channel_mult,
		set_conv_resblock_updown=config.model.set_conv_resblock_updown,
		set_conv_norm_mode=config.model.set_conv_norm_mode,
		set_conv_pool_type=config.model.set_conv_pool_type,
	).to(device)

	network = nn.DataParallel(network)

	if load_epoch is not None:
		network.load_state_dict(torch.load("{}/net_ckpt_epoch{}".format(savepath, load_epoch), map_location=device))

	return network, config


def init_method(config, device):
	
	method_config = ml_collections.ConfigDict()
	
	method_config.interval = config.bridge.interval
	method_config.beta_max = config.bridge.beta_max

	betas = make_beta_schedule(n_timestep=method_config.interval, linear_end=method_config.beta_max / method_config.interval)
	method_config.betas = np.concatenate([betas[:method_config.interval//2], np.flip(betas[:method_config.interval//2])])
	method_config.diffusion = Diffusion(method_config.betas, device)

	return method_config