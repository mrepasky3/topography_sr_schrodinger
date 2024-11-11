import os
import argparse
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.configs import get_vae_config
from utils.config_util import save_config, dump_config, dump_args, merge_args
from utils.init_util import  load_dataset, split_dataset
from utils.opt_util import get_batch_vae
from visualization.plot_metrics import plot_vae_loss, plot_disc_loss
from visualization.plot_patches import validation_patches_vae

from latent_diffusion.loss import LPIPSWithDiscriminator
from latent_diffusion.autoencoder import AutoencoderKL


parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=int) # numeric model id

parser.add_argument('--resolution', type=int, default=20) # in meters

parser.add_argument('--dem_dataset', type=str, default="bettersun", choices=["bettersun"])

parser.add_argument('--recon_plot_freq', type=int, default=5)
parser.add_argument('--num_plots', type=int, default=5)
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--valid_freq', type=int, default=5)

args = parser.parse_args()


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))

if args.dem_dataset == "bettersun":
	savepath = "results/polar_vae_{}MPP_{:03d}".format(args.resolution, args.model_id)
else:
	raise NotImplementedError("dem_dataset must be in [bettersun]")

assert not os.path.exists(savepath), "{} Already Exists!".format(savepath)
os.mkdir(savepath)

dump_args(args, savepath)

config = get_vae_config()
dump_config(config, "{}/config.txt".format(savepath))
save_config(savepath, config)

vae = AutoencoderKL(
	in_channels=1,
	out_ch=1,
	ch=config.model.base_channels,
	ch_mult=config.model.channel_mult,
	num_res_blocks=config.model.num_res_blocks,
	attn_resolutions=config.model.attention_resolutions,
	resolution=96,
	dropout=config.model.dropout,
	z_channels=config.model.z_channels,
	embed_dim=config.model.embed_dim
).to(device)

loss_func = LPIPSWithDiscriminator(
	config.optim.disc_start,
	kl_weight=config.optim.kl_weight,
	disc_weight=config.optim.disc_weight,
	disc_in_channels=1,
	disc_num_layers=config.optim.disc_num_layers
).to(device)

optimizer_ae = optim.Adam(vae.parameters(), lr=config.optim.lr, betas=(0.5, 0.9))
optimizer_disc = optim.Adam(loss_func.discriminator.parameters(), lr=config.optim.lr, betas=(0.5, 0.9))


# load surface patch data (lazily)
patch_dataset_raw = load_dataset(args, p_flip=config.optim.p_flip, dem_only=True)
patch_dataset_train, patch_dataset_val, patch_dataset_test = split_dataset(patch_dataset_raw, args)

batch_size = config.optim.batch_size
patch_dataloader_train = DataLoader(patch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
patch_dataloader_val = DataLoader(patch_dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
plot_patch_dataloader_val = DataLoader(patch_dataset_val, batch_size=args.num_plots, shuffle=False, num_workers=0, drop_last=True)


# train the net
global_step = 0

vae_train_steps, vae_train_losses, vae_train_nll, vae_train_kl, vae_train_g_loss = [], [], [], [], []
disc_train_steps, disc_train_losses = [], []

val_steps = []
vae_val_losses, vae_val_nll, vae_val_kl, vae_val_g_loss = [], [], [], []
disc_val_losses = []

for epoch in range(config.optim.epochs):
	vae.train()
	loss_func.train()
	
	batch_losses = []

	# training loop
	for batch in patch_dataloader_train:
		optimizer_ae.zero_grad()
		optimizer_disc.zero_grad()

		batch_dict = get_batch_vae(batch, device)
		dems_batch = batch_dict.dems_batch
		
		reconstructions, posterior = vae(dems_batch)

		if global_step % config.optim.alternate_period < config.optim.vae_alternate_iter:
			# train encoder+decoder+logvar
			loss, report = loss_func(
				inputs=dems_batch,
				reconstructions=reconstructions,
				posteriors=posterior,
				optimizer_idx=0,
				global_step=global_step,
				last_layer=vae.get_last_layer()
			)
			loss.mean().backward()
			optimizer_ae.step()

			vae_train_steps.append(global_step)
			vae_train_losses.append(report['train/total_loss'].cpu())
			vae_train_nll.append(report['train/nll_loss'].cpu())
			vae_train_kl.append(report['train/kl_loss'].cpu())
			vae_train_g_loss.append(report['train/g_loss'].cpu())

		else:
			# train the discriminator
			loss, report = loss_func(
				inputs=dems_batch,
				reconstructions=reconstructions,
				posteriors=posterior,
				optimizer_idx=1,
				global_step=global_step,
				last_layer=vae.get_last_layer()
			)
			loss.mean().backward()
			optimizer_disc.step()

			disc_train_steps.append(global_step)
			disc_train_losses.append(report['train/disc_loss'].cpu())

		global_step += 1

	plot_vae_loss(
		vae_train_steps,
		vae_train_losses,
		vae_train_nll,
		vae_train_kl,
		vae_train_g_loss,
		savepath,
		split='train',
		save_trajectories=True
	)
	plot_disc_loss(
		disc_train_steps,
		disc_train_losses,
		savepath,
		split='train',
		save_trajectories=True
	)

	if (epoch % args.save_freq == 0) or (epoch == config.optim.epochs-1):
		torch.save(vae.state_dict(), savepath+"/vae_ckpt_iter{}".format(global_step))
		torch.save(loss_func.discriminator.state_dict(), savepath+"/disc_ckpt_iter{}".format(global_step))

	vae.eval()
	loss_func.eval()

	# compute and plot validation metrics
	if (epoch % args.valid_freq == 0) or (epoch == config.optim.epochs-1):
		with torch.no_grad():
			epoch_vae_val_losses, epoch_vae_val_nll, epoch_vae_val_kl, epoch_vae_val_g_loss = 0., 0., 0., 0.,
			epoch_disc_val_losses = 0.
			for batch in patch_dataloader_val:
				batch_dict = get_batch_vae(batch, device)
				dems_batch = batch_dict.dems_batch
				
				reconstructions, posterior = vae(dems_batch)

				_, ae_report = loss_func(
					inputs=dems_batch,
					reconstructions=reconstructions,
					posteriors=posterior,
					optimizer_idx=0,
					global_step=global_step,
					last_layer=vae.get_last_layer(),
					split="val"
				)

				_, disc_report = loss_func(
					inputs=dems_batch,
					reconstructions=reconstructions,
					posteriors=posterior,
					optimizer_idx=1,
					global_step=global_step,
					last_layer=vae.get_last_layer(),
					split="val"
				)

				epoch_vae_val_losses += ae_report['val/total_loss'].cpu()
				epoch_vae_val_nll += ae_report['val/nll_loss'].cpu()
				epoch_vae_val_kl += ae_report['val/kl_loss'].cpu()
				epoch_vae_val_g_loss += ae_report['val/g_loss'].cpu()
				epoch_disc_val_losses += disc_report['val/disc_loss'].cpu()

			val_steps.append(global_step)
			vae_val_losses.append(epoch_vae_val_losses)
			vae_val_nll.append(epoch_vae_val_nll)
			vae_val_kl.append(epoch_vae_val_kl)
			vae_val_g_loss.append(epoch_vae_val_g_loss)
			disc_val_losses.append(epoch_disc_val_losses)

			plot_vae_loss(
				val_steps,
				vae_val_losses,
				vae_val_nll,
				vae_val_kl,
				vae_val_g_loss,
				savepath,
				split='validation',
				save_trajectories=True
			)
			plot_disc_loss(
				val_steps,
				disc_val_losses,
				savepath,
				split='validation',
				save_trajectories=True
			)

	# plot reconstructions
	if (epoch % args.recon_plot_freq == 0) or (epoch == config.optim.epochs-1):
		with torch.no_grad():
			plot_batch = next(iter(plot_patch_dataloader_val))

			batch_dict = get_batch_vae(plot_batch, device)
			dems_batch = batch_dict.dems_batch
			
			reconstructions = vae(dems_batch)[0]

			validation_patches_vae(
				true_dems_batch=batch_dict.dems_batch,
				recon_dems_batch=reconstructions,
				savepath=savepath+"/recon_epoch{:03d}.png".format(epoch)
			)