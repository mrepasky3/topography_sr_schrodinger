import argparse
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader

from utils.config_util import save_config, dump_config, dump_args, merge_args
from utils.init_util import get_path, load_vae, init_model_and_method, init_optim, load_dataset, split_dataset, init_up_down_samplers
from utils.opt_util import save_checkpoint, compute_loss, get_batch
from utils.val_util import init_validation_metric_dict, get_validation_metrics, update_validation_metrics
from utils.sample_util import get_samples
from visualization.plot_metrics import plot_train_loss, plot_val_loss
from visualization.plot_patches import validation_patches


parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=int)

parser.add_argument('--warm_start', action='store_true')
parser.add_argument('--warm_epoch', type=int, default=0)

parser.add_argument('--encoder_id', type=int, default=0)
parser.add_argument('--encoder_iter', type=int, default=5000)

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--downsample', type=int, default=16)

parser.add_argument('--dem_dataset', type=str, default="bettersun", choices=["polar","bettersun"])

parser.add_argument('--num_imgs_lower', type=int, default=5)
parser.add_argument('--num_imgs_upper', type=int, default=100)
parser.add_argument('--img_selection', type=str, default="replacement", choices=["no_replacement","replacement"])
parser.add_argument('--missing_prob', type=float, default=1.0)

parser.add_argument('--recon_plot_freq', type=int, default=5)
parser.add_argument('--num_plots', type=int, default=5)
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--valid_freq', type=int, default=5)

parser.add_argument('--nfe', type=int, default=10)
parser.add_argument('--val_size', type=int, default=1000)

args = parser.parse_args()


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.device_count() >= 1:
	print("Using {} GPUs".format(torch.cuda.device_count()))

savepath = get_path(args, create=(not args.warm_start))

dump_args(args, savepath)

vae = load_vae(args, device)
srmodel, config, method_config = init_model_and_method(args, savepath, device)
srmodel.eval()

if args.warm_start:
	srmodel.load_state_dict(torch.load("{}/warm_ckpt_epoch{}".format(savepath, args.warm_epoch), map_location=device))
	srmodel.eval()

dump_config(config, "{}/config.txt".format(savepath))
save_config(savepath, config)

model_params = list(srmodel.parameters())
optimizer = init_optim(model_params, config)


# load surface patch data
patch_dataset_raw = load_dataset(args, p_flip=config.optim.p_flip)
patch_dataset_train, patch_dataset_val, patch_dataset_test = split_dataset(patch_dataset_raw, args)

batch_size = config.optim.batch_size
patch_dataloader_train = DataLoader(patch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
patch_dataloader_val = DataLoader(patch_dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
plot_patch_dataloader_val = DataLoader(patch_dataset_val, batch_size=args.num_plots, shuffle=False, num_workers=0, drop_last=True)

downsampler, upsampler = init_up_down_samplers(args.downsample, mode=config.model.context_upsample_type, inp_size=96)


# train the net
train_epoch_loss, train_epochs = [], []
train_batch_loss = []

val_epochs, val_metric_dict = init_validation_metric_dict()

for epoch in range(config.optim.epochs):
	srmodel.train()

	batch_losses = []

	# training loop
	for batch in patch_dataloader_train:
		optimizer.zero_grad()

		batch_dict = get_batch(batch, downsampler, upsampler, vae, device)
		loss = compute_loss(method_config, srmodel, batch_dict)
		
		loss.backward()
		optimizer.step()

		batch_losses.append(loss.item())
		train_batch_loss.append(loss.item())

	train_epoch_loss.append(np.sum(batch_losses))
	train_epochs.append(epoch)

	plot_train_loss(
		train_epochs=train_epochs,
		train_loss=train_epoch_loss,
		train_batch_loss=train_batch_loss,
		savepath=savepath,
		batches_per_epoch=len(patch_dataset_train) // batch_size,
		save_trajectories=True
	)

	if (epoch % args.save_freq == 0) or (epoch == config.optim.epochs-1):
		save_checkpoint(srmodel, model_params, epoch, savepath)

	srmodel.eval()

	# compute and plot validation metrics
	if (epoch % args.valid_freq == 0) or (epoch == config.optim.epochs-1):
		with torch.no_grad():
			val_epochs.append(epoch)
			new_val_metric_dict = get_validation_metrics(
				args=args,
				method_config=method_config,
				network=srmodel,
				downsampler=downsampler,
				upsampler=upsampler,
				vae=vae,
				dataloader=patch_dataloader_val,
				device=device
			)
			update_validation_metrics(val_metric_dict, new_val_metric_dict)

			plot_val_loss(
				val_epochs=val_epochs,
				val_metrics=val_metric_dict,
				val_size=args.val_size,
				nfe=args.nfe,
				savepath=savepath,
				save_trajectories=True
			)

	# plot reconstructions
	if (epoch % args.recon_plot_freq == 0) or (epoch == config.optim.epochs-1):
		with torch.no_grad():
			plot_batch = next(iter(plot_patch_dataloader_val))

			batch_dict = get_batch(plot_batch, downsampler, upsampler, vae, device)
			recon_dems_batch = get_samples(
				method_config,
				args.nfe,
				srmodel,
				vae,
				batch_dict,
				device
			)

			validation_patches(
				true_dems_batch=batch_dict.dems_batch,
				interp_dems_batch=batch_dict.cond_dems_batch_upsampled,
				recon_dems_batch=recon_dems_batch,
				savepath=savepath+"/recon_nfe{}_epoch{:03d}.png".format(args.nfe+1, epoch)
			)