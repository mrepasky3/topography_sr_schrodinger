import os
import pickle
import argparse
import numpy as np

import xarray as xr
from xrspatial import hillshade

import torch
from torch.utils.data import DataLoader

from datasets.patches_dataset import PolarPatchRealisticSun
from utils.config_util import merge_args
from utils.init_util import get_path, load_vae, init_model_and_method, init_up_down_samplers
from utils.opt_util import get_repeated_batch
from utils.sample_util import get_samples
from visualization.plot_patches import plot_patch, plot_patch_cbar, plot_imgs


parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=int)
parser.add_argument('--model_epoch', type=int, default=5)

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--downsample', type=int, default=16)

parser.add_argument('--dem_dataset', type=str, default="bettersun", choices=["bettersun","polar"]) # dataset model was trained on

parser.add_argument('--num_imgs', type=int, default=20)
parser.add_argument('--img_selection', type=str, default="replacement", choices=["no_replacement","replacement"])
parser.add_argument('--missing_prob', type=float, default=0.0)
parser.add_argument('--lon', type=float, default=0)
parser.add_argument('--lat', type=float, default=80)

parser.add_argument('--nfe', type=int, default=10)
parser.add_argument('--n_repeat', type=int, default=10)

parser.add_argument('--split', type=str, default="test", choices=["train","val","test"])
parser.add_argument('--patch_idx', type=int, default=0) # patch index in dataset partition

parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))

savepath = get_path(args, create=False)
recon_savepath = savepath+"/recon_"+args.split
if not os.path.exists(recon_savepath):
	os.mkdir(recon_savepath)

with open(savepath+"/argdict", 'rb') as f:
	arg_dict = pickle.load(f)
arg_dict["nfe"] = args.nfe
arg_dict['img_selection'] = args.img_selection
arg_dict["missing_prob"] = args.missing_prob

args = merge_args(arg_dict, parser)


# load the pre-trained model (and vae)
vae = load_vae(args, device)
srmodel, config, method_config = init_model_and_method(args, savepath, device, load_epoch=args.model_epoch)
srmodel.eval()


# choose partition (train/val/test) from dataset
patch_dataset_raw = PolarPatchRealisticSun(
	p_flip=0.,
	num_img_vals=np.arange(args.num_imgs,args.num_imgs+1,1),
	lon_bounds=(args.lon, args.lon),
	lat_bounds=(args.lat, args.lat),
	fixed_imgs=(args.img_selection == "no_replacement"),
	missing_prob=args.missing_prob,
	img_selection=args.img_selection,
)
train_idx = np.load("process_data/thresh2242_train_idx.npy")
val_idx = np.load("process_data/thresh2242_val_idx.npy")
test_idx = np.load("process_data/thresh2242_test_idx.npy")

patch_dataset_train = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(train_idx))
patch_dataset_val = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(val_idx))
patch_dataset_test = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(test_idx))

patch_dataset = {
	"train": patch_dataset_train,
	"val": patch_dataset_val,
	"test": patch_dataset_test
}[args.split]

patch_dataloader = DataLoader(patch_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

downsampler, upsampler = init_up_down_samplers(args.downsample, mode=config.model.context_upsample_type, inp_size=96)

height_scaling = 1000.


# iterate through dataloader until reaching desired batch
_iter_patch_dataloader = iter(patch_dataloader)
for _ in range(args.patch_idx+1):
	batch = next(_iter_patch_dataloader)
batch_dict = get_repeated_batch(args.n_repeat, batch, downsampler, upsampler, vae, device)

dems_batch = batch_dict.dems_batch
cond_dems_batch_upsampled = batch_dict.cond_dems_batch_upsampled


# get samples from superres model
recon_dems_batch = get_samples(
	method_config,
	args.nfe,
	srmodel,
	vae,
	batch_dict,
	device,
	verbose=args.verbose,
)


path_name = "{}/lon{}_lat{}_numimgs{}_{}_missing{:.1f}_patch{}_steps{}_nrepeat{}_epoch{:03d}".format(
	recon_savepath,
	args.lon,
	args.lat,
	args.num_imgs,
	args.img_selection,
	args.missing_prob,
	args.patch_idx,
	args.nfe+1,
	args.n_repeat,
	args.model_epoch
)
if not os.path.exists(path_name):
	os.mkdir(path_name)


# pixel-wise aggregation of multiple recons (averaging)
if args.n_repeat > 1:
	n, c, h, w = recon_dems_batch.size()
	agg_recon_dems_batch = recon_dems_batch.reshape(n//args.n_repeat, args.n_repeat, c, h, w).mean(1)
	std_recon_dems_batch = height_scaling*recon_dems_batch.reshape(n//args.n_repeat, args.n_repeat, c, h, w).std(1)

	recon_dems_batch_errs = height_scaling*(recon_dems_batch - dems_batch[0][None,...])
	recon_dems_batch_mean_err = recon_dems_batch_errs.reshape(n//args.n_repeat, args.n_repeat, c, h, w).mean(1)
	recon_dems_batch_mean_abs_err = torch.abs(recon_dems_batch_errs).reshape(n//args.n_repeat, args.n_repeat, c, h, w).mean(1)
	
	recon_dems_batch = recon_dems_batch.reshape(n//args.n_repeat, args.n_repeat, c, h, w)[:,0]
else:
	agg_recon_dems_batch = recon_dems_batch


# collect all results

true_dem = dems_batch[0].squeeze().cpu().numpy()
interp_dem = cond_dems_batch_upsampled[0].squeeze().cpu().numpy()
agg_recon_dem = agg_recon_dems_batch[0].squeeze().cpu().numpy()
recon_dem = recon_dems_batch[0].squeeze().cpu().numpy()

imgs_batch = batch_dict.imgs_batch[0].cpu().numpy()
masks_batch = batch_dict.masks_batch[0].cpu().numpy()
unmasked_idx = np.where(masks_batch)[0]

true_hillshade = hillshade(xr.DataArray(true_dem))[1:-1,1:-1].data
interp_hillshade = hillshade(xr.DataArray(interp_dem))[1:-1,1:-1].data
agg_recon_hillshade = hillshade(xr.DataArray(agg_recon_dem))[1:-1,1:-1].data
recon_hillshade = hillshade(xr.DataArray(recon_dem))[1:-1,1:-1].data

all_dems = np.concatenate([true_dem[None,...], interp_dem[None,...], agg_recon_dem[None,...], recon_dem[None,...]], axis=0)
vmin_dem, vmax_dem = np.min(all_dems), np.max(all_dems)

all_hills = np.concatenate([true_hillshade[None,...], interp_hillshade[None,...], agg_recon_hillshade[None,...], recon_hillshade[None,...]])
vmin_hill, vmax_hill = np.min(all_hills), np.max(all_hills)


# plot true, interpolated, and reconstructed DEMs and hillshades

plot_imgs(imgs_batch[unmasked_idx], savepath=path_name, img_title="images.png")

plot_patch(true_dem, vmin=vmin_dem, vmax=vmax_dem,
		savepath=path_name, img_title="dem_true.png")
plot_patch(true_hillshade, vmin=vmin_hill, vmax=vmax_hill,
		savepath=path_name, img_title="hillshade_true.png")

plot_patch(interp_dem, vmin=vmin_dem, vmax=vmax_dem,
		savepath=path_name, img_title="dem_interpolated.png")
plot_patch(interp_hillshade, vmin=vmin_hill, vmax=vmax_hill,
		savepath=path_name, img_title="hillshade_interpolated.png")

plot_patch(agg_recon_dem, vmin=vmin_dem, vmax=vmax_dem,
		savepath=path_name, img_title="dem_aggrecon.png")
plot_patch(agg_recon_hillshade, vmin=vmin_hill, vmax=vmax_hill,
		savepath=path_name, img_title="hillshade_aggrecon.png")

plot_patch(recon_dem, vmin=vmin_dem, vmax=vmax_dem,
		savepath=path_name, img_title="dem_recon.png")
plot_patch(recon_hillshade, vmin=vmin_hill, vmax=vmax_hill,
		savepath=path_name, img_title="hillshade_recon.png")


# plot reconstruction error maps

recon_abs_error = np.abs(true_dem - recon_dem)*height_scaling
agg_recon_abs_error = np.abs(true_dem - agg_recon_dem)*height_scaling

plot_patch_cbar(recon_abs_error, vmin=None, vmax=None,
	savepath=path_name, img_title="dem_recon_abs_error")


# plot predicted uncertainty map

if args.n_repeat > 1:
	std_recon_dem = std_recon_dems_batch[0].squeeze().cpu().numpy()

	recon_mean_err = recon_dems_batch_mean_err[0].squeeze().cpu().numpy()
	recon_mean_abs_err = recon_dems_batch_mean_abs_err[0].squeeze().cpu().numpy()

	plot_patch_cbar(agg_recon_abs_error, vmin=None, vmax=None,
		savepath=path_name, img_title="dem_aggrecon_abs_error")

	plot_patch_cbar(std_recon_dem, vmin=None, vmax=None,
		savepath=path_name, img_title="dem_stdrecon")

	plot_patch_cbar(recon_mean_err, vmin=None, vmax=None,
		savepath=path_name, img_title="dem_recon_mean_error")

	plot_patch_cbar(recon_mean_abs_err, vmin=None, vmax=None,
		savepath=path_name, img_title="dem_recon_mean_abs_error")