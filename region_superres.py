import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bridge.schrodinger_bridge import space_indices, compute_pred_x0

from utils.config_util import merge_args
from utils.init_util import get_path, load_vae, init_model_and_method, init_up_down_samplers
from utils.superres_util import unfold_patches, fold_patches, unfold_imgs, fold_filter, duplicated_fold_patches_weighted_stdev, UtilityDataset, get_blurred_grassfire_filter

import xarray as xr
from xrspatial import hillshade

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


parser = argparse.ArgumentParser()
parser.add_argument('--path_to_dem', type=str, default="process_data/large_region_3840px_20MPP")

parser.add_argument('--model_id', type=int, default=5) # numeric model id
parser.add_argument('--model_epoch', type=int, default=55)

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--downsample', type=int, default=16) # multiplicative factor to resolution

parser.add_argument('--dem_dataset', type=str, default="bettersun", choices=['polar','bettersun'])

parser.add_argument('--num_imgs', type=int, default=75)
parser.add_argument('--missing_prob', type=float, default=1.)
parser.add_argument('--lon', type=float, default=0.)
parser.add_argument('--lat', type=float, default=0.)

parser.add_argument('--nfe', type=int, default=10)
parser.add_argument('--n_repeat', type=int, default=4)

parser.add_argument('--region_size', type=int, default=960) # number of pixels to consider overall
parser.add_argument('--stride', type=int, default=48) # stride of moving window
parser.add_argument('--weight_sigma', type=float, default=2.)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--context_upsample_type', type=str, default="bicubic", choices=['bicubic','bilinear','nearest'])

parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))


# some functions to get sliced images

def get_sunpath_cart(lon, lat):
	az = np.arange(0,360,1)
	sunpath_x = np.cos(np.deg2rad(az))
	sunpath_y = np.sin(np.deg2rad(az))
	sunpath_z = 0 * az

	sunpath_x2 = sunpath_x
	sunpath_y2 = sunpath_y * np.cos(np.deg2rad(90-lat)) + sunpath_z * np.sin(np.deg2rad(90-lat))
	sunpath_z2 = -sunpath_y * np.sin(np.deg2rad(90-lat)) + sunpath_z * np.cos(np.deg2rad(90-lat))

	sunpath_x3 = sunpath_x2 * np.cos(np.deg2rad(lon)) + sunpath_y2 * np.sin(np.deg2rad(lon))
	sunpath_y3 = -sunpath_x2 * np.sin(np.deg2rad(lon)) + sunpath_y2 * np.cos(np.deg2rad(lon))
	sunpath_z3 = sunpath_z2

	sunpath_cart = np.concatenate([sunpath_x3.reshape(-1,1), sunpath_y3.reshape(-1,1), sunpath_z3.reshape(-1,1)],axis=-1)
	return sunpath_cart[sunpath_cart[:,2] > 0]

def get_naclike_slice_idx(region_size=96):
	this_angle = np.random.uniform(low=0.,high=180.)
	this_width = np.random.uniform(low=25.,high=50.)
	
	this_origin = np.random.uniform(low=0.,high=region_size-1, size=(2,))
	this_slope = (1 - np.tan(np.deg2rad(this_angle))) / (1 + np.tan(np.deg2rad(this_angle)))
	
	this_intercept = this_origin[1] - this_slope*this_origin[0]
	
	parallel_intercept_1 = this_width * np.sqrt(np.power(this_slope,2)+1) + this_intercept
	line_func_1 = lambda x : this_slope * x + parallel_intercept_1
	
	parallel_intercept_2 = this_width * np.sqrt(np.power(this_slope,2)+1) - this_intercept
	line_func_2 = lambda x : this_slope * x - parallel_intercept_2

	_X, _Y = np.meshgrid(np.arange(region_size),np.arange(region_size))
	missing_data_mask = np.logical_or((_Y - line_func_1(_X)) > 0, (_Y - line_func_2(_X)) < 0)
	missing_data_x_idx, missing_data_y_idx = np.where(missing_data_mask)
	return missing_data_x_idx, missing_data_y_idx


savepath = get_path(args, create=False)

recon_savepath = "{}/recon_{}lon_{}lat_numimgs{}_blur{:.1f}".format(savepath, args.lon, args.lat, args.num_imgs, args.weight_sigma)
	
if not os.path.exists(recon_savepath):
	os.mkdir(recon_savepath)

with open(savepath+"/argdict", 'rb') as f:
	arg_dict = pickle.load(f)
arg_dict['batch_size'] = args.batch_size
arg_dict['missing_prob'] = args.missing_prob
args = merge_args(arg_dict, parser)


# load the pre-trained model (and vae)
vae = load_vae(args, device, requires_grad=False)
srmodel, config, method_config = init_model_and_method(args, savepath, device, load_epoch=args.model_epoch)
srmodel.eval()
	

downsampler, upsampler = init_up_down_samplers(args.downsample, args.context_upsample_type, args.region_size)

height_scaling = 1000.
stride = args.stride


# load large dem and rendered images
root = args.path_to_dem
dem_patch = np.array(xr.open_dataarray(f"{root}/hd_patch.tif").data)
dem_patch = torch.tensor(dem_patch).unsqueeze(1)[:,:,:args.region_size,:args.region_size]

all_imgs_patch = []
for img_name in ["5","10","20","30","45","60-85"]:
	_img_patch = np.array(xr.open_dataarray(f"{root}/imgs_patch_{img_name}.tif").data)
	all_imgs_patch.append(torch.tensor(_img_patch[:,:args.region_size,:args.region_size]).unsqueeze(1))
all_imgs_patch = torch.cat(all_imgs_patch)

azi_ele_list = []
for img_name in ["5","10","20","30","45","60-85"]:
	azi_ele_list += list(np.load(f'{root}/azi_ele_{img_name}.npy'))
azi_ele_list = np.array(azi_ele_list)


# get path of the sun in the sky according to assumed lon-lat
lattice_x = np.sin(np.deg2rad(90-azi_ele_list[:,1]))*np.cos(np.deg2rad(azi_ele_list[:,0]))
lattice_y = np.sin(np.deg2rad(90-azi_ele_list[:,1]))*np.sin(np.deg2rad(azi_ele_list[:,0]))
lattice_z = np.cos(np.deg2rad(90-azi_ele_list[:,1]))
lattice_cart = np.concatenate([lattice_x.reshape(-1,1), lattice_y.reshape(-1,1), lattice_z.reshape(-1,1)],axis=-1)

sunpath_cart = get_sunpath_cart(args.lon, args.lat)


# randomly select rendered sun positions 'close' to path positions
min_dist_lattice = scipy.spatial.distance.cdist(lattice_cart, sunpath_cart, metric='euclidean').min(axis=1)
lattice_mask = min_dist_lattice<0.1
valid_img_idx = np.where(lattice_mask)[0]
selected_imgs_idx = np.random.choice(valid_img_idx, replace=True, size=(args.num_imgs))


# for all images, 'slice' observed data to imitate NAC images
imgs_patch = torch.clone(all_imgs_patch[selected_imgs_idx])
missing_data_idx = np.where(np.random.binomial(1,p=args.missing_prob,size=(imgs_patch.shape[0],)))[0]
for missing_idx in missing_data_idx:
	missing_data_x_idx, missing_data_y_idx = get_naclike_slice_idx(region_size=args.region_size)
	imgs_patch[missing_idx, :, missing_data_x_idx, missing_data_y_idx] = -1.


# get downsampled and interpolated DEM
cond_dem_patch = downsampler(dem_patch)
upsampled_cond_dem_patch = upsampler(cond_dem_patch)


# 'unfold' large region in many small (96 x 96) patches
unfolded_imgs_patched = unfold_imgs(imgs_patch, kernel_size=96, stride=stride)
unfolded_upsampled_cond_dems_patched = unfold_patches(upsampled_cond_dem_patch, kernel_size=96, stride=stride)
unfolded_cond_dems_patched = downsampler(unfolded_upsampled_cond_dems_patched)


# center (zero mean) and scale (km) each small patch
cond_dems_batch_means = unfolded_cond_dems_patched.mean(dim=(-1,-2))[...,None,None]
cond_dems_batch = ((unfolded_cond_dems_patched - cond_dems_batch_means) / height_scaling).to(device)
upsampled_cond_dems_batch = ((unfolded_upsampled_cond_dems_patched - cond_dems_batch_means) / height_scaling).to(device)

imgs_batch = unfolded_imgs_patched.unsqueeze(2).to(device)
batch_size, set_size, _, _, _ = imgs_batch.shape
masks_batch = ((imgs_batch.view(batch_size, set_size, -1).max(axis=-1).values - imgs_batch.view(batch_size, set_size, -1).min(axis=-1).values) > 0).float()


# load all patches into a dataloder so we can iterate efficiently over all patches
utility_dataset = UtilityDataset(
	imgs_batch,
	masks_batch,
	cond_dems_batch,
	upsampled_cond_dems_batch,
)
batch_size = args.batch_size
patch_dataloader = DataLoader(utility_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


# iterate over all patches n_repeat many times per patch
n_repeat = args.n_repeat
with torch.no_grad():
	nfe = min(method_config.interval-1, args.nfe)
	steps = space_indices(method_config.interval, nfe+1)
	
	# create log steps
	log_count = len(steps)-1
	log_steps = [steps[i] for i in space_indices(len(steps)-1, log_count)]
	assert log_steps[0] == 0

	duplicated_recon_dems_batch = []
	for _ in range(n_repeat):
		recon_dems_batch = []
		for batch in tqdm(patch_dataloader):
			imgs_batch = batch['img']
			masks_batch = batch['img_mask']
			cond_dems_batch = batch['low_res_dem']
			upsampled_cond_dems_batch = batch['upsampled_low_res_dem']

			latent_cond_dems_batch = vae.module.encode_sample(upsampled_cond_dems_batch)

			def pred_x0_fn(xt, imgs, masks, step):
				step = torch.full((xt.shape[0],), step, dtype=torch.long)
				out = srmodel(xt, step, cond_dems_batch, imgs, masks)
				return compute_pred_x0(method_config.diffusion, step, xt, out)

			# go from low-res DEM to high-res DEM
			zs, pred_x0 = method_config.diffusion.ddpm_sampling(
				imgs_batch, masks_batch, steps, pred_x0_fn, latent_cond_dems_batch, log_steps=log_steps, verbose=True,
			)
			x_recon = vae.module.decode(zs[:,0,...].to(device))
			recon_dems_batch.append(x_recon)
		recon_dems_batch = torch.cat(recon_dems_batch, dim=0).to(device)
		recon_dems_batch += (cond_dems_batch_means.to(device) / height_scaling) # add back the mean

		duplicated_recon_dems_batch.append(recon_dems_batch)


# merge patches together, blending the overlapping regions using a grassfire filter
filters = get_blurred_grassfire_filter(filt_exp=2, weight_sigma=args.weight_sigma, num_patches=recon_dems_batch.shape[0], device=device)
weighted_folded_recon_dems, weighted_folded_recon_dems_std = duplicated_fold_patches_weighted_stdev(duplicated_recon_dems_batch, filters, (1,1,args.region_size,args.region_size), device, kernel_size=96, stride=stride)




# PLOTTERS

folded_upsampled_cond_dems_patched = fold_patches(unfolded_upsampled_cond_dems_patched, (1,1,args.region_size,args.region_size), kernel_size=96, stride=stride)

vmin_dem = min(
	torch.amin(weighted_folded_recon_dems.cpu()),
	torch.amin(dem_patch.cpu() / height_scaling),
	torch.amin(folded_upsampled_cond_dems_patched.cpu() / height_scaling),
	)

vmax_dem = max(
	torch.amax(weighted_folded_recon_dems.cpu()),
	torch.amax(dem_patch.cpu() / height_scaling),
	torch.amax(folded_upsampled_cond_dems_patched.cpu() / height_scaling),
	)

weighted_recon_terrain = xr.DataArray(weighted_folded_recon_dems.cpu().squeeze())
weighted_recon_hillshade = hillshade(weighted_recon_terrain)[1:-1,1:-1].data

true_terrain = xr.DataArray(dem_patch.cpu().squeeze() / height_scaling)
true_hillshade = hillshade(true_terrain)[1:-1,1:-1].data

interp_terrain = xr.DataArray(folded_upsampled_cond_dems_patched.cpu().squeeze() / height_scaling)
interp_hillshade = hillshade(interp_terrain)[1:-1,1:-1].data


vmin_shade = min(
	np.amin(weighted_recon_hillshade),
	np.amin(true_hillshade),
	np.amin(interp_hillshade),
	)

vmax_shade = max(
	np.amax(weighted_recon_hillshade),
	np.amax(true_hillshade),
	np.amax(interp_hillshade),
	)


# weighted recon DEM and hillshade
weighted_recon_plot_name = "{}/recon_{}px_{}missing_stride{}_{}x_{}_steps{}_dup{}_epoch{:03d}".format(
	recon_savepath,
	args.region_size,
	args.missing_prob,
	args.stride,
	args.downsample,
	args.context_upsample_type,
	args.nfe+1,
	args.n_repeat,
	args.model_epoch,
)

fig, ax = plt.subplots(1,2,figsize=(2*4,4))

orig_im = ax[0].imshow(weighted_folded_recon_dems.cpu().squeeze(),cmap='viridis',
	vmin=vmin_dem, vmax=vmax_dem, extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])
cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(orig_im, cax=cax, orientation='vertical')

orig_im = ax[1].imshow(weighted_recon_hillshade,cmap='viridis',
	vmin=vmin_shade, vmax=vmax_shade, extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])

plt.tight_layout()
plt.savefig(weighted_recon_plot_name+".png")
plt.clf()
plt.close()


# weighted recon prediction uncertainty and error
weighted_recon_std_plot_name = "{}/reconstd_{}px_{}missing_stride{}_{}x_{}_steps{}_dup{}_epoch{:03d}".format(
	recon_savepath,
	args.region_size,
	args.missing_prob,
	args.stride,
	args.downsample,
	args.context_upsample_type,
	args.nfe+1,
	args.n_repeat,
	args.model_epoch,
)

fig, ax = plt.subplots(1,2,figsize=(2*4,4))

orig_im_std = ax[0].imshow(1e3 * weighted_folded_recon_dems_std.cpu().squeeze(),cmap='viridis',
	extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])
cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
ax[0].set_title("Standard Deviation", fontsize=14)
fig.colorbar(orig_im_std, cax=cax, orientation='vertical')

weighted_recon_err = 1e3 * torch.abs(weighted_folded_recon_dems.cpu().squeeze() - (dem_patch.cpu().squeeze() / height_scaling))
orig_im_err = ax[1].imshow(weighted_recon_err,cmap='viridis',
	extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])
cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(orig_im_err, cax=cax, orientation='vertical')
ax[1].set_title("Absolute Error", fontsize=14)

plt.tight_layout()
plt.savefig(weighted_recon_std_plot_name + ".png")
plt.clf()
plt.close()


# filter
filter_plot_name = "{}/filter_{}px_{}missing_stride{}.png".format(
	recon_savepath,
	args.region_size,
	args.missing_prob,
	args.stride,
)

folded_filter = fold_filter(filters, (1,1,args.region_size,args.region_size), kernel_size=96, stride=stride)

fig, ax = plt.subplots(1,2,figsize=(2*4,4))

orig_im = ax[0].imshow(filters[0].cpu().squeeze(),cmap='viridis')
cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(orig_im, cax=cax, orientation='vertical')

orig_im = ax[1].imshow(folded_filter.cpu().squeeze(),cmap='viridis')
cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(orig_im, cax=cax, orientation='vertical')

plt.tight_layout()
plt.savefig(filter_plot_name)
plt.clf()
plt.close()


# true DEM and hillshade
true_plot_name = "{}/true_{}px".format(
	recon_savepath,
	args.region_size
)

fig, ax = plt.subplots(1,2,figsize=(2*4,4))

orig_im = ax[0].imshow(dem_patch.cpu().squeeze() / height_scaling,cmap='viridis',
	vmin=vmin_dem, vmax=vmax_dem, extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])
cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(orig_im, cax=cax, orientation='vertical')

orig_im = ax[1].imshow(true_hillshade,cmap='viridis',
	vmin=vmin_shade, vmax=vmax_shade, extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])

plt.tight_layout()
plt.savefig(true_plot_name+".png")
plt.clf()
plt.close()


# interpolated DEM and hillshade
interp_plot_name = "{}/true_{}px_{}x_{}".format(
	recon_savepath,
	args.region_size,
	args.downsample,
	args.context_upsample_type,
)

fig, ax = plt.subplots(1,2,figsize=(2*4,4))

orig_im = ax[0].imshow(folded_upsampled_cond_dems_patched.cpu().squeeze() / height_scaling,cmap='viridis',
	vmin=vmin_dem, vmax=vmax_dem, extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])
cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
fig.colorbar(orig_im, cax=cax, orientation='vertical')

orig_im = ax[1].imshow(interp_hillshade,cmap='viridis',
	vmin=vmin_shade, vmax=vmax_shade, extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])

plt.tight_layout()
plt.savefig(interp_plot_name+".png")
plt.clf()
plt.close()


# rendered optical images num counts
imgcount_plot_name = "{}/imgcount_{}px_{}missing".format(
	recon_savepath,
	args.region_size,
	args.missing_prob,
)

fig = plt.figure(figsize=(4,4))

orig_im = plt.imshow((imgs_patch != -1).sum(axis=0).squeeze(),cmap='viridis',
	extent=[0. , args.region_size*args.resolution, 0., args.region_size*args.resolution])
plt.title("Count", fontsize=14)
plt.colorbar()

plt.tight_layout()
plt.savefig(imgcount_plot_name + ".png")
plt.clf()
plt.close()