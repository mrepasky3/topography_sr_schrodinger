import os
import numpy as np
import h5py

import xarray as xr
from rasterio.enums import Resampling

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pole', type=str, default="south", choices=["south","north"])

parser.add_argument('--illum_type', type=str, default="lowsunele", choices=["lowsunele","highsunele","fibonacci"])

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--patch_size', type=int, default=96) # in pixels

parser.add_argument('--patch_start', type=int, default=0)
parser.add_argument('--patch_end', type=int, default=99225)

args = parser.parse_args()


if args.illum_type == "highsunele":
	num_imgs = 72
elif args.illum_type == "lowsunele":
	num_imgs = 288
elif args.illum_type == "fibonacci":
	num_imgs = 30

patch_dir = "{}_{}_{}px_{}MPP".format(args.pole, args.illum_type, args.patch_size, args.resolution)
h5_patches_dir = "{}_{}_{}px_{}MPP_h5".format(args.pole, args.illum_type, args.patch_size, args.resolution)

if not os.path.exists(h5_patches_dir):
	os.mkdir(h5_patches_dir)

main_data_path =  "{}/patches_{}px_{}MPP_{:06d}-{:06d}.hdf5".format(h5_patches_dir, args.patch_size, args.resolution, args.patch_start, args.patch_end)

patch_number = args.patch_start
this_patch_dir = "{}/patch_{:06d}".format(patch_dir,patch_number)

hd_patch = xr.open_dataarray("{}/hd_patch.tif".format(this_patch_dir))[:,2:-2,2:-2]

imgs_patch = xr.open_dataarray("{}/imgs_patch.tif".format(this_patch_dir))
imgs_patch = imgs_patch.rio.reproject_match(hd_patch, resampling=Resampling.bilinear)

with h5py.File(main_data_path, 'w') as f:
	f.create_dataset('imgs_patches', data = np.expand_dims(imgs_patch.data, 0), compression="lzf", chunks=True, maxshape=(None,num_imgs,args.patch_size,args.patch_size))
	if args.illum_type in ['lowsunele','fibonacci']:
		f.create_dataset('hd_patches', data = hd_patch.data - hd_patch.data.mean(), compression="lzf", chunks=True, maxshape=(None,args.patch_size,args.patch_size))


with h5py.File(main_data_path, 'a') as f:
	for i in range(args.patch_start+1,args.patch_end):
		this_patch_dir = "{}/patch_{:06d}".format(patch_dir,i)

		hd_patch = xr.open_dataarray("{}/hd_patch.tif".format(this_patch_dir))[:,2:-2,2:-2]
		
		imgs_patch = xr.open_dataarray("{}/imgs_patch.tif".format(this_patch_dir))
		imgs_patch = imgs_patch.rio.reproject_match(hd_patch, resampling=Resampling.bilinear)

		f['imgs_patches'].resize(f['imgs_patches'].shape[0] + 1, axis=0)
		f['imgs_patches'][-1] = imgs_patch

		if args.illum_type in ['lowsunele','fibonacci']:		
			f['hd_patches'].resize(f['hd_patches'].shape[0] + 1, axis=0)
			f['hd_patches'][-1] = hd_patch[0] - hd_patch.data.mean()