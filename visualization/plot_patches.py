import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import xarray as xr
from xrspatial import hillshade


def validation_patches(true_dems_batch, interp_dems_batch, recon_dems_batch, savepath):
	fig, ax = plt.subplots(6,true_dems_batch.shape[0], figsize=(true_dems_batch.shape[0]*4,3*4))
		
	for i in range(true_dems_batch.shape[0]):
		true_terrain = xr.DataArray(true_dems_batch[i].squeeze().cpu())
		true_hillshade = hillshade(true_terrain).data[1:-1,1:-1]
	
		orig_im = ax[0,i].imshow(true_dems_batch[i].squeeze().cpu(), cmap='viridis')
		cax = make_axes_locatable(ax[0,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[0,i].set_title("True DEM", fontsize=16)
		
		orig_im = ax[1,i].imshow(true_hillshade, cmap='viridis')
		cax = make_axes_locatable(ax[1,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[1,i].set_title("True Hillshade", fontsize=16)


		interpolated_terrain = xr.DataArray(interp_dems_batch[i].squeeze().cpu())
		interpolated_hillshade = hillshade(interpolated_terrain).data[1:-1,1:-1]
	
		orig_im = ax[2,i].imshow(interp_dems_batch[i].squeeze().cpu(), cmap='viridis')
		cax = make_axes_locatable(ax[2,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[2,i].set_title("Interpolated DEM", fontsize=16)
		
		orig_im = ax[3,i].imshow(interpolated_hillshade, cmap='viridis')
		cax = make_axes_locatable(ax[3,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[3,i].set_title("Interpolated Hillshade", fontsize=16)


		out_terrain = xr.DataArray(recon_dems_batch[i].squeeze().cpu())
		out_hillshade = hillshade(out_terrain).data[1:-1,1:-1]
	
		orig_im = ax[4,i].imshow(recon_dems_batch[i].squeeze().cpu(), cmap='viridis')
		cax = make_axes_locatable(ax[4,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[4,i].set_title("Reconstructed DEM", fontsize=16)
		
		orig_im = ax[5,i].imshow(out_hillshade, cmap='viridis')
		cax = make_axes_locatable(ax[5,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[5,i].set_title("Reconstructed Hillshade", fontsize=16)

	for i in range(6):
		for j in range(true_dems_batch.shape[0]):
			ax[i,j].axis('off')
	
	plt.tight_layout()
	plt.savefig(savepath)
	plt.clf()
	plt.close()


def validation_patches_vae(true_dems_batch, recon_dems_batch, savepath):
	fig, ax = plt.subplots(4,true_dems_batch.shape[0], figsize=(true_dems_batch.shape[0]*4,4*4))
		
	for i in range(true_dems_batch.shape[0]):
		true_terrain = xr.DataArray(true_dems_batch[i].squeeze().cpu())
		true_hillshade = hillshade(true_terrain).data[1:-1,1:-1]
	
		orig_im = ax[0,i].imshow(true_dems_batch[i].squeeze().cpu(), cmap='viridis', extent=[0. , 2000., 0. , 2000.])
		cax = make_axes_locatable(ax[0,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[0,i].set_title("True DEM", fontsize=16)
		
		orig_im = ax[1,i].imshow(true_hillshade, cmap='viridis', extent=[0. , 2000., 0. , 2000.])
		cax = make_axes_locatable(ax[1,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[1,i].set_title("True Hillshade", fontsize=16)


		out_terrain = xr.DataArray(recon_dems_batch[i].squeeze().cpu())
		out_hillshade = hillshade(out_terrain).data[1:-1,1:-1]
	
		orig_im = ax[2,i].imshow(recon_dems_batch[i].squeeze().cpu(), cmap='viridis', extent=[0. , 2000., 0. , 2000.])
		cax = make_axes_locatable(ax[2,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[2,i].set_title("Reconstructed DEM", fontsize=16)
		
		orig_im = ax[3,i].imshow(out_hillshade, cmap='viridis', extent=[0. , 2000., 0. , 2000.])
		cax = make_axes_locatable(ax[3,i]).append_axes('right', size='5%', pad=0.05)
		fig.colorbar(orig_im, cax=cax, orientation='vertical')
		ax[3,i].set_title("Reconstructed Hillshade", fontsize=16)

	for i in range(4):
		for j in range(true_dems_batch.shape[0]):
			ax[i,j].axis('off')
	
	plt.tight_layout()
	plt.savefig(savepath)
	plt.clf()
	plt.close()


def plot_imgs(imgs, savepath, img_title):
	imgs = np.copy(imgs)
	imgs[imgs == -1] = np.nan
	fig, ax = plt.subplots(1, imgs.shape[0], figsize=(4*imgs.shape[0], 4))
	for i in range(imgs.shape[0]):
		ax[i].imshow(imgs[i].squeeze(), cmap='gray')
		ax[i].axis('off')
	plt.savefig("{}/{}".format(savepath, img_title))
	plt.clf()
	plt.close()


def plot_patch(dem, vmin=None, vmax=None, savepath=None, img_title=None):
	plt.figure(figsize=(4,4))
	plt.imshow(dem, cmap='viridis', vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig("{}/{}".format(savepath, img_title))
	plt.clf()
	plt.close()


def plot_patch_cbar(dem, vmin=None, vmax=None, savepath=None, img_title=None):
	plt.figure(figsize=(4,4))
	orig_im = plt.imshow(dem, cmap='viridis', vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig("{}/{}".format(savepath, img_title + ".png"))
	plt.clf()
	plt.close()

	fig,ax = plt.subplots()
	plt.colorbar(orig_im,ax=ax)
	ax.remove()
	plt.savefig("{}/{}".format(savepath, img_title + "_cbar.png"))
	plt.clf()
	plt.close()