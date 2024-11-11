import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter


def unfold_patches(dem, kernel_size=96, stride=1):
	# take input of size [1, 1, H, W]
	# outputs tensor of size [num_patches, 1, kernel_size, kernel_size]

	num_patches = int(np.ceil((dem.shape[-2]-kernel_size+1)/stride)*np.ceil((dem.shape[-1]-kernel_size+1)/stride))
	
	unfolded_dems = F.unfold(dem, kernel_size=kernel_size, stride=stride)
	unfolded_dems_patched = unfolded_dems.permute(0,2,1).reshape(num_patches, 1, kernel_size, kernel_size)
	return unfolded_dems_patched

def fold_patches(unfolded_dems_patched, output_shape, kernel_size=96, stride=1):
	num_patches = int(np.ceil((output_shape[-2]-kernel_size+1)/stride)*np.ceil((output_shape[-1]-kernel_size+1)/stride))
	assert num_patches == unfolded_dems_patched.shape[0]
	
	folded_dems_patched = F.fold(unfolded_dems_patched.reshape(1, num_patches, kernel_size**2).permute(0,2,1),
								 (output_shape[-2],output_shape[-1]),kernel_size=kernel_size,stride=stride)
	norm_map = F.fold(F.unfold(torch.ones(output_shape).to(unfolded_dems_patched.device),kernel_size=kernel_size,stride=stride),
					  output_shape[-2:],kernel_size=kernel_size ,stride=stride)
	folded_dems_patched /= norm_map
	return folded_dems_patched

def unfold_imgs(imgs, kernel_size=96, stride=1):
	# take input of size [num_img, 1, H, W]
	# outputs tensor of size [num_patches, num_img, kernel_size, kernel_size]

	num_patches = int(np.ceil((imgs.shape[-2]-kernel_size+1)/stride)*np.ceil((imgs.shape[-1]-kernel_size+1)/stride))
	
	unfolded_imgs = F.unfold(imgs, kernel_size=kernel_size, stride=stride)
	unfolded_imgs_patched = unfolded_imgs.reshape(imgs.shape[0], kernel_size, kernel_size, num_patches).permute(3,0,1,2)
	return unfolded_imgs_patched

def fold_filter(filters, output_shape, kernel_size=96, stride=1):
	num_patches = int(np.ceil((output_shape[-2]-kernel_size+1)/stride)*np.ceil((output_shape[-1]-kernel_size+1)/stride))
	assert num_patches == filters.shape[0]
	
	folded_filters = F.fold(filters.reshape(1, num_patches, kernel_size**2).permute(0,2,1),
								 (output_shape[-2],output_shape[-1]),kernel_size=kernel_size,stride=stride)
	return folded_filters

def duplicated_fold_patches_weighted_stdev(duplicated_unfolded_dems_patched, filters, output_shape, device, kernel_size=96, stride=1):
	num_patches = int(np.ceil((output_shape[-2]-kernel_size+1)/stride)*np.ceil((output_shape[-1]-kernel_size+1)/stride))
	assert num_patches == duplicated_unfolded_dems_patched[0].shape[0]

	# compute mean

	folded_agg_weighted_patches = torch.zeros(output_shape).to(device)
	total_weight = torch.zeros(output_shape).to(device)
	total_count = torch.zeros(output_shape).to(device)
	for i in range(len(duplicated_unfolded_dems_patched)):
		weighted_patches = filters * duplicated_unfolded_dems_patched[i]
		
		folded_agg_weighted_patches += F.fold(weighted_patches.reshape(1, num_patches, kernel_size**2).permute(0,2,1),
											  (output_shape[-2],output_shape[-1]),kernel_size=kernel_size,stride=stride)
		total_weight += fold_filter(filters, output_shape, kernel_size=kernel_size, stride=stride)

		total_count += F.fold(F.unfold(torch.ones(output_shape).to(device),kernel_size=kernel_size,stride=stride),
			output_shape[-2:],kernel_size=kernel_size ,stride=stride)
	
	folded_mean_patches = folded_agg_weighted_patches / total_weight


	# compute stddev

	unfolded_mean_patches = F.unfold(folded_mean_patches, kernel_size=kernel_size, stride=stride)
	unfolded_mean_patches = unfolded_mean_patches.permute(0,2,1).reshape(num_patches, 1, kernel_size, kernel_size)

	sq_mean_sub_unfolded_patches = torch.zeros_like(duplicated_unfolded_dems_patched[0])
	for i in range(len(duplicated_unfolded_dems_patched)):
		sq_mean_sub_unfolded_patches += filters*torch.pow(duplicated_unfolded_dems_patched[i] - unfolded_mean_patches,2)
	
	sq_mean_sub_folded_patches = F.fold(sq_mean_sub_unfolded_patches.reshape(1, num_patches, kernel_size**2).permute(0,2,1),
										(output_shape[-2],output_shape[-1]),kernel_size=kernel_size,stride=stride)

	sq_mean_sub_folded_patches /= (total_weight*(total_count-1)/(total_count))
	
	return folded_mean_patches, torch.sqrt(sq_mean_sub_folded_patches)


class UtilityDataset(Dataset):
	def __init__(self, imgs, img_masks, low_res_dems, upsampled_low_res_dems):

		self.imgs = imgs
		self.img_masks = img_masks
		self.low_res_dems = low_res_dems
		self.upsampled_low_res_dems = upsampled_low_res_dems

	def __len__(self):
		return self.low_res_dems.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img = self.imgs[idx]
		img_mask = self.img_masks[idx]
		low_res_dem = self.low_res_dems[idx]
		upsampled_low_res_dem = self.upsampled_low_res_dems[idx]
		
		sample = {
			'img': img,
			'img_mask': img_mask,
			'low_res_dem': low_res_dem,
			'upsampled_low_res_dem' : upsampled_low_res_dem,
		}

		return sample

def get_blurred_grassfire_filter(filt_exp, weight_sigma, num_patches, device):
	filters = np.ones((96,96))
	for i in range(1, filters.shape[0]-1):
		for j in range(1,filters.shape[1]-1):
			filters[i,j] = 1 + min(filters[i-1,j], filters[i,j-1])
	for i in range(1, filters.shape[0]-1)[::-1]:
		for j in range(1,filters.shape[1]-1)[::-1]:
			filters[i,j] = min(filters[i,j],1 + min(filters[i+1,j], filters[i,j+1]))

	filters = np.power(filters, filt_exp)
	filters = gaussian_filter(filters,sigma=weight_sigma)
	filters = torch.from_numpy(filters)
	filters = filters[None,None,...].repeat(num_patches,1,1,1).to(device)
	return filters