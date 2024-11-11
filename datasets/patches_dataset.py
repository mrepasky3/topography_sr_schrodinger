import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import random
import scipy
from operator import itemgetter


class ToTensor(object):
	def __call__(self, sample):
		return torch.from_numpy(sample)


class PolarPatchDataset(Dataset):
	def __init__(self, p_flip=0.):
		
		self.p_flip = p_flip
		
		self.resolution = 20
		patch_top_dir = "process_data"
		south_file_name = patch_top_dir+"/south_fibonacci_96px_20MPP_h5/patches_96px_20MPP"
		north_file_name = patch_top_dir+"/north_fibonacci_96px_20MPP_h5/patches_96px_20MPP"

		file_max_range = 99225
		self.n_records = 2*file_max_range
		
		file_size = 5000
		merge_point = 95000
		
		self.idx_mapping = {}
		for i in np.arange(0, file_max_range, file_size):
			if i == merge_point:
				file_range = "{:06d}-{:06d}.hdf5".format(i, file_max_range)
				for j in range(i, file_max_range):
					self.idx_mapping[j] = (j-i, "{}_{}".format(south_file_name, file_range))
					self.idx_mapping[j+file_max_range] = (j-i, "{}_{}".format(north_file_name, file_range))
				break
			else:
				file_range = "{:06d}-{:06d}.hdf5".format(i, i+file_size)
				for j in range(i, i+file_size):
					self.idx_mapping[j] = (j-i, "{}_{}".format(south_file_name, file_range))
					self.idx_mapping[j+file_max_range] = (j-i, "{}_{}".format(north_file_name, file_range))
		
	def transform(self, high_res_dem, imgs):
		high_res_dem = torch.from_numpy(high_res_dem)
		imgs = torch.from_numpy(imgs)
		
		if random.random() < self.p_flip:
			high_res_dem = TF.hflip(high_res_dem)
			imgs = TF.hflip(imgs)

		if random.random() < self.p_flip:
			high_res_dem = TF.vflip(high_res_dem)
			imgs = TF.vflip(imgs)

		return high_res_dem, imgs

	def __len__(self):
		return self.n_records

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		_idx, _file_name = itemgetter(idx)(self.idx_mapping)

		with h5py.File(_file_name, 'r') as f:
			high_res_dem = f['hd_patches'][_idx]
			imgs = f['imgs_patches'][_idx]
		
		high_res_dem, imgs = self.transform(high_res_dem, imgs)

		sample = {'high_res_dem': high_res_dem, "imgs": imgs}

		return sample


class PolarPatchRealisticSun(Dataset):
	def __init__(
		self, 
		p_flip=0.,
		num_img_vals=np.arange(5,31,1),
		lon_bounds=(0,360),
		lat_bounds=(0,90),
		fixed_imgs=False,
		missing_prob=0.0,
		img_selection='no_replacement',
		dem_only=False
		):
		
		self.p_flip = p_flip
		
		self.resolution = 20
		patch_top_dir = 'process_data'
		south_file_name = patch_top_dir+"/south_lowsunele_96px_20MPP_h5/patches_96px_20MPP"
		north_file_name = patch_top_dir+"/north_lowsunele_96px_20MPP_h5/patches_96px_20MPP"

		south_file_name_hiele = patch_top_dir+"/south_highsunele_96px_20MPP_h5/patches_96px_20MPP"
		north_file_name_hiele = patch_top_dir+"/north_highsunele_96px_20MPP_h5/patches_96px_20MPP"

		file_max_range = 99225
		self.n_records = 2*file_max_range
		
		file_size = 1000
		merge_point = 98000
		
		self.idx_mapping = {}
		self.idx_mapping_hisunangle = {}
		for i in np.arange(0, file_max_range, file_size):
			if i == merge_point:
				file_range = "{:06d}-{:06d}.hdf5".format(i, file_max_range)
				for j in range(i, file_max_range):
					self.idx_mapping[j] = (j-i, "{}_{}".format(south_file_name, file_range))
					self.idx_mapping[j+file_max_range] = (j-i, "{}_{}".format(north_file_name, file_range))

					self.idx_mapping_hisunangle[j] = (j-i, "{}_{}".format(south_file_name_hiele, file_range))
					self.idx_mapping_hisunangle[j+file_max_range] = (j-i, "{}_{}".format(north_file_name_hiele, file_range))
				break
			else:
				file_range = "{:06d}-{:06d}.hdf5".format(i, i+file_size)
				for j in range(i, i+file_size):
					self.idx_mapping[j] = (j-i, "{}_{}".format(south_file_name, file_range))
					self.idx_mapping[j+file_max_range] = (j-i, "{}_{}".format(north_file_name, file_range))

					self.idx_mapping_hisunangle[j] = (j-i, "{}_{}".format(south_file_name_hiele, file_range))
					self.idx_mapping_hisunangle[j+file_max_range] = (j-i, "{}_{}".format(north_file_name_hiele, file_range))

		azi_ele_low = np.load(patch_top_dir+'/south_lowsunele_96px_20MPP/azi_ele.npy')
		azi_ele_hi = np.load(patch_top_dir+'/south_highsunele_96px_20MPP/azi_ele.npy')
		self.azi_ele_low_hi = np.concatenate([azi_ele_low, azi_ele_hi])

		lattice_x = np.sin(np.deg2rad(90-self.azi_ele_low_hi[:,1]))*np.cos(np.deg2rad(self.azi_ele_low_hi[:,0]))
		lattice_y = np.sin(np.deg2rad(90-self.azi_ele_low_hi[:,1]))*np.sin(np.deg2rad(self.azi_ele_low_hi[:,0]))
		lattice_z = np.cos(np.deg2rad(90-self.azi_ele_low_hi[:,1]))

		self.lattice_cart = np.concatenate([lattice_x.reshape(-1,1), lattice_y.reshape(-1,1), lattice_z.reshape(-1,1)],axis=-1)
		
		self.num_img_vals = num_img_vals
		self.max_num_imgs = self.num_img_vals[-1]

		weights_num_img_vals = np.ones(self.num_img_vals.shape)
		self.weights_num_img_vals = weights_num_img_vals / weights_num_img_vals.sum()

		self.lon_bounds = lon_bounds
		self.lat_bounds = lat_bounds

		self.fixed_imgs = fixed_imgs
		self.img_selection = img_selection

		self.missing_prob = missing_prob

		self.dem_only = dem_only

	def get_sunpath_cart(self, lon, lat):
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

	def get_slice_idx(self):
		this_angle = np.random.uniform(low=0.,high=180.)
		if np.random.uniform() < 0.5:
			this_origin = np.random.uniform(low=0.,high=95, size=(1,)).repeat(2)
			
			this_slope = (1 - np.tan(np.deg2rad(this_angle))) / (1 + np.tan(np.deg2rad(this_angle)))
		else:
			this_origin = np.random.uniform(low=0.,high=95, size=(1,)).repeat(2)
			this_origin[0] = 95 - this_origin[0]
			
			this_slope = (1 - np.tan(np.deg2rad(this_angle-90))) / (1 + np.tan(np.deg2rad(this_angle-90)))
		
		this_intercept = this_origin[1] - this_slope*this_origin[0]
		
		line_func = lambda x : this_slope * x + this_intercept

		_X, _Y = np.meshgrid(np.arange(96),np.arange(96))
		if np.random.uniform() < 0.5:
			missing_data_mask = (_Y - line_func(_X)) > 0
		else:
			missing_data_mask = (_Y - line_func(_X)) < 0
		missing_data_x_idx, missing_data_y_idx = np.where(missing_data_mask)
		return missing_data_x_idx, missing_data_y_idx

	def get_naclike_slice_idx(self):
		this_angle = np.random.uniform(low=0.,high=180.)
		this_width = np.random.uniform(low=25.,high=50.)
		if np.random.uniform() < 0.5:
			this_origin = np.random.uniform(low=0.,high=95, size=(1,)).repeat(2)
			
			this_slope = (1 - np.tan(np.deg2rad(this_angle))) / (1 + np.tan(np.deg2rad(this_angle)))
		else:
			this_origin = np.random.uniform(low=0.,high=95, size=(1,)).repeat(2)
			this_origin[0] = 95 - this_origin[0]
			
			this_slope = (1 - np.tan(np.deg2rad(this_angle-90))) / (1 + np.tan(np.deg2rad(this_angle-90)))
		
		this_intercept = this_origin[1] - this_slope*this_origin[0]
		
		parallel_intercept_1 = this_width * np.sqrt(np.power(this_slope,2)+1) + this_intercept
		line_func_1 = lambda x : this_slope * x + parallel_intercept_1
		
		parallel_intercept_2 = this_width * np.sqrt(np.power(this_slope,2)+1) - this_intercept
		line_func_2 = lambda x : this_slope * x - parallel_intercept_2

		_X, _Y = np.meshgrid(np.arange(96),np.arange(96))
		missing_data_mask = np.logical_or((_Y - line_func_1(_X)) > 0, (_Y - line_func_2(_X)) < 0)
		missing_data_x_idx, missing_data_y_idx = np.where(missing_data_mask)
		return missing_data_x_idx, missing_data_y_idx
		
	def transform(self, high_res_dem, imgs):
		high_res_dem = torch.from_numpy(high_res_dem)
		
		if random.random() < self.p_flip:
			high_res_dem = TF.hflip(high_res_dem)
			imgs = TF.hflip(imgs)

		if random.random() < self.p_flip:
			high_res_dem = TF.vflip(high_res_dem)
			imgs = TF.vflip(imgs)

		return high_res_dem, imgs

	def transform_dem(self, high_res_dem):
		high_res_dem = torch.from_numpy(high_res_dem)
		
		if random.random() < self.p_flip:
			high_res_dem = TF.hflip(high_res_dem)

		if random.random() < self.p_flip:
			high_res_dem = TF.vflip(high_res_dem)

		return high_res_dem

	def __len__(self):
		return self.n_records

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		if self.dem_only:
			_idx, _file_name = itemgetter(idx)(self.idx_mapping)
			with h5py.File(_file_name, 'r') as f:
				high_res_dem = f['hd_patches'][_idx]

			high_res_dem = self.transform_dem(high_res_dem)

			sample = {'high_res_dem': high_res_dem}

			return sample

		# get trajectory of sun across sky
		lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
		lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
		sunpath_cart = self.get_sunpath_cart(lon, lat)

		# find rendered sun positions 'close enough' to path positions
		min_dist_lattice = scipy.spatial.distance.cdist(self.lattice_cart, sunpath_cart, metric='euclidean').min(axis=1)
		lattice_mask = min_dist_lattice<0.1
		valid_img_idx = np.where(lattice_mask)[0]

		# randomly select some number of images along this path
		if self.fixed_imgs:
			num_imgs = min(self.max_num_imgs, valid_img_idx.shape[0])
			_uniform_idx = np.round(np.linspace(0, valid_img_idx.shape[0] - 1, num_imgs)).astype(int)
			selected_imgs_idx = valid_img_idx[_uniform_idx]
		else:
			if self.img_selection == 'no_replacement':
				num_imgs = min(np.random.choice(self.num_img_vals, p=self.weights_num_img_vals), valid_img_idx.shape[0])
				selected_imgs_idx = np.random.permutation(valid_img_idx)[:num_imgs]
			elif self.img_selection == 'replacement':
				num_imgs = np.random.choice(self.num_img_vals, p=self.weights_num_img_vals)
				selected_imgs_idx = np.random.choice(valid_img_idx, replace=True, size=(num_imgs))
		
		# load all rendered images
		_idx, _file_name = itemgetter(idx)(self.idx_mapping)
		with h5py.File(_file_name, 'r') as f:
			high_res_dem = f['hd_patches'][_idx]
			imgs_loele = f['imgs_patches'][_idx]

		_idx_hiele, _file_name_hiele = itemgetter(idx)(self.idx_mapping_hisunangle)
		with h5py.File(_file_name_hiele, 'r') as f:
			imgs_hiele = f['imgs_patches'][_idx_hiele]

		imgs = np.concatenate([imgs_loele, imgs_hiele], axis=0)

		# uniformly select sun angles along the trajectory, fill in up to max_num_imgs length
		imgs = torch.from_numpy(imgs[selected_imgs_idx])
		if self.missing_prob > 0.:
			missing_data_idx = np.where(np.random.binomial(1,p=self.missing_prob,size=(imgs.shape[0],)))[0]
			for missing_idx in missing_data_idx:
				missing_data_x_idx, missing_data_y_idx = self.get_naclike_slice_idx()
				imgs[missing_idx, missing_data_x_idx, missing_data_y_idx] = -1.
		
		if imgs.shape[0] < self.max_num_imgs:
			filler_imgs = -1*torch.ones_like(imgs[0])[None,...].repeat(self.max_num_imgs - imgs.shape[0], 1, 1)
			imgs = torch.cat([imgs,filler_imgs],dim=0)
		
		high_res_dem, imgs = self.transform(high_res_dem, imgs)

		sample = {'high_res_dem': high_res_dem, "imgs": imgs}

		return sample