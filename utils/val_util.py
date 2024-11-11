import numpy as np
from tqdm import tqdm

import torch
from torcheval import metrics

from .sample_util import get_samples
from .opt_util import get_batch, get_repeated_batch


def init_validation_metric_dict():
	val_metrics = {}
	
	val_metrics["FID"] = []

	val_metrics["RMSE Elevation"] = []
	val_metrics["Elev. 75th Percentile"] = []
	val_metrics["Elev. 95th Percentile"] = []

	val_metrics["Slope x Error"] =  []
	val_metrics["Slope y Error"] =  []

	val_report_epochs = []

	return val_report_epochs, val_metrics


def update_validation_metrics(running_val_metrics, val_dict):
	for key in val_dict:
		running_val_metrics[key] += val_dict[key]


def get_validation_metrics(
	args,
	method_config,
	network,
	downsampler,
	upsampler,
	vae,
	dataloader,
	device,
	verbose=False
	):

	dataloader = tqdm(dataloader) if verbose else dataloader
	
	val_dict = {}

	n_repeat = args.n_repeat if "n_repeat" in args else 1
	bounded = args.bounded if "bounded" in args else False
	img_bound = args.img_bound if "img_bound" in args else None
	height_scaling = 1000.

	fid_metric = metrics.FrechetInceptionDistance()

	ele_num_pix = 0
	ele_sum_squares = 0.
	ele_bins = torch.linspace(0,1,100000)
	ele_bins_populated = torch.zeros(100000-1)
	
	slope_num_pix = 0
	slope_x_sum_degrees = 0.
	slope_y_sum_degrees = 0.

	num_tested_points = 0	
	for batch in dataloader:
		batch_dict = get_repeated_batch(n_repeat, batch, downsampler, upsampler, vae, device, bounded=bounded, img_bound=img_bound)

		if method_config == None:
			recon_dems_batch = vae.module.decode(batch_dict.latent_cond_dems_batch) if args.latent else batch_dict.cond_dems_batch_upsampled

		else:
			recon_dems_batch = get_samples(method_config, args.nfe, network, vae, batch_dict, device)

			if n_repeat > 1:
				n, c, h, w = recon_dems_batch.size()
				recon_dems_batch = recon_dems_batch.reshape(n//n_repeat, n_repeat, c, h, w).mean(1)

		n, c, h, w = batch_dict.dems_batch.size()
		
		# Update FID
		recon_dems_batch_mins = recon_dems_batch.reshape(n,-1).min(1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		recon_dems_batch_maxs = recon_dems_batch.reshape(n,-1).max(1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		rescaled_recon_dems_batch = (recon_dems_batch - recon_dems_batch_mins) / (recon_dems_batch_maxs - recon_dems_batch_mins)

		dems_batch_mins = batch_dict.dems_batch.reshape(n,-1).min(1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		dems_batch_maxs = batch_dict.dems_batch.reshape(n,-1).max(1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		rescaled_dems_batch = (batch_dict.dems_batch - dems_batch_mins) / (dems_batch_maxs - dems_batch_mins)

		fid_metric.update(rescaled_recon_dems_batch.repeat(1,3,1,1), is_real=False)
		fid_metric.update(rescaled_dems_batch.repeat(1,3,1,1), is_real=True)

		
		# Compute elevation errors
		elevation_resids = (recon_dems_batch - batch_dict.dems_batch).flatten()
		ele_square_resids = torch.pow(elevation_resids,2)
		ele_sum_squares += ele_square_resids.sum()

		ele_abs_resids = torch.abs(elevation_resids)
		ele_bins_populated += torch.histogram(ele_abs_resids.cpu(),ele_bins)[0]

		ele_num_pix += elevation_resids.shape[0]
		
		
		# Compute slope errors
		target_slope_x, target_slope_y = torch.gradient(batch_dict.dems_batch,dim=[-1,-2])
		out_slope_x, out_slope_y = torch.gradient(recon_dems_batch,dim=[-1,-2])

		slope_x_resids = torch.abs((out_slope_x[:,:,1:-1,1:-1] - target_slope_x[:,:,1:-1,1:-1]).flatten())
		slope_x_degrees = torch.arctan(slope_x_resids / (args.resolution / height_scaling)) * (180 / torch.pi)
		slope_x_sum_degrees += slope_x_degrees.sum()
		
		slope_y_resids = torch.abs((out_slope_y[:,:,1:-1,1:-1] - target_slope_y[:,:,1:-1,1:-1]).flatten())
		slope_y_degrees = torch.arctan(slope_y_resids / (args.resolution / height_scaling)) * (180 / torch.pi)
		slope_y_sum_degrees += slope_y_degrees.sum()

		slope_num_pix += slope_y_resids.shape[0]


		num_tested_points += n
		if num_tested_points >= args.val_size:
			break
	
	val_dict["FID"] = [float(fid_metric.compute())]

	val_dict["RMSE Elevation"] = [np.power(float(ele_sum_squares / ele_num_pix), 0.5)]

	abs_ele_error_pdf = ele_bins_populated / ele_num_pix
	abs_ele_error_cdf = torch.cumsum(abs_ele_error_pdf, dim=0)

	val_dict["Elev. 75th Percentile"] = [float(ele_bins[int((abs_ele_error_cdf > 0.75).nonzero()[0])])]
	val_dict["Elev. 95th Percentile"] = [float(ele_bins[int((abs_ele_error_cdf > 0.95).nonzero()[0])])]

	val_dict["Slope x Error"] =  [float(slope_x_sum_degrees / slope_num_pix)]
	val_dict["Slope y Error"] =  [float(slope_y_sum_degrees / slope_num_pix)]

	return val_dict