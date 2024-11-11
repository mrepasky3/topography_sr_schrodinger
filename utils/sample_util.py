import torch

from bridge.schrodinger_bridge import space_indices, compute_pred_x0


def get_samples(
	method_config, 
	nfe, 
	network, 
	vae, 
	batch_dict, 
	device, 
	verbose=False, 
	return_sequence=False
	):

	with torch.no_grad():
		nfe = min(method_config.interval-1, nfe)
		steps = space_indices(method_config.interval, nfe+1)
		
		# create log steps
		log_count = len(steps)-1
		log_steps = [steps[i] for i in space_indices(len(steps)-1, log_count)]
		assert log_steps[0] == 0
		if verbose:
			print(f"[DDPM Sampling] steps={method_config.interval}, {log_steps=}!")
		
		def pred_x0_fn(xt, imgs, masks, step):
			step = torch.full((xt.shape[0],), step, dtype=torch.long)
			out = network(
				xt,
				step,
				batch_dict.cond_dems_batch,
				imgs,
				masks
			)
			return compute_pred_x0(method_config.diffusion, step, xt, out)
		
		zs, pred_x0 = method_config.diffusion.ddpm_sampling(
			batch_dict.imgs_batch,
			batch_dict.masks_batch,
			steps,
			pred_x0_fn,
			batch_dict.latent_cond_dems_batch,
			log_steps=log_steps,
			verbose=verbose,
		)

		if return_sequence:
			# start from degraded (decoded) input, end at improved (decoded) output
			x_recon = []
			for i in range(zs.shape[1])[::-1]:
				x_recon.append(vae.module.decode(zs[:,i,...].to(device)).unsqueeze(1))
			x_recon = torch.cat(x_recon,dim=1) # [batch, seq, 1, 96, 96]
		else:
			# just return the improved (and decoded) output
			z_recon = zs[:,0,...].to(device)
			x_recon = vae.module.decode(z_recon) # [batch, 1, 96, 96]

		return x_recon