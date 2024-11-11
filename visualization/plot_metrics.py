import matplotlib.pyplot as plt
import numpy as np


def plot_train_loss(
	train_epochs,
	train_loss,
	train_batch_loss,
	savepath,
	batches_per_epoch=1,
	save_trajectories=True
	):

	fig, axs = plt.subplots(1,2,figsize=(10,4))

	axs[0].plot(train_epochs, train_loss, color='k')
	axs[0].set_title("Training Loss (Epochs)", fontsize=14)
	axs[0].set_xlabel("Epoch", fontsize=14)
	axs[0].set_yscale('log')

	axs[1].plot(np.arange(len(train_batch_loss)) / batches_per_epoch, train_batch_loss, color='k')
	axs[1].set_title("Training Loss (Batches)", fontsize=14)
	axs[1].set_xlabel("Epoch", fontsize=14)
	axs[1].set_yscale('log')

	plt.tight_layout()
	plt.savefig("{}/training_loss.png".format(savepath))
	plt.clf()
	plt.close()

	if save_trajectories:
		np.save("{}/training_loss.npy".format(savepath),train_loss)
		np.save("{}/training_epochs.npy".format(savepath),train_epochs)


def plot_val_loss(
	val_epochs,
	val_metrics,
	val_size,
	nfe,
	savepath,
	save_trajectories=True,
	dataset="val"
	):

	prefix = dataset + "_"

	num_metrics = len(val_metrics)

	fig, axs = plt.subplots(1,num_metrics,figsize=(num_metrics*5,4))

	for i, key in enumerate(val_metrics.keys()):

		axs[i].plot(val_epochs, val_metrics[key], color='k')
		axs[i].set_title(key, fontsize=14)
		axs[i].set_xlabel("Epoch", fontsize=14)
		axs[i].set_yscale('log')
		
		if save_trajectories:
			np.save("{}/{}{}_size{}_steps{}.npy".format(savepath, prefix, key, val_size, nfe+1), val_metrics[key])

	plt.tight_layout()
	plt.savefig("{}/{}metrics_size{}_steps{}.png".format(savepath, prefix, val_size, nfe+1))
	plt.clf()
	plt.close()

	if save_trajectories:
		np.save("{}/{}size{}_steps{}_epochs.npy".format(savepath, prefix, val_size, nfe+1), val_epochs)


def plot_vae_loss(
	report_steps,
	total_loss,
	nll_loss,
	kl_loss,
	g_loss,
	savepath,
	split='train',
	save_trajectories=True
	):

	fig, axs = plt.subplots(1,4, figsize=(22,4))

	axs[0].plot(report_steps, total_loss, color='k')
	axs[0].set_title("Total Loss ({})".format(split.capitalize()), fontsize=14)
	axs[0].set_xlabel("Iteration", fontsize=14)

	axs[1].plot(report_steps, nll_loss, color='k')
	axs[1].set_title("NLL Loss ({})".format(split.capitalize()), fontsize=14)
	axs[1].set_xlabel("Iteration", fontsize=14)
	axs[1].set_yscale('log')

	axs[2].plot(report_steps, kl_loss, color='k')
	axs[2].set_title("KL Loss ({})".format(split.capitalize()), fontsize=14)
	axs[2].set_xlabel("Iteration", fontsize=14)
	axs[2].set_yscale('log')

	axs[3].plot(report_steps, g_loss, color='k')
	axs[3].set_title("Adversarial Loss ({})".format(split.capitalize()), fontsize=14)
	axs[3].set_xlabel("Iteration", fontsize=14)

	plt.tight_layout()
	plt.savefig("{}/vae_{}_loss_curves.png".format(savepath, split))
	plt.clf()
	plt.close()

	if save_trajectories:
		np.save("{}/vae_{}_report_iters.npy".format(savepath, split),report_steps)
		np.save("{}/vae_{}_total_loss.npy".format(savepath, split),total_loss)
		np.save("{}/vae_{}_nll_loss.npy".format(savepath, split),nll_loss)
		np.save("{}/vae_{}_kl_loss.npy".format(savepath, split),kl_loss)
		np.save("{}/vae_{}_g_loss.npy".format(savepath, split),g_loss)


def plot_disc_loss(
	report_steps,
	disc_loss,
	savepath,
	split='train',
	save_trajectories=True
	):

	fig = plt.figure(figsize=(6,4))

	plt.plot(report_steps, disc_loss, color='k')
	plt.title("Discriminator Loss ({})".format(split.capitalize()), fontsize=14)
	plt.xlabel("Iteration", fontsize=14)

	plt.tight_layout()
	plt.savefig("{}/disc_{}_loss_curve.png".format(savepath, split))
	plt.clf()
	plt.close()

	if save_trajectories:
		np.save("{}/disc_{}_report_iters.npy".format(savepath, split),report_steps)
		np.save("{}/disc_{}_loss.npy".format(savepath, split),disc_loss)