import pickle
import argparse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.patches_dataset import PolarPatchRealisticSun
from utils.config_util import merge_args
from utils.init_util import get_path, load_vae, init_model_and_method, split_dataset, init_up_down_samplers
from utils.val_util import get_validation_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=int)
parser.add_argument('--model_epoch', type=int, default=5)

parser.add_argument('--resolution', type=int, default=20) # in meters
parser.add_argument('--downsample', type=int, default=16)

parser.add_argument('--dem_dataset', type=str, default="bettersun", choices=["bettersun","polar"]) # dataset model was trained on

parser.add_argument('--num_imgs', type=int, default=20)
parser.add_argument('--img_selection', type=str, default="replacement", choices=["no_replacement","replacement"])
parser.add_argument('--missing_prob', type=float, default=1.0)
parser.add_argument('--lat_lower', type=float, default=0)
parser.add_argument('--lat_upper', type=float, default=90)

parser.add_argument('--nfe', type=int, default=10)
parser.add_argument('--n_repeat', type=int, default=10)

parser.add_argument('--val_size', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=250)


parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))

savepath = get_path(args, create=False)

with open(savepath+"/argdict", 'rb') as f:
	arg_dict = pickle.load(f)
arg_dict['nfe'] = args.nfe
arg_dict['img_selection'] = args.img_selection
arg_dict['missing_prob'] = args.missing_prob
args = merge_args(arg_dict, parser)


# load the pre-trained model (and vae)
vae = load_vae(args, device)
srmodel, config, method_config = init_model_and_method(args, savepath, device, load_epoch=args.model_epoch)
srmodel.eval()


# force load this dataset (not necessarily same as dem_dataset)
patch_dataset_raw = PolarPatchRealisticSun(
	p_flip=0.,
	num_img_vals=np.arange(args.num_imgs,args.num_imgs+1,1),
	lat_bounds=(args.lat_lower, args.lat_upper),
	missing_prob=args.missing_prob,
	img_selection=args.img_selection
)

test_idx = np.load("process_data/thresh2242_test_idx.npy")
patch_dataset_test = torch.utils.data.Subset(patch_dataset_raw, torch.tensor(test_idx))

batch_size = args.batch_size//args.n_repeat
patch_dataloader_test = DataLoader(patch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

downsampler, upsampler = init_up_down_samplers(args.downsample, mode=config.model.context_upsample_type, inp_size=96)


# compute validation metrics on the test dataset
metric_dict = get_validation_metrics(
	args=args,
	method_config=method_config,
	network=srmodel,
	downsampler=downsampler,
	upsampler=upsampler,
	vae=vae,
	dataloader=patch_dataloader_test,
	device=device,
	verbose=args.verbose
)

filename = "evaluation_latrange{}-{}_numimgs{}_{}_missing{:.1f}_testsize{}_steps{}_nrepeat{}_epoch{:03d}".format(
	args.lat_lower,
	args.lat_upper,
	args.num_imgs,
	args.img_selection,
	args.missing_prob,
	args.val_size,
	args.nfe+1,
	args.n_repeat,
	args.model_epoch
)

df_path = "{}/{}.csv".format(savepath, filename)
pd.DataFrame(metric_dict).to_csv(df_path,index=False)