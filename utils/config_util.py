import pickle
import ml_collections
import json
import os

def save_config(path, config, overwrite=False):
	if not overwrite:
		if not os.path.exists("{}/config".format(path)):
			with open("{}/config".format(path), "wb") as f:
				pickle.dump(config, f)
	else:
		with open("{}/config".format(path), "wb") as f:
			pickle.dump(config, f)

def dump_config(config, path):
	with open(path, 'w') as f:
		for key in config.keys():
			if type(config[key]) == ml_collections.ConfigDict:
				for sub_key in config[key]:
					f.write("{}.{} = {}\n".format(key, sub_key, config[key][sub_key]))

def load_config(path, default=None):
	if os.path.exists("{}/config".format(path)):
		with open("{}/config".format(path), "rb") as f:
			return pickle.load(f)
	else:
		assert default is not None
		return default

def dump_args(args, path):
	argdict = vars(args)
	with open("{}/argdict".format(path), "wb") as f:
		pickle.dump(argdict, f)
	with open("{}/argdict.txt".format(path), "w") as f:
		json.dump(args.__dict__, f)

def merge_args(arg_dict, parser):
	'''
	Merge arguments from a dictionary
	with those from argparse, and
	return a new argparse Namespace
	'''
	args_inp = parser.parse_args()
	args_inp_vars = vars(args_inp)
	for key in arg_dict:
		if key in args_inp:
			if key != "num_plots" and key != "nfe_plot" and key != "nfe_val" and key != "val_size":
				assert arg_dict[key] == args_inp_vars[key], '{} must be {}'.format(key, arg_dict[key])
		else:
			parser.add_argument('--'+key, default=arg_dict[key])
	return parser.parse_args()