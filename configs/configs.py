import ml_collections


def get_set_dit_config():
	config = ml_collections.ConfigDict()

	# DEM model
	config.model = model = ml_collections.ConfigDict()
	model.context_upsample_type = "bicubic"
	model.name = "Diffusion Transformer"
	model.hidden_size = 768
	model.patch_size = 2
	model.depth = 6
	model.num_heads = 12
	
	model.cond_patch_size = 2
	model.cond_depth = 6
	model.cond_num_heads = 12

	model.set_conv_model_channels = 4
	model.set_conv_out_channels = 4
	model.set_conv_num_res_blocks = 2
	model.set_conv_channel_mult = (1,1,2,2)
	model.set_conv_resblock_updown = True
	model.set_conv_norm_mode = "half"
	model.set_conv_pool_type = "mean"
	model.set_patch_size = 2
	model.set_depth = 6
	model.set_num_heads = 12

	config.bridge = bridge = ml_collections.ConfigDict()
	bridge.interval = 1000
	bridge.beta_max = 0.3

	# optimization
	config.optim = optim = ml_collections.ConfigDict()
	optim.epochs = 1000
	optim.batch_size = 256
	optim.lr = 1e-4
	optim.weight_decay = 0.0
	optim.p_flip = 0.5

	return config


def get_vae_config():
	config = ml_collections.ConfigDict()

	# DEM model
	config.model = model = ml_collections.ConfigDict()
	model.name = "VAE"
	model.base_channels = 64
	model.channel_mult = [1,2,4,4]
	model.num_res_blocks = 2
	model.attention_resolutions = []
	model.dropout = 0.0
	model.z_channels = 4
	model.embed_dim = 4

	# optimization
	config.optim = optim = ml_collections.ConfigDict()
	optim.epochs = 3000
	optim.batch_size = 64
	optim.lr = 4.5e-6
	optim.weight_decay = 0.0
	optim.disc_start = 5000
	optim.kl_weight = 0.000001
	optim.disc_weight = 0.5
	optim.disc_num_layers = 4
	optim.alternate_period = 10
	optim.vae_alternate_iter = 5
	optim.p_flip = 0.5

	return config