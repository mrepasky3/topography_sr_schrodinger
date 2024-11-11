# ---------------------------------------------------------------
# Adapted from DiT repository: https://github.com/facebookresearch/DiT
#
# William Peebles and Saining Xie. "Scalable diffusion models with transformers." 
# In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# U-net components adapted from guided diffusion repository: https://github.com/openai/guided-diffusion
#
# Prafulla Dhariwal and Alexander Nichol. "Diffusion models beat gans on image synthesis."
# Advances in neural information processing systems 34 (2021): 8780-8794.
# ---------------------------------------------------------------


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
	return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                       Embedding Layers for Timesteps                          #
#################################################################################

class TimestepEmbedder(nn.Module):
	"""
	Embeds scalar timesteps into vector representations.
	"""
	def __init__(self, hidden_size, frequency_embedding_size=256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t, dim, max_period=10000):
		"""
		Create sinusoidal timestep embeddings.
		:param t: a 1-D Tensor of N indices, one per batch element.
						  These may be fractional.
		:param dim: the dimension of the output.
		:param max_period: controls the minimum frequency of the embeddings.
		:return: an (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
		).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t):
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb


#################################################################################
#                            Convolutional Encoder                              #
#################################################################################

class GroupNormHalf(nn.GroupNorm):
	def forward(self, x):
		return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
	def forward(self, x):
		return super().forward(x.float()).type(x.dtype)


def normalization(channels, norm_mode="half"):
	"""
	Make a standard normalization layer.

	:param channels: number of input channels.
	:return: an nn.Module for normalization.
	"""
	if norm_mode == "half":
		return GroupNormHalf(channels//2, channels)
	elif norm_mode == "32":
		return GroupNorm32(32, channels)

def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


def set_pool(x_set, mask, pool_type="mean"):
	bs, ss, _, _, _ = x_set.shape
	if pool_type == 'max':
		m = (1-mask.view(bs, ss, 1, 1, 1))*-1e10
		z = x_set + m
		z = z.max(dim=1)[0]
	elif pool_type == 'mean':
		m = mask.view(bs, ss, 1, 1, 1)
		z = x_set * m
		zsum = z.sum(dim=1)
		denominator = torch.max(mask.sum(axis=1),torch.tensor(1.)) # number of 'kept' images per set
		z = zsum / denominator.view(-1,1,1,1)
	return z


class Upsample(nn.Module):
	"""
	An upsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	"""

	def __init__(self, channels, use_conv, out_channels=None):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		if use_conv:
			self.conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

	def forward(self, x):
		assert x.shape[1] == self.channels
		x = F.interpolate(x, scale_factor=2, mode="nearest")
		if self.use_conv:
			x = self.conv(x)
		return x


class Downsample(nn.Module):
	"""
	A downsampling layer with an optional convolution.

	:param channels: channels in the inputs and outputs.
	:param use_conv: a bool determining if a convolution is applied.
	"""

	def __init__(self, channels, use_conv, out_channels=None):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		if use_conv:
			self.op = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=1)
		else:
			assert self.channels == self.out_channels
			self.op = nn.AvgPool2d(2)

	def forward(self, x):
		assert x.shape[1] == self.channels
		return self.op(x)


class ResBlock(nn.Module):
	"""
	A residual block that can optionally change the number of channels.

	:param channels: the number of input channels.
	:param dropout: the rate of dropout.
	:param out_channels: if specified, the number of out channels.
	:param use_conv: if True and out_channels is specified, use a spatial
		convolution instead of a smaller 1x1 convolution to change the
		channels in the skip connection.
	:param up: if True, use this block for upsampling.
	:param down: if True, use this block for downsampling.
	"""

	def __init__(
		self,
		channels,
		dropout,
		out_channels=None,
		use_conv=False,
		up=False,
		down=False,
		norm_mode="half",
	):
		super().__init__()
		self.channels = channels
		self.dropout = dropout
		self.out_channels = out_channels or channels
		self.use_conv = use_conv

		self.in_layers = nn.Sequential(
			normalization(channels, norm_mode=norm_mode),
			nn.SiLU(),
			nn.Conv2d(channels, self.out_channels, 3, padding=1),
		)

		self.updown = up or down

		if up:
			self.h_upd = Upsample(channels, False)
			self.x_upd = Upsample(channels, False)
		elif down:
			self.h_upd = Downsample(channels, False)
			self.x_upd = Downsample(channels, False)
		else:
			self.h_upd = self.x_upd = nn.Identity()

		self.out_layers = nn.Sequential(
			normalization(self.out_channels, norm_mode=norm_mode),
			nn.SiLU(),
			nn.Dropout(p=dropout),
			zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
		)

		if self.out_channels == channels:
			self.skip_connection = nn.Identity()
		elif use_conv:
			self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
		else:
			self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

	def forward(self, x):
		if self.updown:
			in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
			h = in_rest(x)
			h = self.h_upd(h)
			x = self.x_upd(x)
			h = in_conv(h)
		else:
			h = self.in_layers(x)
		h = self.out_layers(h)
		return self.skip_connection(x) + h


class SetConvolutionalEncoder(nn.Module):

	def __init__(
		self,
		in_channels,
		model_channels,
		out_channels,
		num_res_blocks,
		dropout=0,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		use_fp16=False,
		resblock_updown=False,
		norm_mode="half",
		pool_type="mean",
	):
		super().__init__()

		self.in_channels = in_channels
		self.model_channels = model_channels
		self.out_channels = out_channels
		self.num_res_blocks = num_res_blocks
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.dtype = torch.float16 if use_fp16 else torch.float32
		self.norm_mode = norm_mode
		self.pool_type = pool_type
		
		ch = input_ch = int(channel_mult[0] * model_channels)
		self.input_blocks = nn.ModuleList(
			[nn.Sequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
		)

		input_block_chans = [ch]
		ds = 1
		for level, mult in enumerate(channel_mult):
			
			for _ in range(num_res_blocks):
				layers = [ResBlock(ch, dropout, out_channels=int(mult * model_channels), norm_mode=self.norm_mode)]
				ch = int(mult * model_channels)
				self.input_blocks.append(nn.Sequential(*layers))
				input_block_chans.append(ch)
			
			if level != len(channel_mult) - 1:
				out_ch = ch
				if resblock_updown:
					self.input_blocks.append(nn.Sequential(ResBlock(ch, dropout, out_channels=out_ch, down=True, norm_mode=self.norm_mode)))
				else:
					self.input_blocks.append(nn.Sequential(Downsample(ch, conv_resample, out_channels=out_ch)))
				ch = out_ch
				input_block_chans.append(ch)
				ds *= 2

		self.out_conv = nn.Sequential(
			normalization(input_block_chans[-1], norm_mode=self.norm_mode),
			nn.SiLU(),
			zero_module(nn.Conv2d(input_block_chans[-1], out_channels, 3, padding=1)),
		)

	def forward(self, x_set, mask):
		"""
		Apply the model to an input batch.

		:param x_set: an [N x S x C x H x W] Tensor of inputs.
		:param mask: an [N x S] Tensor of binary labels.
		:return: an [N x C_out x H x W] Tensor of outputs.
		"""

		bs, ss, _, _, _ = x_set.shape

		h_set = x_set.type(self.dtype)
		for module in self.input_blocks:
			_, _, oc, oh, ow = h_set.shape
			h_set = module(h_set.view(bs*ss, oc, oh, ow))
			_, nc, nh, nw = h_set.shape
			h_set = h_set.view(bs, ss, nc, nh, nw)
			
		h = set_pool(h_set,mask,pool_type=self.pool_type)   # (N x C x h x w)
		h = h.type(x_set.dtype)
		h = self.out_conv(h)                                # (N x C_out x h x w)
		return h


#################################################################################
#                                 ViT Encoder                                   #
#################################################################################

class ViTBlock(nn.Module):
	"""
	A ViT block.
	"""
	def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, **block_kwargs):
		super().__init__()
		self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout, **block_kwargs)
		self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		mlp_hidden_dim = int(hidden_size * mlp_ratio)
		approx_gelu = lambda: nn.GELU(approximate="tanh")
		self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x

class ViTFinalLayer(nn.Module):
	"""
	The final layer of ViT.
	"""
	def __init__(self, hidden_size, out_size):
		super().__init__()
		self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.linear = nn.Linear(hidden_size, out_size, bias=True)

	def forward(self, x):
		x = self.norm_final(x)
		x = self.linear(x)
		return x

class ViT(nn.Module):
	def __init__(
		self,
		input_size=32,
		patch_size=2,
		in_channels=4,
		hidden_size=1152,
		depth=28,
		num_heads=16,
		mlp_ratio=4.0,
		dropout=0.0,
	):
		super().__init__()
		self.in_channels = in_channels
		self.patch_size = patch_size
		self.num_heads = num_heads

		self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
		num_patches = self.x_embedder.num_patches
		# Will use fixed sin-cos embedding:
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
		
		self.out_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)

		self.blocks = nn.ModuleList([
			ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
		])
		self.final_layer = ViTFinalLayer(hidden_size, hidden_size)
		self.initialize_weights()

	def initialize_weights(self):
		# Initialize transformer layers:
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				torch.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
		self.apply(_basic_init)

		# Initialize (and freeze) pos_embed by sin-cos embedding:
		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

		# Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
		w = self.x_embedder.proj.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		nn.init.constant_(self.x_embedder.proj.bias, 0)

		# Zero-out output layers:
		nn.init.constant_(self.final_layer.linear.weight, 0)
		nn.init.constant_(self.final_layer.linear.bias, 0)

	def forward(self, x):
		"""
		Forward pass of DiT.
		x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
		"""
		x = self.x_embedder(x) + self.pos_embed                             # (N, T, D), where T = H * W / patch_size ** 2
		x = torch.cat([self.out_embed.repeat((x.shape[0],1,1)),x],dim=1)    # (N, T+1, D)
		for block in self.blocks:
			x = block(x)                      # (N, T+1, D)
		x = x[:,0]                            # (N, D)
		x = self.final_layer(x)               # (N, D)
		return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
	"""
	A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
	"""
	def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, **block_kwargs):
		super().__init__()
		self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout, **block_kwargs)
		self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		mlp_hidden_dim = int(hidden_size * mlp_ratio)
		approx_gelu = lambda: nn.GELU(approximate="tanh")
		self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 6 * hidden_size, bias=True)
		)

	def forward(self, x, c):
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
		x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
		x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
		return x


class FinalLayer(nn.Module):
	"""
	The final layer of DiT.
	"""
	def __init__(self, hidden_size, patch_size, out_channels):
		super().__init__()
		self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 2 * hidden_size, bias=True)
		)

	def forward(self, x, c):
		shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
		x = modulate(self.norm_final(x), shift, scale)
		x = self.linear(x)
		return x


class SetDiT(nn.Module):
	"""
	Diffusion model with a Transformer backbone.
	"""
	def __init__(
		self,
		input_size=32,
		patch_size=2,
		in_channels=4,
		hidden_size=1152,
		depth=28,
		num_heads=16,
		mlp_ratio=4.0,
		
		cond_input_size=32,
		cond_patch_size=8,
		cond_in_channels=1,
		cond_depth=12,
		cond_num_heads=12,

		set_input_size=96,
		set_patch_size=2,
		set_in_channels=1,
		set_depth = 12,
		set_num_heads=12,
		set_conv_model_channels=4,
		set_conv_out_channels=4,
		set_conv_num_res_blocks=2,
		set_conv_channel_mult=(1,1,2,2),
		set_conv_resblock_updown=True,
		set_conv_norm_mode="half",
		set_conv_pool_type="mean",
		
		learn_sigma=True,
		dropout=0.,
	):
		super().__init__()
		self.learn_sigma = learn_sigma
		self.in_channels = in_channels
		self.out_channels = in_channels * 2 if learn_sigma else in_channels
		self.patch_size = patch_size
		self.num_heads = num_heads

		self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
		self.t_embedder = TimestepEmbedder(hidden_size)
		self.y_embedder = ViT(
			input_size=cond_input_size,
			patch_size=cond_patch_size,
			in_channels=cond_in_channels,
			hidden_size=hidden_size,
			depth=cond_depth,
			num_heads=cond_num_heads,
			mlp_ratio=mlp_ratio,
			dropout=dropout
			)
		self.y_set_conv_embedder = SetConvolutionalEncoder(
			in_channels=set_in_channels,
			model_channels=set_conv_model_channels,
			out_channels=set_conv_out_channels,
			num_res_blocks=set_conv_num_res_blocks,
			dropout=dropout,
			channel_mult=set_conv_channel_mult,
			conv_resample=True,
			resblock_updown=set_conv_resblock_updown,
			norm_mode=set_conv_norm_mode,
			pool_type=set_conv_pool_type,
			)
		self.y_set_vit_embedder = ViT(
			input_size=set_input_size//(2**(len(set_conv_channel_mult)-1)),
			patch_size=set_patch_size,
			in_channels=set_conv_out_channels,
			hidden_size=hidden_size,
			depth=set_depth,
			num_heads=set_num_heads,
			mlp_ratio=mlp_ratio,
			dropout=dropout
			)
		num_patches = self.x_embedder.num_patches
		# Will use fixed sin-cos embedding:
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

		self.blocks = nn.ModuleList([
			DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
		])
		self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
		self.initialize_weights()

	def initialize_weights(self):
		# Initialize transformer layers:
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				torch.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
		self.apply(_basic_init)

		# Initialize (and freeze) pos_embed by sin-cos embedding:
		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

		# Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
		w = self.x_embedder.proj.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		nn.init.constant_(self.x_embedder.proj.bias, 0)

		# Initialize timestep embedding MLP:
		nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
		nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

		# Zero-out adaLN modulation layers in DiT blocks:
		for block in self.blocks:
			nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

		# Zero-out output layers:
		nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
		nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
		nn.init.constant_(self.final_layer.linear.weight, 0)
		nn.init.constant_(self.final_layer.linear.bias, 0)

	def unpatchify(self, x):
		"""
		x: (N, T, patch_size**2 * C)
		imgs: (N, H, W, C)
		"""
		c = self.out_channels
		p = self.x_embedder.patch_size[0]
		h = w = int(x.shape[1] ** 0.5)
		assert h * w == x.shape[1]

		x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
		return imgs

	def forward(self, x, t, y, y_set, y_set_mask):
		"""
		Forward pass of DiT.
		x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
		t: (N,) tensor of diffusion timesteps
		y: (N, _C, _H, _W) tensor of conditional labels
		y_set: (N, S, __C, __H, __W) tensor of set of conditional labels
		y_set_mask: (N, S) tensor of set mask labels
		"""
		x = self.x_embedder(x) + self.pos_embed                                               # (N, T, D), where T = H * W / patch_size ** 2
		t = self.t_embedder(t)                                                                # (N, D)
		y = self.y_embedder(y)                                                                # (N, D)
		y_set = self.y_set_vit_embedder(self.y_set_conv_embedder(y_set, y_set_mask))          # (N, D)
		c = t + y + y_set                                                                     # (N, D)
		for block in self.blocks:
			x = block(x, c)                      # (N, T, D)
		x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
		x = self.unpatchify(x)                   # (N, out_channels, H, W)
		return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
	"""
	grid_size: int of the grid height and width
	return:
	pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
	"""
	grid_h = np.arange(grid_size, dtype=np.float32)
	grid_w = np.arange(grid_size, dtype=np.float32)
	grid = np.meshgrid(grid_w, grid_h)  # here w goes first
	grid = np.stack(grid, axis=0)

	grid = grid.reshape([2, 1, grid_size, grid_size])
	pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
	if cls_token and extra_tokens > 0:
		pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
	return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
	assert embed_dim % 2 == 0

	# use half of dimensions to encode grid_h
	emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
	emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

	emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
	return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
	"""
	embed_dim: output dimension for each position
	pos: a list of positions to be encoded: size (M,)
	out: (M, D)
	"""
	assert embed_dim % 2 == 0
	omega = np.arange(embed_dim // 2, dtype=np.float64)
	omega /= embed_dim / 2.
	omega = 1. / 10000**omega  # (D/2,)

	pos = pos.reshape(-1)  # (M,)
	out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

	emb_sin = np.sin(out) # (M, D/2)
	emb_cos = np.cos(out) # (M, D/2)

	emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
	return emb