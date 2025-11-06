import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...component import Component


class SelfAttention(Component):
	def __init__(self, bias: bool = False, dropout: float = 0.0):
		super().__init__(name="SelfAttention")
		self.bias = bias
		self.dropout_p = dropout
		self.norm = None
		self.qkv = None
		self.proj = None
		self.attn_drop = None
		self.proj_drop = None

	def build(self, cfg):
		super().build(cfg)
		head_dim = cfg.dim // cfg.num_heads
		if cfg.dim % cfg.num_heads != 0:
			raise ValueError("cfg.dim must be divisible by cfg.num_heads")
		self.norm = nn.LayerNorm(cfg.dim)
		self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=self.bias)
		self.proj = nn.Linear(cfg.dim, cfg.dim, bias=self.bias)
		self.attn_drop = nn.Dropout(self.dropout_p)
		self.proj_drop = nn.Dropout(self.dropout_p)

	def forward(self, x, mask: torch.Tensor = None):
		return x
