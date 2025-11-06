import math
import torch
import torch.nn as nn
from ...component import Component


class SinusoidalPE(Component):
	def __init__(self, max_seq_len: int):
		super().__init__(name="SinusoidalPE")
		self.max_seq_len = max_seq_len
		self.register_buffer('pe', None, persistent=False)

	def build(self, cfg):
		super().build(cfg)

	def forward(self, x):
		return x
