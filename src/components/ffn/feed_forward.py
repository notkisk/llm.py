import torch
import torch.nn as nn
import torch.nn.functional as F
from ...component import Component


class FeedForward(Component):
	def __init__(self, dropout: float = 0.0):
		super().__init__(name="FeedForward")
		self.dropout_p = dropout
		self.norm = None
		self.fc1 = None
		self.fc2 = None
		self.drop = None

	def build(self, cfg):
		super().build(cfg)
		hidden = getattr(cfg, 'hidden', cfg.dim * 4)
		self.norm = nn.LayerNorm(cfg.dim)
		self.fc1 = nn.Linear(cfg.dim, hidden)
		self.fc2 = nn.Linear(hidden, cfg.dim)
		self.drop = nn.Dropout(self.dropout_p)

	def forward(self, x):
		y = self.norm(x)
		y = self.fc1(y)
		y = F.gelu(y)
		y = self.drop(y)
		y = self.fc2(y)
		y = self.drop(y)
		return x + y
