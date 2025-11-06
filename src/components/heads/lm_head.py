import torch
import torch.nn as nn
from ...component import Component


class LMHead(Component):
	def __init__(self, tie_weights: bool = True):
		super().__init__(name="LMHead")
		self.tie_weights = tie_weights
		self.proj = None
		self._tied = False

	def build(self, cfg):
		super().build(cfg)
		self.proj = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
		if self.tie_weights and hasattr(self, 'weight_to_tie') and self.weight_to_tie is not None:
			self.proj.weight = self.weight_to_tie
			self._tied = True

	def forward(self, x):
		return self.proj(x)
