import torch
import torch.nn as nn
from ...component import Component


class RotaryPE(Component):
	def __init__(self):
		super().__init__(name="RotaryPE")

	def build(self, cfg):
		super().build(cfg)

	def forward(self, x):
		return x
