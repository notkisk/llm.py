import torch
import torch.nn as nn
from .component import Component

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.components = nn.ModuleList()

    def add(self, component: Component):
        component.build(self.cfg)
        self.components.append(component)
        return self  

    def repeat(self, component_cls, times: int, **kwargs):
        """Repeat-add a component class multiple times.

        Args:
            component_cls: a Component subclass (not an instance)
            times: number of repetitions
            **kwargs: passed to component class constructor
        """
        if times <= 0:
            return self
        for _ in range(times):
            comp = component_cls(**kwargs)
            self.add(comp)
        return self

    def validate(self):
        if not hasattr(self.cfg, 'vocab_size') or self.cfg.vocab_size <= 0:
            raise ValueError("cfg.vocab_size must be a positive integer")
        if not hasattr(self.cfg, 'dim') or self.cfg.dim <= 0:
            raise ValueError("cfg.dim must be a positive integer")

        num_heads = getattr(self.cfg, 'num_heads', None)
        if num_heads is not None:
            if num_heads <= 0:
                raise ValueError("cfg.num_heads must be positive")
            if self.cfg.dim % num_heads != 0:
                raise ValueError("cfg.dim must be divisible by cfg.num_heads")

        names = [c.__class__.__name__ for c in self.components]
        if any(name.lower().endswith('head') for name in names[:-1]):
            raise ValueError("LM head must be the final component")
        if names:
            first = names[0].lower()
            if 'embedding' not in first:
                raise ValueError("First component should be an embedding")

    def forward(self, x):
        for comp in self.components:
            x = comp(x)
        return x
