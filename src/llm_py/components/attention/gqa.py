"""
Grouped Query Attention (GQA) implementation.
Configurable number of KV heads between MHA and MQA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...component import Component
from ..pos.rotary import apply_rotary_pos_emb
from .sliding_window import create_sliding_window_mask, apply_rolling_buffer

class GroupedQueryAttention(Component):
	"""Grouped Query Attention: configurable number of KV heads.
	
	Args:
		num_kv_heads: Number of key/value heads (must divide num_heads)
		bias: Whether to use bias in linear layers
		dropout: Attention dropout probability
		window_size: Sliding window size for local attention (None = full attention)
	"""
	def __init__(self, num_kv_heads: int = None, bias: bool = False, dropout: float = 0.0, window_size=None):
		super().__init__(name="GroupedQueryAttention")
		self.num_kv_heads = num_kv_heads
		self.bias = bias
		self.dropout_p = dropout
		self.window_size = window_size
		self.q_proj = None
		self.k_proj = None
		self.v_proj = None
		self.proj = None
		self.attn_drop = None
		self.proj_drop = None

	def build(self, cfg):
		super().build(cfg)
		if cfg.dim % cfg.num_heads != 0:
			raise ValueError("cfg.dim must be divisible by cfg.num_heads")

		if self.num_kv_heads is None:
			self.num_kv_heads = cfg.num_heads

		if cfg.num_heads % self.num_kv_heads != 0:
			raise ValueError(f"num_heads ({cfg.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

		self.head_dim = cfg.dim // cfg.num_heads
		self.scale = self.head_dim ** -0.5
		self.num_groups = cfg.num_heads // self.num_kv_heads

		self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=self.bias)
		kv_dim = self.num_kv_heads * self.head_dim
		self.k_proj = nn.Linear(cfg.dim, kv_dim, bias=self.bias)
		self.v_proj = nn.Linear(cfg.dim, kv_dim, bias=self.bias)
		self.proj = nn.Linear(cfg.dim, cfg.dim, bias=self.bias)
		self.attn_drop = nn.Dropout(self.dropout_p)
		self.proj_drop = nn.Dropout(self.dropout_p)

	def forward(self, x, mask: torch.Tensor = None, past_key_value = None, rotary_pos_emb = None, attention_bias = None, **kwargs):
		"""Forward pass.
		
		Args:
			x: Input tensor of shape (batch, seq_len, dim)
			mask: Optional attention mask
			past_key_value: Optional tuple of (past_k, past_v) for KV caching
		
		Returns:
			Tuple of (output tensor with residual connection, current_key_value)
		"""
		B, T, C = x.shape
		if C != self.cfg.dim:
			raise ValueError(f"Input dimension mismatch: {C} != {self.cfg.dim}")

		q = self.q_proj(x)  # (batch, seq_len, dim)
		k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
		v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)

		q = q.view(B, T, self.cfg.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
		k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (batch, kv_heads, seq_len, head_dim)
		v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

		# Apply RoPE
		if rotary_pos_emb is not None:
			cos, sin = rotary_pos_emb
			q = apply_rotary_pos_emb(q, cos, sin)
			k = apply_rotary_pos_emb(k, cos, sin)

		# Apply rolling buffer for KV cache management (before repeat_interleave for efficiency)
		if self.window_size is not None:
			k, v, current_key_value = apply_rolling_buffer(k, v, past_key_value, self.window_size)
		else:
			# Standard KV cache concatenation
			if past_key_value is not None:
				past_k, past_v = past_key_value
				k = torch.cat([past_k, k], dim=2)
				v = torch.cat([past_v, v], dim=2)
			current_key_value = (k, v)

		# Expand KV heads to match query heads
		k = k.repeat_interleave(self.num_groups, dim=1)  # (batch, heads, seq_len, head_dim)
		v = v.repeat_interleave(self.num_groups, dim=1)

		attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, seq_len)
		
		if attention_bias is not None:
			attn = attn + attention_bias

		# Apply masking
		T_k = k.size(2)
		if self.window_size is not None:
			sliding_mask = create_sliding_window_mask(T, T_k, self.window_size, x.device)
			attn = attn.masked_fill(sliding_mask, float('-inf'))
		else:
			causal_mask = torch.triu(torch.ones(T, T_k, device=x.device, dtype=torch.bool), diagonal=1 + T_k - T)
			attn = attn.masked_fill(causal_mask, float('-inf'))

		if mask is not None:
			if mask.dim() == 2:
				mask = mask.unsqueeze(1).unsqueeze(1)
			attn = attn.masked_fill(mask == 0, float('-inf'))

		attn = F.softmax(attn, dim=-1)
		attn = self.attn_drop(attn)

		out = attn @ v  # (batch, heads, seq_len, head_dim)
		
		out = out.transpose(1, 2).contiguous().view(B, T, C)
		out = self.proj(out)
		out = self.proj_drop(out)

		return x + out, current_key_value

