"""
Sliding Window Attention utilities.

Provides helper functions for:
- Creating sliding window attention masks
- Managing rolling buffer KV cache
"""

import torch


def create_sliding_window_mask(T, T_k, window_size, device):
    """
    Creates a sliding window attention mask combined with causal masking.
    
    Args:
        T: Query sequence length
        T_k: Key sequence length
        window_size: Window size (how many previous tokens to attend to)
        device: torch device
        
    Returns:
        Boolean mask of shape (T, T_k) where True = masked (not attended to)
    """
    # Start with causal mask: positions where j > i (upper triangle)
    # diagonal parameter: 1 + T_k - T handles the case where T_k != T (e.g., with KV cache)
    causal_mask = torch.triu(torch.ones(T, T_k, device=device, dtype=torch.bool), diagonal=1 + T_k - T)
    
    if window_size is None:
        return causal_mask
    
    # Create window mask: positions further back than window_size
    # For each query position i, can only attend to keys in range [max(0, i - window_size + 1), i]
    positions = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    key_positions = torch.arange(T_k, device=device).unsqueeze(0)  # (1, T_k)
    
    # Adjust for KV cache: when T_k > T, the new queries are at the end
    offset = T_k - T
    distance = positions - (key_positions - offset)
    
    # Mask positions that are too far back (distance >= window_size)
    window_mask = distance >= window_size
    
    # Combine: mask if either causal violation OR outside window
    return causal_mask | window_mask


def apply_rolling_buffer(k, v, past_key_value, window_size):
    """
    Manages KV cache with rolling buffer to limit memory usage.
    
    When window_size is set, only keeps the most recent window_size entries in cache.
    Older entries are discarded.
    
    Args:
        k: New keys (B, H, T_new, D)
        v: New values (B, H, T_new, D)
        past_key_value: Tuple of (past_k, past_v) or None
        window_size: Maximum cache size (None = no limit)
        
    Returns:
        Tuple of (k_full, v_full, new_cache_tuple)
        where k_full and v_full include both past and new entries
    """
    if past_key_value is not None:
        past_k, past_v = past_key_value
        # Concatenate new with old
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    
    # Apply rolling buffer: keep only last window_size entries
    if window_size is not None and k.size(2) > window_size:
        k = k[:, :, -window_size:, :]
        v = v[:, :, -window_size:, :]
    
    current_key_value = (k, v)
    return k, v, current_key_value
