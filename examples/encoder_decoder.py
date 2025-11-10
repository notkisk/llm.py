"""
Encoder-Decoder model example using cross-attention.
"""

from src import (
    Model, medium_config, Embedding, LearnedPE,
    SelfAttention, CrossAttention, FeedForward, LMHead
)

cfg = medium_config(vocab_size=50000, max_seq_len=2048)

encoder = (
    Model(cfg)
        .add(Embedding())
        .add(LearnedPE(max_seq_len=cfg.max_seq_len))
        .repeat(SelfAttention, 6, dropout=0.1)
        .repeat(FeedForward, 6, dropout=0.1)
)

decoder = (
    Model(cfg)
        .add(Embedding())
        .add(LearnedPE(max_seq_len=cfg.max_seq_len))
        .repeat(SelfAttention, 6, dropout=0.1)  
        .repeat(CrossAttention, 6, dropout=0.1)  
        .repeat(FeedForward, 6, dropout=0.1)
        .add(LMHead(tie_weights=True))
)

# Validate
encoder.validate()
decoder.validate()

print("Encoder Model:")
print(encoder)
print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

print("\nDecoder Model:")
print(decoder)
print(f"\nDecoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

import torch

enc_input = torch.randint(0, cfg.vocab_size, (2, 128))
enc_output = encoder(enc_input)
print(f"\nEncoder input shape: {enc_input.shape}")
print(f"Encoder output shape: {enc_output.shape}")

dec_input = torch.randint(0, cfg.vocab_size, (2, 64))
# XXX: CrossAttention expects context parameter in forward pass
# This is a simplified example - in practice you'd need to modify
# the Model.forward to handle cross-attention properly
print(f"\nDecoder input shape: {dec_input.shape}")

