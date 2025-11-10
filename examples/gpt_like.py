"""
GPT-like model example using rotary positional encoding.
"""

from src import Model, small_config, RotaryPE, SelfAttention, FeedForward, LMHead, Embedding

cfg = small_config(vocab_size=10000, max_seq_len=1024)

model = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(SelfAttention, 6, dropout=0.1)
        .repeat(FeedForward, 6, dropout=0.1)
        .add(LMHead(tie_weights=True))
)

model.validate()

print("GPT-like Model:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

import torch
x = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {out.shape}")

