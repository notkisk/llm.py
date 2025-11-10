"""
Mixture of Experts (MoE) model example.
"""

from src import Model, medium_config, Embedding, RotaryPE, SelfAttention, FeedForward, MoEHead

# Create config
cfg = medium_config(vocab_size=50000, max_seq_len=2048)

# Build MoE model, isn't this awesome?
model = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(SelfAttention, 8, dropout=0.1)
        .repeat(FeedForward, 8, dropout=0.1)
        .add(MoEHead(num_experts=8, top_k=2))
)

# Validate #TODO: validation is currently pretty basic, it does not take into account a lot of edge cases, will revist it later(probably means never)
model.validate()

# Print model info
print("MoE Model:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
import torch
x = torch.randint(0, cfg.vocab_size, (2, 64))
out = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {out.shape}")

# Get load balancing loss if training
model.train()
out = model(x)
moe_head = model.components[-1]
if hasattr(moe_head, 'get_load_balancing_loss'):
    lb_loss = moe_head.get_load_balancing_loss()
    if lb_loss is not None:
        print(f"Load balancing loss: {lb_loss.item():.4f}")

