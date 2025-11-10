from src import (
    Model, small_config,
    Embedding, RotaryPE, SelfAttention, FeedForward, LMHead
)

cfg = small_config(vocab_size=10000)

model = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(SelfAttention, 4, dropout=0.1)
        .add(FeedForward())
        .add(LMHead(tie_weights=True))
)

model.validate()

print("=" * 50)
print("Model Properties:")
print("=" * 50)
print(f"Name: {model.name}")
print(f"Vocabulary size: {model.vocab_size:,}")
print(f"Model dimension: {model.dim}")
print(f"Number of heads: {model.num_heads}")
print(f"Max sequence length: {model.max_seq_len}")
print(f"Hidden dimension: {model.hidden_dim}")
print(f"Number of components: {model.num_components}")
print(f"Component names: {model.component_names}")
print(f"Total parameters: {model.num_parameters:,}")
print(f"Trainable parameters: {model.num_trainable_parameters:,}")
print(f"Model size: {model.model_size_mb:.2f} MB")

print("\n" + "=" * 50)
print("Model Summary:")
print("=" * 50)
model.summary()

print("\n" + "=" * 50)
print("Model Representation:")
print("=" * 50)
print(model)

from src.configs import Config
import torch
x = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {out.shape}")

