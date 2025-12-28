from llm_py import (
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

from llm_py.configs import Config
import torch

# Run a single forward pass
print("\n" + "=" * 50)
print("Forward Pass Check:")
print("=" * 50)
x = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")  

# Run Generation
print("\n" + "=" * 50)
print("Generation Demo (Random Weights):")
print("=" * 50)
start_ids = torch.tensor([[1]], dtype=torch.long)
print(f"Starting ID: {start_ids.tolist()}")

generated = model.generate(start_ids, max_new_tokens=10, temperature=0.0)
print(f"Generated Sequence: {generated.tolist()}")
print("(Note: Output is random integers because model is initialized with random weights)")



model_sliding = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(SelfAttention, 4, dropout=0.1, window_size=128)
        .add(FeedForward())
        .add(LMHead(tie_weights=True))
)

model_sliding.validate()
print(f"  Parameters: {model_sliding.num_parameters:,}")

# Test forward pass
x_test = torch.randint(0, cfg.vocab_size, (2, 32))
out_test = model_sliding(x_test)
print(f"✓ Forward pass: input {x_test.shape} → output {out_test.shape}")

# Test generation with KV cache and rolling buffer
generated_sliding = model_sliding.generate(start_ids, max_new_tokens=20, temperature=0.0)
print(f"✓ Generation with rolling buffer: {len(generated_sliding[0])} tokens")

model_mixed = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(SelfAttention, 2, window_size=256)  
        .repeat(SelfAttention, 2)                    
        .add(FeedForward())
        .add(LMHead(tie_weights=True))
)

model_mixed.validate()

x_test = torch.randint(0, cfg.vocab_size, (2, 32))
out_test = model_mixed(x_test)
print(f"✓ Forward pass: input {x_test.shape} → output {out_test.shape}")

from llm_py import MultiQueryAttention, GroupedQueryAttention

model_mqa = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(MultiQueryAttention, 4, window_size=512)
        .add(FeedForward())
        .add(LMHead(tie_weights=True))
)

model_mqa.validate()
print(f"MQA with sliding window (512 tokens): {model_mqa.num_parameters:,} params")

model_gqa = (
    Model(cfg)
        .add(Embedding())
        .add(RotaryPE())
        .repeat(GroupedQueryAttention, 4, num_kv_heads=4, window_size=256)
        .add(FeedForward())
        .add(LMHead(tie_weights=True))
)

model_gqa.validate()
print(f"GQA with sliding window (256 tokens): {model_gqa.num_parameters:,} params")

x_test = torch.randint(0, cfg.vocab_size, (2, 32))
out_mqa = model_mqa(x_test)
out_gqa = model_gqa(x_test)
print(f"MQA forward: {x_test.shape} → {out_mqa.shape}")
print(f"GQA forward: {x_test.shape} → {out_gqa.shape}")