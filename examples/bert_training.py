# examples/bert_training.py
import argparse
import os
import time
from tinygrad import Tensor, nn, dtypes, UOp
from tinygrad.nn import optim
from tinygrad.helpers import getenv, DEBUG, GlobalCounters
from tinygrad.apps.llm import Transformer, TransformerBlock, precompute_freqs_cis, apply_rope # Import necessary components

# --- Configuration ---
VOCAB_SIZE = 1000
MAX_CONTEXT = 128
DIM = 64
HIDDEN_DIM = DIM * 4
N_HEADS = 4
N_KV_HEADS = N_HEADS
NUM_BLOCKS = 2
NORM_EPS = 1e-5
ROPE_THETA = 10000.0
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
TRAIN_STEPS = 10

# --- Trainable Transformer Class ---
class TrainableTransformer(Transformer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # Return raw logits for training
    return self.output(self.output_norm(x)) # (B, T, vocab_size)

# --- Synthetic Dataset Generation ---
def generate_synthetic_batch(batch_size, seq_len, vocab_size):
  # Generate random token IDs
  tokens = Tensor.randint(batch_size, seq_len, low=0, high=vocab_size, dtype=dtypes.int)
  # For MLM-like objective, predict the next token
  # Input sequence will be [token_0, token_1, ..., token_{seq_len-2}]
  # Target will be [token_1, token_2, ..., token_{seq_len-1}]
  x = tokens[:, :-1].realize()
  y = tokens[:, 1:].realize()
  return x, y

# --- Main Training Function ---
def train(args):
  # Set environment variable for Flash Attention
  os.environ["FLASH_ATTENTION"] = "1" if args.flash_attention else "0"
  print(f"Flash Attention {'enabled' if args.flash_attention else 'disabled'}.")

  # Enable training mode for Tensor
  Tensor.training = True

  # Initialize the model
  model = TrainableTransformer(
      num_blocks=NUM_BLOCKS,
      dim=DIM,
      hidden_dim=HIDDEN_DIM,
      n_heads=N_HEADS,
      n_kv_heads=N_KV_HEADS,
      norm_eps=NORM_EPS,
      vocab_size=VOCAB_SIZE,
      head_dim=DIM // N_HEADS,
      rope_theta=ROPE_THETA,
      max_context=MAX_CONTEXT,
      causal=False,
  )

  # Set requires_grad=True for all parameters
  for param in nn.state.get_parameters(model):
    param.requires_grad = True

  if DEBUG >= 1:
    print("Model parameters and their requires_grad status:")
    for i, param in enumerate(nn.state.get_parameters(model)):
      print(f"  Parameter {i}: shape={param.shape}, requires_grad={param.requires_grad}")

  # Optimizer
  optimizer = optim.Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)

  # Training loop
  for step in range(args.steps):
    GlobalCounters.reset()
    start_time = time.perf_counter()

    # Generate batch
    input_tokens, target_tokens = generate_synthetic_batch(BATCH_SIZE, MAX_CONTEXT, VOCAB_SIZE)

    # Forward pass
    # Pass 0 as start_pos since we are training a fresh model with full sequence
    logits = model(input_tokens, 0) # (B, T-1, vocab_size)

    # Calculate loss (Cross-Entropy)
    # Reshape logits and targets for cross_entropy
    loss = logits.reshape(-1, VOCAB_SIZE).sparse_categorical_crossentropy(target_tokens.reshape(-1)).mean()

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Realize all pending ops
    Tensor.realize_batch([loss])

    end_time = time.perf_counter()
    step_duration = end_time - start_time
    tokens_per_second = (BATCH_SIZE * (MAX_CONTEXT - 1)) / step_duration

    if DEBUG >= 1 or step % 1 == 0:
      print(f"Step {step+1}/{args.steps} | Loss: {loss.item():.4f} | Time: {step_duration:.4f}s | Tokens/sec: {tokens_per_second:.2f}")

  print("Training finished.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Minimal BERT-like training script for tinygrad.")
  parser.add_argument("--flash_attention", action="store_true", help="Enable Flash Attention (via env var).")
  parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="Number of training steps.")
  args = parser.parse_args()

  train(args)
