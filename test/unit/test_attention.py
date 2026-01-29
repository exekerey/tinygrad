import os, time, unittest
from tinygrad import Tensor, dtypes, TinyJit, UOp, nn
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tinygrad.apps.llm import apply_rope as apply_rope_new, precompute_freqs_cis
#from tinygrad.engine.realize import run_schedule

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

# TODO: test_scheduler, but just in uint
class TestAttention(unittest.TestCase):
  def test_half_qkv_buffers(self):
    BS, seqlen, dim = 10, 4, 100
    q = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    k = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    v = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    attn = q.scaled_dot_product_attention(k, v)
    sched = attn.schedule()
    # attention has 4 kernels now
    self.assertEqual(len(sched), 4)
    # softmax_inputs = sched[1:4]
    # for i,si in enumerate(softmax_inputs):
    #   assert all(b.dtype == dtypes.half for b in si.bufs), f"non half {si.bufs=} in kernel {i}"

  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_apply_rope_jit_prune(self):
    def rope_fn(x_in, pos): return apply_rope(x_in, pos)
    rope_noprune = TinyJit(rope_fn)
    rope_prune = TinyJit(rope_fn, prune=True)

    v_pos = UOp.variable("start_pos", 0, 100)
    for _ in range(3):
      rope_noprune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
      rope_prune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
    noprune_size = len(rope_noprune.captured.jit_cache)
    prune_size = len(rope_prune.captured.jit_cache)

    self.assertGreater(noprune_size, prune_size)
    self.assertGreaterEqual(noprune_size, 2)
    self.assertEqual(prune_size, 1)

  def test_attention_causal_mask_equivalence(self):
    B, H, T, D = 2, 3, 5, 8
    q = Tensor.randn(B, H, T, D, dtype=dtypes.float32)
    k = Tensor.randn(B, H, T, D, dtype=dtypes.float32)
    v = Tensor.randn(B, H, T, D, dtype=dtypes.float32)
    mask = Tensor.full((1, 1, T, T), float("-inf"), dtype=dtypes.float32).triu(1)
    out_mask = q.scaled_dot_product_attention(k, v, attn_mask=mask)
    out_causal = q.scaled_dot_product_attention(k, v, is_causal=True)
    self.assertLess((out_mask - out_causal).abs().max().item(), 1e-4)

  @unittest.skipUnless(os.getenv("SPEED_TEST"), "set SPEED_TEST=1 to run speed comparison")
  def test_attention_speed_flash_vs_normal(self):
    B = int(os.getenv("SPEED_B", 4))
    H = int(os.getenv("SPEED_H", 8))
    T = int(os.getenv("SPEED_T", 128))
    D = int(os.getenv("SPEED_D", 64))
    warmup = int(os.getenv("SPEED_WARMUP", 2))
    iters = int(os.getenv("SPEED_ITERS", 5))

    q = Tensor.randn(B, H, T, D, dtype=dtypes.float16).contiguous().realize()
    k = Tensor.randn(B, H, T, D, dtype=dtypes.float16).contiguous().realize()
    v = Tensor.randn(B, H, T, D, dtype=dtypes.float16).contiguous().realize()

    def bench(flash:bool) -> float:
      os.environ["FLASH_ATTENTION"] = "1" if flash else "0"
      getenv.cache_clear()
      for _ in range(warmup):
        q.scaled_dot_product_attention(k, v, is_causal=True).realize()
      st = time.perf_counter()
      for _ in range(iters):
        q.scaled_dot_product_attention(k, v, is_causal=True).realize()
      return time.perf_counter() - st

    try:
      t_flash = bench(True)
    except Exception as e:
      raise unittest.SkipTest(f"flash attention unavailable: {e}")
    t_normal = bench(False)

    print(f"flash {t_flash:.6f}s vs normal {t_normal:.6f}s (iters={iters})")
    if os.getenv("REQUIRE_FLASH_FASTER"):
      self.assertLess(t_flash, t_normal)

  @unittest.skipUnless(os.getenv("BERT_SPEED_TEST"), "set BERT_SPEED_TEST=1 to run BERT training speed comparison")
  def test_bert_training_speed_flash_vs_normal(self):
    try:
      import extra.thunder.tiny.fa
    except Exception as e:
      raise unittest.SkipTest(f"flash attention unavailable: {e}")

    from examples.bert_training import TrainableTransformer, VOCAB_SIZE, MAX_CONTEXT, DIM, HIDDEN_DIM, N_HEADS, N_KV_HEADS, NUM_BLOCKS, NORM_EPS, ROPE_THETA

    bs = int(os.getenv("BERT_SPEED_BS", 4))
    context = int(os.getenv("BERT_SPEED_CONTEXT", MAX_CONTEXT))
    warmup = int(os.getenv("BERT_SPEED_WARMUP", 1))
    iters = int(os.getenv("BERT_SPEED_ITERS", 3))

    Tensor.manual_seed(0)
    tokens = Tensor.randint(bs, context, low=0, high=VOCAB_SIZE, dtype=dtypes.int)
    input_tokens = tokens[:, :-1].realize()
    target_tokens = tokens[:, 1:].realize()

    def build_model():
      Tensor.manual_seed(0)
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
          max_context=context,
          causal=False,
      )
      for param in nn.state.get_parameters(model):
        param.requires_grad = True
      return model

    def step(model, opt):
      logits = model(input_tokens, 0)
      loss = logits.reshape(-1, VOCAB_SIZE).sparse_categorical_crossentropy(target_tokens.reshape(-1)).mean()
      opt.zero_grad()
      loss.backward()
      opt.step()
      Tensor.realize(loss)

    def bench(flash:bool) -> float:
      os.environ["FLASH_ATTENTION"] = "1" if flash else "0"
      getenv.cache_clear()
      Tensor.training = True
      model = build_model()
      opt = optim.Adam(nn.state.get_parameters(model), lr=1e-3)
      for _ in range(warmup):
        step(model, opt)
      st = time.perf_counter()
      for _ in range(iters):
        step(model, opt)
      return time.perf_counter() - st

    old_training = Tensor.training
    try:
      t_flash = bench(True)
      t_normal = bench(False)
    finally:
      Tensor.training = old_training

    print(f"bert flash {t_flash:.6f}s vs normal {t_normal:.6f}s (iters={iters})")
    if os.getenv("REQUIRE_FLASH_FASTER"):
      self.assertLess(t_flash, t_normal)

if __name__ == '__main__':
  unittest.main()
