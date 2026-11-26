"""HuggingFace configuration for LoomFormer (Llama-backbone RMSNorm/SwiGLU, Medium/Large/XL) and MHA baselines."""
from transformers import PretrainedConfig


class LoomFormerConfig(PretrainedConfig):
  """Configuration class for `LoomFormerModel` and `LoomFormerForCausalLM`.

  Supports both LoomFormer (`loom_enabled=True`) and matched standard Transformer MHA baselines
  (`loom_enabled=False`, with optional `mha_nope_dim=1` for odd `head_dim=75` at Medium scale).
  """

  model_type = "loomformer"

  def __init__(
      self,
      vocab_size: int = 50432,
      hidden_size: int = 1536,
      num_hidden_layers: int = 22,
      num_attention_heads: int = 16,
      head_dim: int = 96,
      intermediate_size: int = 3317,
      last_ffn_x_dim: int = 5083,
      rms_norm_eps: float = 1e-6,
      rope_min_timescale: float = 1.0,
      rope_max_timescale: float = 10000.0,
      mha_nope_dim: int = 0,
      loom_enabled: bool = True,
      d_k: int = 32,
      d_r: int = 256,
      rope_dim: int = 24,
      ffn_m_dim: int = 1539,
      read_key_eps: float = 1e-4,
      read_key_scale: float = 0.1,
      write_eps: float = 1e-6,
      read_gate_init: float = 0.05,
      static_read_gate_init: float = 0.99,
      write_gate_init: float = 0.1,
      m_shift_init: float = 0.5,
      clone_vo_init: bool = True,
      embed_write_divisor: float = 0.0018161707314488303,
      ffn_write_divisor: float = 1.0,
      situ_beta1: float = 0.0,
      situ_beta2: float = 0.0,
      max_position_embeddings: int = 4096,
      initializer_range: float = 0.006,
      tie_word_embeddings: bool = False,
      use_cache: bool = True,
      bos_token_id: int = 0,
      eos_token_id: int = 0,
      **kwargs,
  ):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.head_dim = head_dim
    self.intermediate_size = intermediate_size
    self.last_ffn_x_dim = last_ffn_x_dim
    self.rms_norm_eps = rms_norm_eps
    self.rope_min_timescale = rope_min_timescale
    self.rope_max_timescale = rope_max_timescale
    self.mha_nope_dim = mha_nope_dim
    self.loom_enabled = loom_enabled
    self.d_k = d_k
    self.d_r = d_r
    self.rope_dim = rope_dim
    self.ffn_m_dim = ffn_m_dim
    self.read_key_eps = read_key_eps
    self.read_key_scale = read_key_scale
    self.write_eps = write_eps
    self.read_gate_init = read_gate_init
    self.static_read_gate_init = static_read_gate_init
    self.write_gate_init = write_gate_init
    self.m_shift_init = m_shift_init
    self.clone_vo_init = clone_vo_init
    self.embed_write_divisor = embed_write_divisor
    self.ffn_write_divisor = ffn_write_divisor
    self.situ_beta1 = situ_beta1
    self.situ_beta2 = situ_beta2
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.use_cache = use_cache
    assert num_attention_heads * head_dim == hidden_size, "hidden_size must equal num_attention_heads * head_dim"
    if loom_enabled:
      assert rope_dim % 2 == 0, "rope_dim must be even"
    else:
      assert (head_dim - mha_nope_dim) % 2 == 0, "head_dim - mha_nope_dim must be even"
    super().__init__(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )

  @property
  def qk_dim(self) -> int:
    return self.head_dim + self.rope_dim if self.loom_enabled else self.head_dim

  def ffn_x_dim_of(self, layer_idx: int) -> int:
    if self.loom_enabled and layer_idx == self.num_hidden_layers - 1:
      return self.last_ffn_x_dim
    return self.intermediate_size

  def has_write(self, layer_idx: int) -> bool:
    return self.loom_enabled and layer_idx < self.num_hidden_layers - 1
