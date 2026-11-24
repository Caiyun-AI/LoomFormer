"""HuggingFace configuration for LoomPythia (Pythia-backbone LoomFormer, e.g. LoomPythia-2.8B)."""
from transformers import PretrainedConfig


class LoomPythiaConfig(PretrainedConfig):
  """Configuration class for `LoomPythiaModel` and `LoomPythiaForCausalLM`.

  Defaults match `LoomPythia-2.8B` pretrained on 300B Pile tokens (Paper Section 5.2 & Appendix D).
  Notation follows Paper Section 3.2:
    - `head_dim` ($d = 80$): per-head content / matrix-memory slot width ($D = n \cdot d = 2560$).
    - `d_k` ($d_k = 64$): number of matrix-memory slots ($M_t \in \mathbb{R}^{d_k \times d}$).
    - `d_r` ($D_r = 640$): shared low-rank port bottleneck width.
    - `rope_dim` ($r = 20$): vector-stream RoPE coordinates per head (full Q/K head width $d + r = 100$).
    - `intermediate_size` ($7808$): vector-stream FFN (`ffn_x`) width in layers $0 \dots L-2$.
    - `last_ffn_x_dim` ($9984$): terminal layer ($L-1$) `ffn_x` width (no matrix write in layer $L-1$).
    - `ffn_m_dim` ($2944$): matrix-stream FFN (`ffn_m`) width in layers $0 \dots L-2$.
  """

  model_type = "loompythia"

  def __init__(
      self,
      vocab_size: int = 50304,
      hidden_size: int = 2560,
      num_hidden_layers: int = 32,
      num_attention_heads: int = 32,
      head_dim: int = 80,
      d_k: int = 64,
      d_r: int = 640,
      rope_dim: int = 20,
      intermediate_size: int = 7808,
      last_ffn_x_dim: int = 9984,
      ffn_m_dim: int = 2944,
      layer_norm_eps: float = 1e-5,
      read_key_eps: float = 1e-4,
      read_key_scale: float = 0.1,
      write_eps: float = 1e-6,
      static_read_gate_init: float = 0.99,
      embed_write_divisor: float = 0.0020825863,
      ffn_write_divisor: float = 1.0,
      rope_min_timescale: float = 1.0,
      rope_max_timescale: float = 10000.0,
      max_position_embeddings: int = 2048,
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
    self.d_k = d_k
    self.d_r = d_r
    self.rope_dim = rope_dim
    self.intermediate_size = intermediate_size
    self.last_ffn_x_dim = last_ffn_x_dim
    self.ffn_m_dim = ffn_m_dim
    self.layer_norm_eps = layer_norm_eps
    self.read_key_eps = read_key_eps
    self.read_key_scale = read_key_scale
    self.write_eps = write_eps
    self.static_read_gate_init = static_read_gate_init
    self.embed_write_divisor = embed_write_divisor
    self.ffn_write_divisor = ffn_write_divisor
    self.rope_min_timescale = rope_min_timescale
    self.rope_max_timescale = rope_max_timescale
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.use_cache = use_cache
    assert num_attention_heads * head_dim == hidden_size, "hidden_size must equal num_attention_heads * head_dim"
    assert rope_dim % 2 == 0, "rope_dim must be even"
    super().__init__(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )

  @property
  def qk_dim(self) -> int:
    return self.head_dim + self.rope_dim

  def ffn_x_dim_of(self, layer_idx: int) -> int:
    return self.last_ffn_x_dim if layer_idx == self.num_hidden_layers - 1 else self.intermediate_size

  def has_write(self, layer_idx: int) -> bool:
    return layer_idx < self.num_hidden_layers - 1
