from transformers import PretrainedConfig


class Qwen3ASREncoderConfig(PretrainedConfig):
    model_type = "qwen3_asr_encoder_portable"

    def __init__(
        self,
        d_model=896,
        downsample_hidden_size=480,
        encoder_attention_heads=14,
        encoder_ffn_dim=3584,
        encoder_layers=18,
        num_mel_bins=128,
        n_window=50,
        n_window_infer=800,
        max_position_embeddings=13,
        activation_function="gelu",
        activation_dropout=0.0,
        attention_dropout=0.0,
        dropout=0.0,
        scale_embedding=False,
        initializer_range=0.02,
        # Cross-attention fusion (v2): Q from the final layer (H18: highest-level
        # continuous semantic view, decides what/is being said), K from a
        # high-but-not-final layer (H14: still semantic, used as a retrieval
        # index), V from a mid layer (H9: retains more raw acoustic/paralinguistic
        # detail -- pitch, timbre, prosody -- not yet abstracted away). Indices
        # are 0-indexed positions in self.layers, i.e. fusion_k_layer_idx=13 means
        # "snapshot after 14 layers have run" (H14), fusion_v_layer_idx=8 means
        # "after 9 layers" (H9). Q always uses the final layer's output.
        fusion_k_layer_idx=13,
        fusion_v_layer_idx=8,
        fusion_num_heads=8,
        **kwargs,
    ):
        self.d_model = d_model
        self.downsample_hidden_size = downsample_hidden_size
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.num_mel_bins = num_mel_bins
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.scale_embedding = scale_embedding
        self.initializer_range = initializer_range
        self.fusion_k_layer_idx = fusion_k_layer_idx
        self.fusion_v_layer_idx = fusion_v_layer_idx
        self.fusion_num_heads = fusion_num_heads
        super().__init__(**kwargs)

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def encoder_ds_factor(self):
        # Three stride-2 convs in the frontend -> 2*2*2 = 8x downsampling from
        # mel frames to encoder output frames (vs. Whisper's single 2x conv).
        return 8
