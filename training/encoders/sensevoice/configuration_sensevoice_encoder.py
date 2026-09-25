"""Config for the SenseVoice-Small audio encoder as a standalone transformers model (DuplexJev)."""
from transformers import PretrainedConfig


class SenseVoiceEncoderConfig(PretrainedConfig):
    model_type = "sensevoice_encoder_portable"

    def __init__(self, input_size=560, output_size=512, attention_heads=4, linear_units=2048, num_blocks=50, tp_blocks=20,
                 kernel_size=11, sanm_shfit=0, n_mels=80, lfr_m=7, lfr_n=6, frame_length=25, frame_shift=10,
                 num_query_embeddings=16, language_id=0, textnorm_id=15, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.tp_blocks = tp_blocks
        self.kernel_size = kernel_size
        self.sanm_shfit = sanm_shfit
        self.n_mels = n_mels
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.num_query_embeddings = num_query_embeddings
        self.language_id = language_id      # "auto"
        self.textnorm_id = textnorm_id      # "woitn"
        super().__init__(**kwargs)

    @property
    def hidden_size(self):
        return self.output_size

    @property
    def d_model(self):
        return self.output_size

    @property
    def encoder_ds_factor(self):
        # one LFR frame = lfr_n 10 ms frames (60 ms): the processor counts 10 ms frames
        return self.lfr_n
