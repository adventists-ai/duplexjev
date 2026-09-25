"""Pass-through 'feature extractor': cuts 16 kHz audio into 10 ms frames of raw samples.

The encoder computes SenseVoice's own frontend (Kaldi fbank, LFR, CMVN) from these frames, so the frame count seen
by the Ultravox processor is the usual 10 ms mel-frame count and its length logic stays unchanged."""
import numpy as np
from transformers import BatchFeature, SequenceFeatureExtractor


class SenseVoiceWaveFrames(SequenceFeatureExtractor):
    model_input_names = ["input_features", "attention_mask"]

    def __init__(self, feature_size=160, sampling_rate=16000, padding_value=0.0, hop_length=160, chunk_length=30, **kwargs):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.nb_max_frames = chunk_length * sampling_rate // hop_length

    @property
    def feature_extractor(self):  # the Ultravox processor reads audio_processor.feature_extractor.hop_length
        return self

    def __call__(self, raw_speech, sampling_rate=None, return_attention_mask=True, return_tensors=None, **kwargs):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"expected {self.sampling_rate} Hz audio, got {sampling_rate}")
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        h = self.hop_length
        frames = [int(np.ceil(len(x) / h)) for x in raw_speech]
        n = max(frames)
        feats = np.zeros((len(raw_speech), h, n), dtype=np.float32)
        mask = np.zeros((len(raw_speech), n), dtype=np.int32)
        for i, x in enumerate(raw_speech):
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            y = np.zeros(frames[i] * h, dtype=np.float32)
            y[: len(x)] = x
            feats[i, :, : frames[i]] = y.reshape(frames[i], h).T
            mask[i, : frames[i]] = 1
        out = {"input_features": feats}
        if return_attention_mask:
            out["attention_mask"] = mask
        return BatchFeature(out, tensor_type=return_tensors)
