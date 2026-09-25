# SenseVoice-Small encoder as a standalone transformers model

Builds [`adventists-ai/SenseVoice-Small-Encoder`](https://huggingface.co/adventists-ai/SenseVoice-Small-Encoder) from
FunAudioLLM/SenseVoiceSmall (`model.pt`, `am.mvn`) and FunASR's `models/sense_voice/model.py` (for the SANM blocks):

```bash
python build_sensevoice_encoder.py /path/SenseVoiceSmall /path/funasr/models/sense_voice/model.py OUT_DIR .
```

- The feature extractor cuts 16 kHz audio into 10 ms frames of raw samples; the encoder computes SenseVoice's frontend
  (Kaldi fbank 80, LFR 7/6, CMVN), prepends the four task queries and runs the 50 + 20 SANM blocks. The query frames
  are dropped: one output frame = 60 ms. Use `stack_factor: 3` (5.6 audio tokens/s) in the Ultravox config.
- Checked by greedy decoding with SenseVoiceSmall's own CTC head (correct Chinese and English transcripts, fp32 and bf16).
- Needs `torchaudio`. The weights are under the FunASR Model License (attribution, keep the model name); the SANM code
  is from FunASR (MIT); this packaging code is Apache-2.0.
