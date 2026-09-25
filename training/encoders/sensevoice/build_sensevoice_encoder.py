"""Build the portable SenseVoice encoder dir from FunAudioLLM/SenseVoiceSmall (model.pt, am.mvn) and funasr source."""
import json, os, re, shutil, sys
import numpy as np, torch
SRC, FUN, OUT, HERE = sys.argv[1:5]  # SenseVoiceSmall dir, funasr sense_voice/model.py, output dir, dir of these files
os.makedirs(OUT, exist_ok=True)
lines = open(FUN).read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("class SinusoidalPositionEncoder"))
end = next(i for i, l in enumerate(lines) if l.startswith("class SenseVoiceSmall"))
body = [l for l in lines[start:end] if not l.startswith("@tables.register")]
# masks follow the activations' dtype (funasr builds them in float32, which breaks bf16 weights)
src_mask = "masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]"
assert sum(src_mask in l for l in body) == 1
body = [l.replace(src_mask, "masks = sequence_mask(ilens, maxlen=maxlen, dtype=xs_pad.dtype, device=ilens.device)[:, None, :]") for l in body]
hdr = ['"""SANM encoder blocks, copied from funasr/models/sense_voice/model.py (MIT License, Alibaba DAMO Speech Lab)."""',
       "from typing import Optional", "import numpy as np", "import torch", "import torch.nn.functional as F",
       "from torch import nn", ""]
open(os.path.join(OUT, "sensevoice_sanm.py"), "w").write("\n".join(hdr + body).rstrip() + "\n")
for f in ("configuration_sensevoice_encoder.py", "modeling_sensevoice_encoder.py", "feature_extraction_sensevoice.py"):
    shutil.copy(os.path.join(HERE, f), os.path.join(OUT, f))
# cmvn
means, vars_ = [], []
L = open(os.path.join(SRC, "am.mvn")).read().splitlines()
for i, l in enumerate(L):
    t = l.split()
    if t and t[0] == "<AddShift>":
        means = L[i + 1].split()[3:-1]
    if t and t[0] == "<Rescale>":
        vars_ = L[i + 1].split()[3:-1]
cmvn = torch.tensor(np.array([means, vars_], dtype=np.float32))
sd = torch.load(os.path.join(SRC, "model.pt"), map_location="cpu")
sd = sd.get("state_dict", sd)
new = {k: v for k, v in sd.items() if k.startswith("encoder.") or k == "embed.weight"}
new["cmvn"] = cmvn
from safetensors.torch import save_file
save_file({k: v.contiguous() for k, v in new.items()}, os.path.join(OUT, "model.safetensors"))
cfg = {"architectures": ["SenseVoiceEncoder"], "model_type": "sensevoice_encoder_portable",
       "auto_map": {"AutoConfig": "configuration_sensevoice_encoder.SenseVoiceEncoderConfig",
                    "AutoModel": "modeling_sensevoice_encoder.SenseVoiceEncoder"},
       "input_size": 560, "output_size": 512, "attention_heads": 4, "linear_units": 2048, "num_blocks": 50, "tp_blocks": 20,
       "kernel_size": 11, "sanm_shfit": 0, "n_mels": 80, "lfr_m": 7, "lfr_n": 6, "frame_length": 25, "frame_shift": 10,
       "num_query_embeddings": int(sd["embed.weight"].shape[0]), "language_id": 0, "textnorm_id": 15, "torch_dtype": "float32"}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
pp = {"feature_extractor_type": "SenseVoiceWaveFrames", "auto_map": {"AutoFeatureExtractor": "feature_extraction_sensevoice.SenseVoiceWaveFrames",
      "AutoProcessor": "feature_extraction_sensevoice.SenseVoiceWaveFrames"},
      "feature_size": 160, "sampling_rate": 16000, "hop_length": 160, "chunk_length": 30, "padding_value": 0.0}
json.dump(pp, open(os.path.join(OUT, "preprocessor_config.json"), "w"), indent=2)
print("kept", len(new), "tensors;", sum(v.numel() for v in new.values()) // 10**6, "M params; embed", tuple(sd["embed.weight"].shape))
