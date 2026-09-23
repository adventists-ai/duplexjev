#!/usr/bin/env bash
# stack_factor=2，A/B 从头训。用法：
#   bash run_sf2_ab.sh smoke
#   bash run_sf2_ab.sh
set -uo pipefail
MODE=${1:-train}
ROOT=/data/exp01/exp03_train
PY=$ROOT/venv/bin/python
SCR=$ROOT/scripts_ml_asr
UV=$ROOT/ultravox
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 100 包：PACK=N bash run_sf2_ab.sh   （默认 0）。训练只读 packs100/pNNN，目录结构与 official 相同。
PACK=${PACK:-0}
PACKDIR=$ROOT/data/ml_asr/packs100/p$(printf %03d "$PACK")
[[ -d "$PACKDIR" ]] || { echo "no $PACKDIR"; exit 1; }
export ML_ASR_OFFICIAL_ROOT=$PACKDIR
unset ML_ASR_PACK
export ML_ASR_N_PACKS=10
# 从头训不做 C1 排除（wenet_lhotse 默认会读 c1 清单，必须显式指向空）
export ML_ASR_C1_EXCLUDE=/dev/null
export ML_ASR_CONT_DIR=$ROOT/artifacts/ml_asr/cont_qwen32b

say() { echo "[$(date +%F' '%T)] $*" >&2; }
die() { say "中止：$*"; exit 1; }

if [[ "$MODE" != smoke ]]; then
  $PY "$SCR/cont_coverage.py" "$PACK" 0.95 || die "包 $PACK 续写覆盖率不足 95%"
fi

$PY "$SCR/patch_stack_factor.py"
cp "$SCR/official_local.py" "$UV/ultravox/data/configs/official_local.py"
cp "$SCR/wenet_lhotse.py" "$UV/ultravox/data/configs/wenet_lhotse.py"
cp "$SCR/continuation_store.py" "$UV/ultravox/data/configs/continuation_store.py"
$PY "$SCR/patch_wenet_loader.py"
$PY "$SCR/patch_pack_filter.py"
$PY "$SCR/patch_qwen_cont.py"
$PY - "$UV/ultravox/data/registry.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
if "official_local" not in s:
    s += "\nfrom ultravox.data.configs import official_local\nofficial_local.apply(DATASET_MAP)\n"
    p.write_text(s)
    print("registry patched")
else:
    print("registry already patched")
PY

cp "$SCR/v1_h18_sf2.yaml" "$UV/ultravox/training/configs/ml_asr_v1_h18_sf2.yaml"
cp "$SCR/v2_xattn_sf2.yaml" "$UV/ultravox/training/configs/ml_asr_v2_xattn_sf2.yaml"
if [[ "$MODE" == smoke ]]; then
  $PY - <<'PY'
from pathlib import Path
d = Path("/data/exp01/exp03_train/ultravox/ultravox/training/configs")
for name in ("ml_asr_v1_h18_sf2.yaml", "ml_asr_v2_xattn_sf2.yaml"):
    p = d / name
    t = p.read_text().replace("max_steps: 32000", "max_steps: 2").replace("save_steps: 2000", "save_steps: 2")
    p.write_text(t)
    print("smoke", name)
PY
  LOCKB=$ROOT/artifacts/ml_asr/sf2_B_SMOKE
  LOCKA=$ROOT/artifacts/ml_asr/sf2_A_SMOKE
  LOGB=$ROOT/logs/ml_asr_sf2_B_smoke.log
  LOGA=$ROOT/logs/ml_asr_sf2_A_smoke.log
else
  LOCKB=$ROOT/artifacts/ml_asr/sf2_B_STARTED
  LOCKA=$ROOT/artifacts/ml_asr/sf2_A_STARTED
  LOGB=$ROOT/logs/ml_asr_sf2_B.log
  LOGA=$ROOT/logs/ml_asr_sf2_A.log
  [[ ! -f "$LOCKB" ]] || die "$LOCKB 已在"
  [[ ! -f "$LOCKA" ]] || die "$LOCKA 已在"
fi

mkdir -p "$ROOT/artifacts/ml_asr" "$ROOT/logs"
printf '%s\n' "{\"line\":\"B\",\"stack_factor\":2,\"from_scratch\":true,\"mode\":\"$MODE\"}" >"$LOCKB"
printf '%s\n' "{\"line\":\"A\",\"stack_factor\":2,\"from_scratch\":true,\"mode\":\"$MODE\"}" >"$LOCKA"

say "B 卡 0-3  sf2  port 29504"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ML_ASR_C1_EXCLUDE="$ML_ASR_C1_EXCLUDE" ML_ASR_CONT_DIR="$ML_ASR_CONT_DIR" \
  bash -c "cd \"$UV\" && \"$PY\" -m torch.distributed.run --nproc_per_node=4 --master_port=29504 -m ultravox.training.train --config_path ultravox/training/configs/ml_asr_v1_h18_sf2.yaml" \
  >"$LOGB" 2>&1 &
echo $! > "$ROOT/artifacts/ml_asr/sf2_B.pid"

say "A 卡 4-7  sf2  port 29505"
nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ML_ASR_C1_EXCLUDE="$ML_ASR_C1_EXCLUDE" ML_ASR_CONT_DIR="$ML_ASR_CONT_DIR" \
  bash -c "cd \"$UV\" && \"$PY\" -m torch.distributed.run --nproc_per_node=4 --master_port=29505 -m ultravox.training.train --config_path ultravox/training/configs/ml_asr_v2_xattn_sf2.yaml" \
  >"$LOGA" 2>&1 &
echo $! > "$ROOT/artifacts/ml_asr/sf2_A.pid"
say "B pid $(cat $ROOT/artifacts/ml_asr/sf2_B.pid)  A pid $(cat $ROOT/artifacts/ml_asr/sf2_A.pid)"
if [[ "$MODE" != smoke ]]; then
  printf '| %s | p%03d | A=ml_asr_v2_xattn_sf2 B=ml_asr_v1_h18_sf2 | 开训 |\n' "$(date '+%F %T')" "$PACK" >> "$ROOT/data/ml_asr/packs100/PACK_LEDGER.md"
fi
