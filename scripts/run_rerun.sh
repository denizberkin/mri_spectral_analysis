#!/usr/bin/env bash
# Paper rerun: MIM pretraining vs random init, one or more seeds, ending in leaderboard-ready ZIPs.
#
#   scripts/run_rerun.sh <mim|scratch|both> <seed> [<seed> ...]
#
#   scripts/run_rerun.sh both 1 2 3     # everything, seed by seed (mim then scratch for each seed)
#   scripts/run_rerun.sh mim 2          # one run
#
# No per-run config files: the two committed final-model configs are the templates and each run only
# overrides name, seed (and, for `scratch`, removes the masked-image stage) with --set:
#   arm mim      = the final recipe, unchanged
#   arm scratch  = the same with pretrain_steps = 0, so MIM is the only difference
#
# Steps per run, each skipped if its output exists (a crashed job restarts cheaply):
#   1. stage 1   configs/task3_retro_pretrain_big.toml   [30k-step MIM +] 25 supervised epochs
#   2. widen     epoch-25 checkpoint -> 4-channel input
#   3. stage 2   configs/task3_mc_ssim_slice.toml        8 epochs from the widened checkpoint
#   4. soup      uniform mean of stage-2 epochs 6, 7, 8 (fixed in advance, NOT picked by score)
#   5. submission 4-flip TTA -> $SUBMISSION_DIR/rerun_<arm>_s<seed>_avg_tta/task3.zip
#
# Needs: .env set, `python scripts/preprocess.py extract-slices --splits pro_train retro_train` done once,
# a clean tree (every run is traceable to one commit; ALLOW_DIRTY=1 overrides). A run that died inside a
# stage restarts that stage from its beginning. Interpreter: PYTHON=/path/to/python (default: python).
set -euo pipefail

usage() { echo "usage: run_rerun.sh <mim|scratch|both> <seed> [<seed> ...]" >&2; exit 2; }
[ $# -ge 2 ] || usage
ARM_ARG="$1"; shift
case "$ARM_ARG" in mim) ARMS=(mim) ;; scratch) ARMS=(scratch) ;; both) ARMS=(mim scratch) ;; *) usage ;; esac
for seed in "$@"; do case "$seed" in ''|*[!0-9]*) echo "seed must be an integer, got '$seed'" >&2; exit 2 ;; esac; done
SEEDS=("$@")

PY="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/experiment-pipeline"

if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
  echo "tracked files have uncommitted changes; commit first so the run is traceable (or ALLOW_DIRTY=1)" >&2
  exit 1
fi

run_one() {
  local arm="$1" seed="$2"
  local s1="rerun_${arm}_s${seed}_stage1" s2="rerun_${arm}_s${seed}_stage2" name="rerun_${arm}_s${seed}_avg_tta"
  local a1="runs/$s1/artifacts" a2="runs/$s2/artifacts"

  local sets1=(--set "experiment.name=\"$s1\"" --set "tracking.params.run_name=\"$s1\""
               --set "experiment.seed=$seed" --set "data.params.seed=$seed")
  local sets2=(--set "experiment.name=\"$s2\"" --set "tracking.params.run_name=\"$s2\""
               --set "experiment.seed=$seed" --set "data.params.seed=$seed"
               --set "model.params.checkpoint=\"$a1/widened4_from_e25.pt\"")
  if [ "$arm" = scratch ]; then
    sets1+=(--set "training.params.pretrain_steps=0" --set "data.params.include_retro=false")
  fi

  echo "=================== $arm seed $seed"
  mkdir -p "runs/$s1"
  {
    echo "date:        $(date -u +%FT%TZ)"
    echo "host:        $(hostname)"
    echo "arm/seed:    $arm / $seed"
    echo "git commit:  $(git -C "$ROOT" rev-parse HEAD)"
    echo "git branch:  $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
    echo "python:      $($PY -V 2>&1)"
    echo "gpu:         $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "stage 1:     configs/task3_retro_pretrain_big.toml ${sets1[*]}"
    echo "stage 2:     configs/task3_mc_ssim_slice.toml ${sets2[*]}"
    echo "---- pip freeze"
    $PY -m pip freeze 2>/dev/null || true
  } > "runs/$s1/provenance.txt"
  echo "provenance: runs/$s1/provenance.txt"

  if [ ! -f "$a1/task3_unet_finetune_25.pt" ]; then
    echo "[1/5] stage 1: $s1"
    $PY -m eval_pipeline run configs/task3_retro_pretrain_big.toml "${sets1[@]}" --stages training
  else echo "[1/5] stage 1 already done"; fi

  if [ ! -f "$a1/widened4_from_e25.pt" ]; then
    echo "[2/5] widen to 4 channels"
    $PY ../scripts/widen_input_channels.py --channels 4 \
        --checkpoint "$a1/task3_unet_finetune_25.pt" --output "$a1/widened4_from_e25.pt"
  else echo "[2/5] widen already done"; fi

  if [ ! -f "$a2/task3_unet_finetune_8.pt" ]; then
    echo "[3/5] stage 2: $s2"
    $PY -m eval_pipeline run configs/task3_mc_ssim_slice.toml "${sets2[@]}" --stages training
  else echo "[3/5] stage 2 already done"; fi

  if [ ! -f "$a2/avg_e6_e8.pt" ]; then
    echo "[4/5] soup of epochs 6-8"
    $PY ../scripts/make_soup.py \
        --member "$a2/task3_unet_finetune_6.pt:1" \
        --member "$a2/task3_unet_finetune_7.pt:1" \
        --member "$a2/task3_unet_finetune_8.pt:1" \
        --out "$a2/avg_e6_e8.pt"
  else echo "[4/5] soup already done"; fi

  echo "[5/5] submission with 4-flip TTA: $name"
  $PY ../scripts/make_task3_submission.py --tta --checkpoint "$ROOT/experiment-pipeline/$a2/avg_e6_e8.pt" --name "$name"
  echo "DONE $arm seed $seed: upload \$SUBMISSION_DIR/$name/task3.zip, then fill rerun/leaderboard_scores.csv"
}

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    run_one "$arm" "$seed"
  done
done
