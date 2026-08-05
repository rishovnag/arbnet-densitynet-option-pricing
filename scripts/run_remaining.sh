#!/usr/bin/env bash
# =============================================================================
# Unified pipeline for the entire remainder of the ArbNet experiments.
#
#   bash scripts/run_remaining.sh           # full run
#   QUICK=1 bash scripts/run_remaining.sh   # smoke only (~minutes), validates wiring
#   STRIDE=20 EPOCHS=150 bash scripts/run_remaining.sh   # fast subsampled full pass
#
# Knobs (env vars):
#   QUICK=1     run only the numpy prototypes + the 2-day smoke gate, then stop.
#   STRIDE=N    subsample every Nth trading day for the real study/ablation (default 1 = all).
#   EPOCHS=N    training epochs (default 150).
#   SEEDS="0"   space-separated seeds for the corrected real study (default "0").
#
# Design:
#  * Writes corrected results to NEW files; never overwrites your existing
#    results/*.json (they are also backed up to results/backup_<ts>/ first).
#  * The H3 fix (causal context normalization) invalidated the old context-using
#    runs, so the corrected real study + matched lambda-sweep are retrained FRESH.
#    The synthetic study has no context, so DensityNet is just MERGED in.
#  * A 2-day smoke gate runs all models first and ABORTS the whole pipeline if any
#    model errors -- so a multi-hour run never starts against broken code.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root

STRIDE="${STRIDE:-1}"
EPOCHS="${EPOCHS:-150}"
SEEDS="${SEEDS:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
ALL_REAL="arbnet ackerer bs arbnet_bump ackerer_matched arbnet_density"

echo "=== ArbNet remaining-experiments pipeline ($TS) ==="
echo "STRIDE=$STRIDE EPOCHS=$EPOCHS SEEDS='$SEEDS'  QUICK=${QUICK:-0}"

# --- 0. backup existing results --------------------------------------------
mkdir -p "results/backup_$TS"
cp -f results/*.json "results/backup_$TS/" 2>/dev/null || true
echo "Backed up existing results to results/backup_$TS/"

# --- 1. numpy prototypes (no torch) ----------------------------------------
echo; echo "### 1. Prototypes (numpy) ###"
python scripts/density_calendar_prototype.py
python scripts/convex_density_prototype.py
python scripts/bump_prototype.py

# --- 2. smoke gate: 2 days, all models, must all succeed -------------------
echo; echo "### 2. Smoke gate (2 days, all models) ###"
python scripts/train_nse.py --models $ALL_REAL --max_days 2 --n_epochs 3 \
    --out results/_smoke.json
python - <<PY
import json,sys
d=json.load(open("results/_smoke.json"))
need=set("$ALL_REAL".split()); have=set(d.get("aggregate",{}))
miss=need-have
print("smoke models present:",sorted(have))
sys.exit("SMOKE FAILED, missing models: %s"%miss if miss else 0)
PY
echo "Smoke gate passed."
rm -f results/_smoke.json

if [ "${QUICK:-0}" = "1" ]; then
  echo; echo "QUICK=1 -> stopping after smoke gate."; exit 0
fi

# --- 3. corrected real study (FRESH; H3 invalidated old context runs) -------
echo; echo "### 3. Corrected real walk-forward (all models, fresh) ###"
for SD in $SEEDS; do
  OUT="results/nse_study_corrected_seed${SD}.json"
  echo "-- seed $SD -> $OUT"
  python scripts/train_nse.py --models $ALL_REAL --stride "$STRIDE" \
      --n_epochs "$EPOCHS" --seed "$SD" --out "$OUT"
done

# --- 4. synthetic: merge DensityNet into a COPY (original study.json untouched) ----
echo; echo "### 4. Synthetic study: add DensityNet (into a copy) ###"
cp -f results/study.json results/study_corrected.json
python scripts/run_study.py --models ArbNet_density --resume \
    --n_surfaces 10 --seeds_per_surface 3 --n_epochs "$EPOCHS" \
    --out results/study_corrected.json

# --- 5. no-context ablation (real) -----------------------------------------
echo; echo "### 5. No-context ablation (real) ###"
python scripts/train_nse.py --models $ALL_REAL --no_context --stride "$STRIDE" \
    --n_epochs "$EPOCHS" --out results/nse_study_corrected_nocontext.json

# --- 6. lambda-sweeps on the corrected pipeline ----------------------------
echo; echo "### 6. Ackerer lambda-sweeps (matched + plain) ###"
python scripts/lambda_sweep.py --stride 20 --n_epochs "$EPOCHS" \
    --out results/lambda_sweep_matched_corrected.json
python scripts/lambda_sweep.py --stride 20 --plain --n_epochs "$EPOCHS" \
    --out results/lambda_sweep_plain_corrected.json

echo; echo "=== DONE. New/updated files in results/ (originals in results/backup_$TS/) ==="
ls -la results/*.json
