#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run_llm_experiment.sh — One-shot launcher for GPT-2 + Proximal GaLore
#
#  Usage (on the server):
#      chmod +x run_llm_experiment.sh
#      ./run_llm_experiment.sh
#
#  To re-plot only (no training):
#      ./run_llm_experiment.sh --plot-only
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Results & log paths ──
RESULTS_DIR="results/gpt2"
mkdir -p "$RESULTS_DIR"
BASH_LOG="$RESULTS_DIR/run.log"

# ── Tee all output (stdout + stderr) to both console AND log file ──
exec > >(tee -a "$BASH_LOG") 2>&1

# ── Colors for pretty output ──
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  GPT-2 + Proximal GaLore  —  LLM Experiment Runner ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Bash log:   $BASH_LOG"
echo "  Python log: $RESULTS_DIR/experiment.log"
echo ""

# ── 1. Check Python ──
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] Python:${NC} $(python3 --version)"

# ── 4. Check GPU ──
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'[✓] GPU: {name}  ({mem:.0f} GB)')
else:
    print('[!] No CUDA GPU detected — will run on CPU (slow)')
"

# ── 5. Run experiment ──
echo ""
echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}  Starting experiment ...${NC}"
echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"
echo ""

# Default hyperparameters (override via env vars)
MAX_STEPS="${MAX_STEPS:-1500}"
LOG_EVERY="${LOG_EVERY:-50}"
LR="${LR:-5e-5}"
THRESHOLD="${THRESHOLD:-0.03}"
UPDATE_PROJ_GAP="${UPDATE_PROJ_GAP:-100}"
MIN_RANK="${MIN_RANK:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"

# Pass --plot-only if given, plus all training params
EXTRA_ARGS=""
if [[ "${1:-}" == "--plot-only" ]]; then
    EXTRA_ARGS="--plot-only"
    echo -e "${YELLOW}[*] Plot-only mode — skipping training${NC}"
fi

echo "  Hyperparameters:"
echo "    MAX_STEPS=$MAX_STEPS  LOG_EVERY=$LOG_EVERY  LR=$LR"
echo "    THRESHOLD=$THRESHOLD  UPDATE_PROJ_GAP=$UPDATE_PROJ_GAP  MIN_RANK=$MIN_RANK"
echo "    BATCH_SIZE=$BATCH_SIZE  BLOCK_SIZE=$BLOCK_SIZE"
echo ""

python3 llm_experiment.py \
    $EXTRA_ARGS \
    --max-steps "$MAX_STEPS" \
    --log-every "$LOG_EVERY" \
    --lr "$LR" \
    --threshold "$THRESHOLD" \
    --update-proj-gap "$UPDATE_PROJ_GAP" \
    --min-rank "$MIN_RANK" \
    --batch-size "$BATCH_SIZE" \
    --block-size "$BLOCK_SIZE"

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done! Finished at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Raw data:    results/gpt2/*.json, *.csv"
echo "  Plots:       results/gpt2/plots/*.png"
echo "  Python log:  results/gpt2/experiment.log"
echo "  Bash log:    $BASH_LOG"
echo ""
echo "  To re-plot:  ./run_llm_experiment.sh --plot-only"
