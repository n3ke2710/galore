#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run_llm_experiment.sh — GPT-2 × 3 optimizers on WikiText-2
#
#  Usage:
#      ./run_llm_experiment.sh              # train all 3 + plot
#      ./run_llm_experiment.sh --plot-only   # regenerate plots only
#
#  Override hyperparams via env vars:
#      EPOCHS=5 LR=1e-4 ./run_llm_experiment.sh
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="results/gpt2"
mkdir -p "$RESULTS_DIR"
BASH_LOG="$RESULTS_DIR/run.log"
exec > >(tee -a "$BASH_LOG") 2>&1

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  GPT-2 × 3 Optimizers  —  LLM Experiment (H200 ready) ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Logs:    $BASH_LOG  |  $RESULTS_DIR/experiment.log"
echo ""

# ── Python / venv ──
if ! command -v python3 &>/dev/null; then echo -e "${RED}python3 not found${NC}"; exit 1; fi
echo -e "${GREEN}[✓] Python:${NC} $(python3 --version)"

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[*] Creating virtualenv ...${NC}"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}[✓] Venv:${NC} $VENV_DIR"

echo -e "${YELLOW}[*] Installing deps ...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}[✓] Deps OK${NC}"

# ── GPU check ──
python3 -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.get_device_name(0)
    g = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'[✓] GPU: {n}  ({g:.0f} GB)')
else:
    print('[!] No CUDA — CPU only')
"

# ── Defaults (override via env) ──
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
LOG_EVERY="${LOG_EVERY:-25}"
RANK="${RANK:-128}"
THRESHOLD="${THRESHOLD:-0.03}"
UPDATE_PROJ_GAP="${UPDATE_PROJ_GAP:-200}"
MIN_RANK="${MIN_RANK:-4}"
SEED="${SEED:-42}"
OPTIMIZERS="${OPTIMIZERS:-adamw,galore,prox}"

EXTRA_ARGS=""
if [[ "${1:-}" == "--plot-only" ]]; then
    EXTRA_ARGS="--plot-only"
    echo -e "${YELLOW}[*] Plot-only mode${NC}"
fi

echo ""
echo "  Config: EPOCHS=$EPOCHS  LR=$LR  BATCH=$BATCH_SIZE  BLOCK=$BLOCK_SIZE"
echo "  Opts:   $OPTIMIZERS  |  rank=$RANK  threshold=$THRESHOLD"
echo ""

python3 llm_experiment.py \
    $EXTRA_ARGS \
    --optimizers "$OPTIMIZERS" \
    --epochs "$EPOCHS" \
    --log-every "$LOG_EVERY" \
    --lr "$LR" \
    --weight-decay 0.01 \
    --seed "$SEED" \
    --rank "$RANK" \
    --threshold "$THRESHOLD" \
    --update-proj-gap "$UPDATE_PROJ_GAP" \
    --min-rank "$MIN_RANK" \
    --batch-size "$BATCH_SIZE" \
    --block-size "$BLOCK_SIZE"

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo "  Data:   results/gpt2/{adamw,galore,prox}/*.json"
echo "  Plots:  results/gpt2/plots/*.png"
echo "  Replot: ./run_llm_experiment.sh --plot-only"
