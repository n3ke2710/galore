#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run_sweep_experiment.sh — Hyperparameter Sweep (H200 ready)
#
#  Usage:
#      ./run_sweep_experiment.sh              # train sweep + plot
#      ./run_sweep_experiment.sh --plot-only   # regenerate plots only
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SWEEP_DIR="results/sweep"
mkdir -p "$SWEEP_DIR"
BASH_LOG="$SWEEP_DIR/run_sweep.log"
exec > >(tee -a "$BASH_LOG") 2>&1

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Hyperparameter Sweep: Proximal GaLore vs GaLore vs AdamW ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"

# Check python / venv
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}[!] Virtualenv not found. Run ./run_llm_experiment.sh first to set up.${NC}"
    exit 1
fi
source "$VENV_DIR/bin/activate"

EXTRA_ARGS=""
if [[ "${1:-}" == "--plot-only" ]]; then
    EXTRA_ARGS="--plot-only"
    echo -e "${YELLOW}[*] Plot-only mode${NC}"
fi

python3 sweep.py $EXTRA_ARGS

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Sweep Complete! ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo "  Summary table: $SWEEP_DIR/sweep_summary.md"
echo "  Pareto Plot:   $SWEEP_DIR/plots/pareto_front.png"
echo "  Loss Curves:   $SWEEP_DIR/plots/sweep_loss_curves.png"
echo ""
