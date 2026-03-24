#!/bin/bash
# launch.sh — Launch thunder-train across Mac4 + Mac5 via mlx.launch
#
# Uses the ring backend over the Thunderbolt interface (10.0.5.x).
# All arguments after the script name are passed through to thunder_train.py.
#
# Usage:
#   ./launch.sh --model mlx-community/Qwen2.5-7B-Instruct-4bit \
#     --train-data ~/projects/karl/autocontinue-data/train_merged.jsonl \
#     --valid-data ~/projects/karl/autocontinue-data/eval_merged.jsonl \
#     --strategy data --num-iters 800 --batch-size 4 \
#     --adapter-path ~/projects/karl/thunder-adapter-v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOSTFILE="$SCRIPT_DIR/hostfile.json"
TRAIN_SCRIPT="$SCRIPT_DIR/thunder_train.py"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Thunder-Train: Distributed LoRA Training${NC}"
echo "=========================================="

# Check hostfile exists
if [ ! -f "$HOSTFILE" ]; then
    echo -e "${RED}ERROR: hostfile.json not found at $HOSTFILE${NC}"
    exit 1
fi

# Check training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}ERROR: thunder_train.py not found at $TRAIN_SCRIPT${NC}"
    exit 1
fi

# Check MLX version
echo -n "Checking MLX version... "
MLX_VERSION=$(python3 -c "import mlx.core as mx; print(mx.__version__)" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo -e "${RED}FAILED (MLX not installed)${NC}"
    exit 1
fi

# Parse version: need >= 0.30
MLX_MAJOR=$(echo "$MLX_VERSION" | cut -d. -f1)
MLX_MINOR=$(echo "$MLX_VERSION" | cut -d. -f2)
if [ "$MLX_MAJOR" -eq 0 ] && [ "$MLX_MINOR" -lt 30 ]; then
    echo -e "${RED}FAILED (MLX $MLX_VERSION, need >= 0.30)${NC}"
    exit 1
fi
echo -e "${GREEN}MLX $MLX_VERSION${NC}"

# Check mlx distributed launcher exists
echo -n "Checking mlx distributed launcher... "
if ! python3 -c "import mlx._distributed_utils.launch" &> /dev/null; then
    echo -e "${RED}FAILED (mlx._distributed_utils not found — need mlx >= 0.30)${NC}"
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# Quick connectivity check (non-blocking, 2s timeout)
echo -n "Checking TB link... "
if ping -c 1 -W 2000 10.0.5.1 &> /dev/null && ping -c 1 -W 2000 10.0.5.2 &> /dev/null; then
    echo -e "${GREEN}Mac4 + Mac5 reachable${NC}"
else
    echo -e "${YELLOW}WARNING: Could not ping both machines. Proceeding anyway (mlx.launch will report SSH errors).${NC}"
fi

echo ""
echo "Hostfile: $HOSTFILE"
echo "Script:   $TRAIN_SCRIPT"
echo "Args:     $@"
echo ""

# Set environment for optimal GPU sync
export MLX_METAL_FAST_SYNCH=1

# Launch distributed training
echo -e "${GREEN}Launching distributed training...${NC}"
echo "---"

python3 "$SCRIPT_DIR/mlx_launch.py" \
    --hostfile "$HOSTFILE" \
    --backend ring \
    --python "$HOME/bin/python3" \
    --verbose \
    -- "$TRAIN_SCRIPT" "$@"

echo "---"
echo -e "${GREEN}Thunder-Train complete.${NC}"
