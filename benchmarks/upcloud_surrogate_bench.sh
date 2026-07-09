#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — UpCloud Surrogate Gradient Training Benchmark
#
# Head-to-head: SC-NeuroCore vs Norse vs snnTorch on MNIST
# Architecture: 784 → 128 → 128 → 10, T=25, beta=0.9
#
# Usage:
#   ssh root@<upcloud-ip>
#   git clone https://github.com/anulum/sc-neurocore.git && cd sc-neurocore
#   bash benchmarks/upcloud_surrogate_bench.sh
#
# Prerequisites: Ubuntu 24.04, Python 3.12+
# Estimated runtime: ~8 min (CPU), ~2 min (GPU)
# Estimated cost: ~€0.02 on HICPU-8xCPU-16GB

set -euo pipefail

RESULTS_DIR="benchmarks/results/upcloud_surrogate_$(date -u +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " SNN Surrogate Gradient Training Benchmark"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"

# --- System info ---
echo ""
echo "--- System ---"
uname -a
python3 --version
cat /proc/cpuinfo | grep "model name" | head -1 || true
free -h | head -2

python3 -c "
import json, platform, os
info = {
    'platform': platform.platform(),
    'python': platform.python_version(),
    'cpu': 'unknown',
    'cpu_count': os.cpu_count(),
}
try:
    with open('/proc/cpuinfo') as f:
        for line in f:
            if line.startswith('model name'):
                info['cpu'] = line.split(':')[1].strip()
                break
except: pass
print(json.dumps(info, indent=2))
" > "$RESULTS_DIR/system_info.json"

# --- Setup Python env ---
echo ""
echo "--- Setting up Python environment ---"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip

# Install SC-NeuroCore with training deps
pip install -q -e ".[dev,training]"
pip install -q torchvision==0.21.0

# Install competitors
pip install -q norse==1.1.0 snntorch==0.9.1

echo ""
echo "--- Environment ready ---"
python -c "import sc_neurocore; print(f'sc-neurocore {sc_neurocore.__version__}')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import norse; print(f'norse {norse.__version__}')" 2>/dev/null || echo "norse: not available"
python -c "import snntorch; print(f'snntorch {snntorch.__version__}')" 2>/dev/null || echo "snntorch: not available"

# --- Detect device ---
DEVICE="cpu"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo "Using device: $DEVICE"

# ===================================================================
# Run benchmark: all 3 frameworks, 10 epochs
# ===================================================================
echo ""
echo "============================================"
echo " Running 3-framework benchmark (10 epochs)"
echo "============================================"
cd benchmarks
python surrogate_training_bench.py \
    --epochs 10 \
    --device "$DEVICE" \
    --json "../$RESULTS_DIR/surrogate_training.json" \
    2>&1 | tee "../$RESULTS_DIR/surrogate_training.log"
cd ..

# ===================================================================
# Summary
# ===================================================================
echo ""
echo "============================================"
echo " Benchmark complete"
echo " Results in: $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR/"
echo ""
echo "To copy results back:"
echo "  scp -r root@<ip>:sc-neurocore/$RESULTS_DIR ."
