#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — UpCloud P4 Benchmark Re-Run
#
# Runs the 3 P4 items from TODO_MARKET_READINESS.md on UpCloud EPYC:
#   1. Brunel v12_stdp_lif (snn_comparison.py --variant v12)
#   2. Rust Criterion kuramoto (cargo bench --bench full_bench kuramoto)
#   3. Brunel scaling spot-check (scaling_benchmark.py --scales 1000 5000)
#   + Brian2 head-to-head re-run with current code (brian2_benchmark.py)
#
# Usage:
#   ssh root@<upcloud-ip>
#   git clone https://github.com/anulum/sc-neurocore.git && cd sc-neurocore
#   bash benchmarks/upcloud_rerun_p4.sh
#
# Prerequisites: Ubuntu 24.04, Python 3.12+, Rust toolchain
# Estimated runtime: ~20 min (v12 is the bottleneck at ~160s)
# Estimated cost: ~€0.20 on GPU-8xCPU-64GB-1xL40S

set -euo pipefail

RESULTS_DIR="benchmarks/results/upcloud_p4_rerun_$(date -u +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " SC-NeuroCore P4 Benchmark Re-Run"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"

# --- System info ---
echo ""
echo "--- System ---"
uname -a
python3 --version
if command -v rustc &>/dev/null; then
    rustc --version
    cargo --version
fi
cat /proc/cpuinfo | grep "model name" | head -1 || true
free -h | head -2

# Capture system info to JSON
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
pip install -q -e ".[dev]"
pip install -q brian2==2.7.1

echo "--- Environment ready ---"
python -c "import sc_neurocore; print(f'sc-neurocore {sc_neurocore.__version__}')"
python -c "import brian2; print(f'brian2 {brian2.__version__}')"
python -c "import numba; print(f'numba {numba.__version__}')"

# ===================================================================
# P4.1: Brunel v12_stdp_lif re-run
# ===================================================================
echo ""
echo "============================================"
echo " P4.1: Brunel v12_stdp_lif"
echo "============================================"
cd benchmarks
python snn_comparison.py \
    --variant brian2 --variant v12 --variant v18 --variant v21 \
    --json "../$RESULTS_DIR/snn_v12_rerun.json" \
    --markdown
cd ..

# ===================================================================
# P4.2: Rust Criterion kuramoto re-run
# ===================================================================
echo ""
echo "============================================"
echo " P4.2: Rust Criterion kuramoto"
echo "============================================"
if command -v cargo &>/dev/null; then
    cd engine
    cargo bench --bench full_bench -- kuramoto 2>&1 | tee "../$RESULTS_DIR/rust_kuramoto.txt"

    # Also run full bench for comparison with previous UpCloud numbers
    echo ""
    echo "--- Full Criterion bench ---"
    cargo bench --bench full_bench 2>&1 | tee "../$RESULTS_DIR/rust_full_bench.txt"
    cd ..
else
    echo "SKIP: Rust toolchain not installed"
    echo "Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

# ===================================================================
# P4.3: Brunel scaling spot-check
# ===================================================================
echo ""
echo "============================================"
echo " P4.3: Brunel scaling spot-check"
echo "============================================"
cd benchmarks
python scaling_benchmark.py \
    --scales 1000 5000 \
    --regimes AI \
    --repeats 3 \
    --json "../$RESULTS_DIR/scaling_spotcheck.json" \
    --markdown
cd ..

# ===================================================================
# Bonus: Brian2 head-to-head (fresh numbers with PoissonInput fix)
# ===================================================================
echo ""
echo "============================================"
echo " Brian2 head-to-head (1K + 10K)"
echo "============================================"
cd benchmarks
python brian2_benchmark.py \
    --scales 1000 10000 \
    --repeats 3 \
    --json "../$RESULTS_DIR/brian2_headtohead.json" \
    --markdown
cd ..

# ===================================================================
# Summary
# ===================================================================
echo ""
echo "============================================"
echo " All P4 benchmarks complete"
echo " Results in: $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR/"
echo ""
echo "To copy results back:"
echo "  scp -r root@<ip>:sc-neurocore/$RESULTS_DIR ."
