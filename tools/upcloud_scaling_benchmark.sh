#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# SC-NeuroCore scaling benchmark runner for UpCloud GPU servers.
# Target: NVIDIA L40S GPU + AMD EPYC 9575F (Zen 5, AVX-512).
#
# Measures wall-clock time, memory footprint, and spike statistics
# as neuron count scales from 1K to 50K on the Brunel balanced
# network. Compares SC-NeuroCore (NumPy dense, NumPy sparse,
# PyTorch CUDA) vs Brian2 vs NEST (if available).
#
# Usage:
#   rsync -az --exclude target/ --exclude .git/ . root@<ip>:/opt/sc-neurocore/
#   ssh root@<ip> 'cd /opt/sc-neurocore && bash tools/upcloud_scaling_benchmark.sh'
#
# Estimated runtime: ~20 min (Python scaling) + ~10 min (Rust scaling) = ~30 min
# Cost at €1.11/hr (1×L40S): ~€0.55

set -euo pipefail

RESULTS_DIR="benchmarks/results/upcloud"
mkdir -p "$RESULTS_DIR"

log() { echo "=== $(date +%H:%M:%S) $1 ==="; }

# ---- System info ----
log "Collecting system info"
{
  echo "# UpCloud Scaling Benchmark Environment"
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Hostname: $(hostname)"
  echo ""
  echo "## CPU"
  lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket|Flags' || true
  echo ""
  echo "## Memory"
  free -h | head -2
  echo ""
  echo "## GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected"
  echo ""
  echo "## OS"
  uname -a
  cat /etc/os-release 2>/dev/null | head -3 || true
  echo ""
  echo "## SIMD"
  if grep -q avx512 /proc/cpuinfo 2>/dev/null; then echo "AVX-512: YES"; else echo "AVX-512: NO"; fi
  if grep -q avx2 /proc/cpuinfo 2>/dev/null; then echo "AVX2: YES"; else echo "AVX2: NO"; fi
} > "$RESULTS_DIR/scaling_system_info.md"
cat "$RESULTS_DIR/scaling_system_info.md"

# ---- Install dependencies ----
log "Installing system packages"
apt-get update -qq && apt-get install -y -qq build-essential pkg-config curl git python3 python3-pip python3-venv > /dev/null

# Rust
if ! command -v cargo &>/dev/null; then
  log "Installing Rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  source "$HOME/.cargo/env"
fi
rustup default stable
rustc --version

# Python venv
log "Setting up Python venv"
python3 -m venv /opt/bench-venv
source /opt/bench-venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy scipy brian2

# GPU packages
if nvidia-smi &>/dev/null; then
  log "Installing GPU packages"
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
fi

# Install sc-neurocore
log "Installing sc-neurocore"
pip install --quiet -e ".[dev]" 2>/dev/null || pip install --quiet -e . 2>/dev/null || true

# ---- Phase 1: Rust Scaling Benchmarks (Criterion) ----
log "Phase 1: Rust scaling benchmarks"
cd engine

cargo bench --bench scaling_bench -- --output-format bencher 2>/dev/null | tee "../$RESULTS_DIR/rust_scaling_bench.txt"

cd ..

# ---- Phase 2: Python Scaling Benchmark ----
log "Phase 2: Python scaling benchmark (1K → 50K neurons, 3 repeats)"
python benchmarks/scaling_benchmark.py \
  --scales 1000 2000 5000 10000 20000 50000 \
  --sim-ms 500 \
  --repeats 3 \
  --json "$RESULTS_DIR/scaling_results.json" \
  --markdown 2>&1 | tee "$RESULTS_DIR/scaling_results.md"

# ---- Phase 3: Extended GPU scaling (if CUDA available) ----
if nvidia-smi &>/dev/null; then
  log "Phase 3: GPU-only extended scaling (up to 100K neurons)"
  python benchmarks/scaling_benchmark.py \
    --scales 10000 20000 50000 100000 \
    --sim-ms 200 \
    --repeats 3 \
    --simulators sc_pytorch_cuda \
    --json "$RESULTS_DIR/gpu_scaling_results.json" \
    --markdown 2>&1 | tee "$RESULTS_DIR/gpu_scaling_results.md"
fi

# ---- Summary ----
log "Scaling benchmark complete"
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
echo "To download results:"
echo "  scp -r root@\$(hostname -I | awk '{print \$1}'):$(pwd)/$RESULTS_DIR/ ."
