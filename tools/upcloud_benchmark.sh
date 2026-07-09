#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore full benchmark runner for UpCloud GPU servers.
# Targets: NVIDIA L40S GPU + AMD EPYC 9575F (Zen 5, AVX-512).
#
# Usage:
#   ssh root@<upcloud-ip> 'bash -s' < tools/upcloud_benchmark.sh
#
# Or copy the repo first:
#   rsync -az --exclude target/ --exclude .git/ . root@<ip>:/opt/sc-neurocore/
#   ssh root@<ip> 'cd /opt/sc-neurocore && bash tools/upcloud_benchmark.sh'
#
# Estimated runtime: ~15 min (Rust) + ~10 min (Python) = ~25 min total.
# Cost at €1.11/hr (1×L40S): ~€0.50

set -euo pipefail

RESULTS_DIR="benchmarks/results/upcloud"
mkdir -p "$RESULTS_DIR"

log() { echo "=== $(date +%H:%M:%S) $1 ==="; }

# ---- System info ----
log "Collecting system info"
{
  echo "# UpCloud Benchmark Environment"
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
} > "$RESULTS_DIR/system_info.md"
cat "$RESULTS_DIR/system_info.md"

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
pip install --quiet numpy==2.2.3 brian2==2.7.1

# GPU Python packages (if NVIDIA GPU present)
if nvidia-smi &>/dev/null; then
  log "Installing GPU packages (CuPy + PyTorch CUDA)"
  pip install --quiet cupy-cuda12x==13.4.1 torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

# ---- Phase 1: Rust Criterion Benchmarks (CPU, AVX-512) ----
log "Phase 1: Rust engine benchmarks (criterion)"
cd engine

# Full benchmark suite
cargo bench --bench full_bench -- --output-format bencher 2>/dev/null | tee "../$RESULTS_DIR/rust_full_bench.txt"

# Bitstream-only benchmark
cargo bench --bench bitstream_bench -- --output-format bencher 2>/dev/null | tee "../$RESULTS_DIR/rust_bitstream_bench.txt"

# Detect AVX-512 usage
log "Checking SIMD features"
{
  echo "## SIMD Feature Detection"
  echo '```'
  if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    echo "AVX-512: SUPPORTED"
  else
    echo "AVX-512: not available"
  fi
  if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo "AVX2: SUPPORTED"
  else
    echo "AVX2: not available"
  fi
  echo '```'
} >> "../$RESULTS_DIR/system_info.md"

cd ..

# ---- Phase 2: Rust cargo test (timing) ----
log "Phase 2: Rust test suite"
cd engine
cargo test --release 2>&1 | tail -20 | tee "../$RESULTS_DIR/rust_tests.txt"
cd ..

# ---- Phase 3: Python benchmark suite (CPU + GPU) ----
log "Phase 3: Python benchmark suite"
pip install --quiet -e ".[dev]" 2>/dev/null || pip install --quiet -e . 2>/dev/null || true

python benchmarks/benchmark_suite.py --markdown > "$RESULTS_DIR/python_benchmarks.md" 2>&1 || \
  python benchmarks/benchmark_suite.py > "$RESULTS_DIR/python_benchmarks.md" 2>&1 || \
  echo "Python benchmark_suite.py failed" > "$RESULTS_DIR/python_benchmarks.md"
cat "$RESULTS_DIR/python_benchmarks.md"

# ---- Phase 4: SNN Comparison (20 variants + Brian2) ----
log "Phase 4: SNN comparison benchmark"
cd benchmarks
python snn_comparison.py --all --json "../$RESULTS_DIR/snn_comparison_upcloud.json" \
  --markdown > "../$RESULTS_DIR/snn_comparison_upcloud.md" 2>&1 || \
  echo "snn_comparison.py failed or skipped" > "../$RESULTS_DIR/snn_comparison_upcloud.md"
cd ..
cat "$RESULTS_DIR/snn_comparison_upcloud.md"

# ---- Phase 5: Python test suite ----
log "Phase 5: Python test suite"
python -m pytest tests/ -q --tb=short 2>&1 | tail -5 | tee "$RESULTS_DIR/python_tests.txt"

# ---- Summary ----
log "Benchmark complete"
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
echo "To download results:"
echo "  scp -r root@\$(hostname -I | awk '{print \$1}'):$(pwd)/$RESULTS_DIR/ ."
