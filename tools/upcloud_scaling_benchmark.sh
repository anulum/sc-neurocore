#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# © 1998–2026 Miroslav Šotek. All rights reserved.
#
# SC-NeuroCore 6-phase scaling benchmark runner for UpCloud GPU servers.
# Target: NVIDIA L40S GPU + AMD EPYC 9575F (Zen 5, AVX-512).
#
# Phases:
#   1. System info collection
#   2. Rust criterion benchmarks (scaling_bench + full_bench)
#   3. Python scaling — 4 Brunel regimes × 7 scales × 3 repeats
#   4. SC network benchmark — 5 sizes × 3 bitstream lengths
#   5. Rust-vs-Python parity benchmark
#   6. GPU-extended scaling (sparse CUDA, up to 100K)
#
# Usage:
#   rsync -az --exclude target/ --exclude .git/ . root@<ip>:/opt/sc-neurocore/
#   ssh root@<ip> 'cd /opt/sc-neurocore && bash tools/upcloud_scaling_benchmark.sh'
#
# Estimated runtime: ~45 min.  Cost at €1.11/hr (1×L40S): ~€0.85

set -euo pipefail

RESULTS_DIR="benchmarks/results/upcloud"
mkdir -p "$RESULTS_DIR"

log() { echo "=== $(date +%H:%M:%S) Phase $1: $2 ==="; }

# ---- Phase 1: System info ----
log 1 "Collecting system info"
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
log 1 "Installing dependencies"
apt-get update -qq && apt-get install -y -qq build-essential pkg-config curl git python3 python3-pip python3-venv > /dev/null

if ! command -v cargo &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  source "$HOME/.cargo/env"
fi
rustup default stable
rustc --version

python3 -m venv /opt/bench-venv
source /opt/bench-venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy==2.2.3 scipy==1.15.2 brian2==2.7.1 nest-simulator==3.8

if nvidia-smi &>/dev/null; then
  pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip install --quiet norse==1.1.0 snntorch==0.9.1 || true
fi

pip install --quiet -e ".[dev]" 2>/dev/null || pip install --quiet -e . 2>/dev/null || true

# ---- Phase 2: Rust criterion benchmarks ----
log 2 "Rust scaling benchmarks (Criterion)"
cd engine
cargo bench --bench scaling_bench -- --output-format bencher 2>/dev/null | tee "../$RESULTS_DIR/rust_scaling_bench.txt"
cd ..

# ---- Phase 3: Python scaling — 4 Brunel regimes ----
log 3 "Python scaling benchmark (4 regimes × 7 scales × 3 repeats)"
python benchmarks/scaling_benchmark.py \
  --scales 1000 2000 5000 10000 20000 50000 \
  --regimes SR SI AI AR \
  --sim-ms 500 \
  --repeats 3 \
  --json "$RESULTS_DIR/scaling_results.json" \
  --markdown 2>&1 | tee "$RESULTS_DIR/scaling_results.md"

# ---- Phase 4: SC network benchmark ----
log 4 "SC network benchmark (5 sizes × 3 bitstream lengths)"
python benchmarks/sc_network_benchmark.py \
  --scales 100 200 500 1000 2000 \
  --bitstream-lengths 256 512 1024 \
  --sim-steps 50 \
  --repeats 3 \
  --json "$RESULTS_DIR/sc_network_results.json" \
  --markdown 2>&1 | tee "$RESULTS_DIR/sc_network_results.md"

# ---- Phase 5: Rust-vs-Python parity ----
log 5 "Rust-vs-Python parity benchmark"
python benchmarks/rust_python_parity_bench.py \
  --repeats 3 \
  --json "$RESULTS_DIR/rust_python_parity.json" \
  --markdown 2>&1 | tee "$RESULTS_DIR/rust_python_parity.md"

# ---- Phase 6: GPU-extended scaling (includes Norse + snnTorch) ----
if nvidia-smi &>/dev/null; then
  log 6 "GPU-extended scaling (up to 100K, all GPU sims)"
  python benchmarks/scaling_benchmark.py \
    --scales 20000 50000 100000 \
    --regimes AI SR SI AR \
    --sim-ms 500 \
    --repeats 3 \
    --simulators sc_pytorch_cuda sc_pytorch_cuda_sparse norse snntorch \
    --json "$RESULTS_DIR/gpu_scaling_results.json" \
    --markdown 2>&1 | tee "$RESULTS_DIR/gpu_scaling_results.md"
else
  log 6 "SKIP — no NVIDIA GPU detected"
fi

# ---- Summary ----
echo ""
echo "=== $(date +%H:%M:%S) Benchmark complete ==="
echo ""
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
echo "To download results:"
echo "  scp -r root@\$(hostname -I | awk '{print \$1}'):$(pwd)/$RESULTS_DIR/ ."
