#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z"
ORCA="/home/anulum/.local/sc-neurocore/orca-6.1.1/orca"
LOCK=/home/anulum/compute-queue/active_sc_neurocore_orca_followup.lock
export HWLOC_COMPONENTS=-gl
export OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "codex sc-neurocore-posner-followup-nmr-hydration-dimer 720" > "$LOCK"
cleanup() { rm -f "$LOCK"; }
trap cleanup EXIT
run_job() {
  local job="$1" inp="$2"
  local dir="$ROOT/$job/run" outdir="$ROOT/$job/output"
  mkdir -p "$outdir"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/started_at.txt"
  cd "$dir"
  taskset -c 0-23 "$ORCA" "$inp" > "$outdir/${inp%.inp}.out" 2>&1
  echo $? > "$outdir/exit_status.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/finished_at.txt"
}
run_job 01_neutral_nmr_r7 posner_neutral_nmr_r7.inp
run_job 02_hydration6_sp posner_hydration6_sp.inp
run_job 03_dimer_sp posner_dimer_sp.inp
