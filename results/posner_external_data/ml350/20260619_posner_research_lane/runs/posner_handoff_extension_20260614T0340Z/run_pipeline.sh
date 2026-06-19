#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/anulum/sc-neurocore-orca-runs/posner_handoff_extension_20260614T0340Z"
PREV="/home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z"
ORCA="/home/anulum/.local/sc-neurocore/orca-6.1.1/orca"
PREV_LOCK=/home/anulum/compute-queue/active_sc_neurocore_orca_followup.lock
LOCK=/home/anulum/compute-queue/active_sc_neurocore_orca_handoff_extension.lock
export HWLOC_COMPONENTS=-gl
export OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
while [[ -e "$PREV_LOCK" ]]; do
  sleep 300
done
for job in 01_neutral_nmr_r7 02_hydration6_sp 03_dimer_sp; do
  status="$PREV/$job/output/exit_status.txt"
  out="$PREV/$job/output/${job#??_}.out"
  case "$job" in
    01_neutral_nmr_r7) out="$PREV/$job/output/posner_neutral_nmr_r7.out" ;;
    02_hydration6_sp) out="$PREV/$job/output/posner_hydration6_sp.out" ;;
    03_dimer_sp) out="$PREV/$job/output/posner_dimer_sp.out" ;;
  esac
  [[ -f "$status" ]] || { echo "missing status for $job" >&2; exit 20; }
  [[ "$(cat "$status")" == "0" ]] || { echo "nonzero status for $job" >&2; exit 21; }
  grep -q 'ORCA TERMINATED NORMALLY' "$out" || { echo "missing normal termination for $job" >&2; exit 22; }
done
echo "codex sc-neurocore-posner-handoff-extension 720" > "$LOCK"
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
run_job 04_hydration6_nmr posner_hydration6_nmr.inp
run_job 05_neutral_ir_freq_r7 posner_neutral_ir_freq_r7.inp
run_job 06_dimer_far_sp posner_dimer_far_sp.inp
