#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/anulum/sc-neurocore-orca-runs/posner_tier2_physics_20260614T0718Z"
PREV="/home/anulum/sc-neurocore-orca-runs/posner_handoff_extension_20260614T0340Z"
ORCA="/home/anulum/.local/sc-neurocore/orca-6.1.1/orca"
PREV_LOCK=/home/anulum/compute-queue/active_sc_neurocore_orca_handoff_extension.lock
LOCK=/home/anulum/compute-queue/active_sc_neurocore_orca_tier2_physics.lock
export HWLOC_COMPONENTS=-gl
export OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
while [[ -e "$PREV_LOCK" ]]; do
  sleep 300
done
for job in 04_hydration6_nmr 05_neutral_ir_freq_r7 06_dimer_far_sp; do
  status="$PREV/$job/output/exit_status.txt"
  [[ -f "$status" ]] || { echo "missing status for $job" >&2; exit 20; }
  [[ "$(cat "$status")" == "0" ]] || { echo "nonzero status for $job" >&2; exit 21; }
  out_count=$(find "$PREV/$job/output" -maxdepth 1 -name '*.out' -print | wc -l)
  [[ "$out_count" -ge 1 ]] || { echo "missing ORCA output for $job" >&2; exit 22; }
  grep -q 'ORCA TERMINATED NORMALLY' "$PREV/$job/output"/*.out || { echo "missing normal termination for $job" >&2; exit 23; }
done
echo "codex sc-neurocore-posner-tier2-physics-fullhost 4320" > "$LOCK"
cleanup() { rm -f "$LOCK"; }
trap cleanup EXIT
run_job() {
  local job="$1" inp="$2"
  local dir="$ROOT/$job/run" outdir="$ROOT/$job/output"
  mkdir -p "$outdir"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/started_at.txt"
  cd "$dir"
  set +e
  taskset -c 0-23 "$ORCA" "$inp" > "$outdir/${inp%.inp}.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$outdir/exit_status.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/finished_at.txt"
  find "$dir" -maxdepth 1 -name '*.xyz' -type f -exec cp -f {} "$outdir" \;
  return "$rc"
}
run_nested_job() {
  local group="$1" name="$2" inp="$3"
  local dir="$ROOT/$group/run/$name" outdir="$ROOT/$group/output/$name"
  mkdir -p "$outdir"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/started_at.txt"
  cd "$dir"
  set +e
  taskset -c 0-23 "$ORCA" "$inp" > "$outdir/${inp%.inp}.out" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$outdir/exit_status.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/finished_at.txt"
  find "$dir" -maxdepth 1 -name '*.xyz' -type f -exec cp -f {} "$outdir" \;
  return "$rc"
}
run_job 07_hydration6_opt posner_hydration6_opt.inp
run_job 08_dimer_opt posner_dimer_opt.inp
python3 "$ROOT/scripts/generate_dependent_inputs.py"
for name in dimer_full_basis mono_a_own_basis mono_b_own_basis mono_a_full_basis mono_b_full_basis; do
  run_nested_job 09_bsse_counterpoise "$name" "$name.inp"
done
for name in distance_10A distance_12A distance_14A distance_16A distance_18A; do
  run_nested_job 10_dimer_distance_scan "$name" "$name.inp"
done
run_job 11_hydration6_cpcm_sp posner_hydration6_cpcm_sp.inp
run_job 12_hydration6_pbe0_sp posner_hydration6_pbe0_sp.inp
run_job 13_dimer_pbe0_sp posner_dimer_pbe0_sp.inp
