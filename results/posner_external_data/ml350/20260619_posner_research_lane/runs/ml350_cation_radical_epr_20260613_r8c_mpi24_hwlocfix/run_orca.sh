#!/usr/bin/env bash
set -u -o pipefail
cd "/home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/run"
export HWLOC_COMPONENTS=-gl
export OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ORCA_MPI_RANKS=24
export ORCA_MAXCORE_MB=4000
printf '%s\n' "$(date -Is)" > "/home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/output/started_at.txt"
set +e
"/home/anulum/.local/sc-neurocore/orca-6.1.1/orca" posner_cation_radical_epr_r8c.inp > "/home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/output/posner_cation_radical_epr_r8c.out" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "/home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/output/exit_status.txt"
printf '%s\n' "$(date -Is)" > "/home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/output/finished_at.txt"
rm -f "/home/anulum/compute-queue/active_sc_neurocore_orca_r8c.lock"
exit "$status"
