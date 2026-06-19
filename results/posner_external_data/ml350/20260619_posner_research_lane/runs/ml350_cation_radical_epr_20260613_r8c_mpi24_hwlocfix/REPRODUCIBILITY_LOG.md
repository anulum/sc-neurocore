# SC-NEUROCORE ORCA r8c MPI24 Reproducibility Log

created_at_utc=2026-06-13T20:51:41+00:00
host=god-of-the-math
run_root=/home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix
tmux_session=scn_orca_r8c_mpi24
lock_file=/home/anulum/compute-queue/active_sc_neurocore_orca_r8c.lock

## Purpose

Cation-radical EPR/HFC single-point continuation from the converged r7 neutral Posner geometry.
This supersedes r8b single-rank after fixing OpenMPI local launch hangs.

## MPI Fix

OpenMPI hung because hwloc probed GL/X11 sockets. The verified fix is:

- HWLOC_COMPONENTS=-gl
- OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1

Smoke evidence: env HWLOC_COMPONENTS=-gl OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1 mpirun -np 24 hostname returned 24 local ranks.

## Compute

orca_binary=/home/anulum/.local/sc-neurocore/orca-6.1.1/orca
nprocs=24
maxcore_mb_per_rank=4000
estimated_memory_ceiling_mb=96000
thread_env=OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

## Chemistry

method=UKS B3LYP
basis=def2-TZVP
dispersion=D3BJ
acceleration=RIJCOSX
scf=VeryTightSCF
grid=DefGrid3
job_type=SP
charge=1
multiplicity=2
eprnmr=gtensor true; nuclei all P {aiso, adip, aorb}; nuclei all Ca {aiso, adip}; printlevel 5

## Source

source_run=/home/anulum/sc-neurocore-orca-runs/ml350_r6_continuation_20260531/run
source_xyz=/home/anulum/sc-neurocore-orca-runs/ml350_r6_continuation_20260531/run/posner_ml350_neutral_opt_20260531_r7_continue.xyz
source_out=/home/anulum/sc-neurocore-orca-runs/ml350_r6_continuation_20260531/run/posner_ml350_neutral_opt_20260531_r7_continue.out

## Hashes

3ef73733b0eff3b81f1487cc2fe8c113fd9b6128e7ce28e9d1528dac7671e8e9  /home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/run/input.xyz
c6f58d96400255a048b27dd8976663055947f0fffa27365722ba28a646e5a525  /home/anulum/sc-neurocore-orca-runs/ml350_cation_radical_epr_20260613_r8c_mpi24_hwlocfix/run/posner_cation_radical_epr_r8c.inp
