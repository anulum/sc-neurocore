# SC-NEUROCORE Posner ORCA follow-up pipeline

created_at_utc=2026-06-14T02:16:59Z
host=god-of-the-math
run_root=/home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z
source_xyz=/home/anulum/sc-neurocore-orca-runs/ml350_r6_continuation_20260531/run/posner_ml350_neutral_opt_20260531_r7_continue.xyz
orca_binary=/home/anulum/.local/sc-neurocore/orca-6.1.1/orca
nprocs=24
maxcore_mb_per_rank=4000
estimated_memory_ceiling_mb=96000
mpi_env=HWLOC_COMPONENTS=-gl OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
thread_env=OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

## Jobs

1. 01_neutral_nmr_r7: neutral r7 geometry, B3LYP/def2-TZVP NMR, 31P SHIFT+SSALL, Ca SHIFT.
2. 02_hydration6_sp: heuristic six-water first-shell candidate, single-point energy only.
3. 03_dimer_sp: x-separated neutral dimer first scan point, single-point energy only.

Hydration and dimer geometries are initial candidates, not optimised structural evidence.
3ef73733b0eff3b81f1487cc2fe8c113fd9b6128e7ce28e9d1528dac7671e8e9  /home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z/01_neutral_nmr_r7/run/input.xyz
7131c1233442e4bf049cbf5c076ecb1e005b37a2bd6a6b48557aecc10d75e4ce  /home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z/02_hydration6_sp/run/input.xyz
f474a526ec907180f85856292c243362f8be6a15022da6a110f94b28639fd2b8  /home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z/03_dimer_sp/run/input.xyz
12adcea685a0c30ae9d0046b108708de76ee1743c368724ee6cb91f3048b1796  /home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z/01_neutral_nmr_r7/run/posner_neutral_nmr_r7.inp
4000002c4dee371c706463bde4d23cabad011e4acc07b8aab01dd2ec0d7c1dcd  /home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z/02_hydration6_sp/run/posner_hydration6_sp.inp
4000002c4dee371c706463bde4d23cabad011e4acc07b8aab01dd2ec0d7c1dcd  /home/anulum/sc-neurocore-orca-runs/posner_followup_20260614T0218Z/03_dimer_sp/run/posner_dimer_sp.inp
