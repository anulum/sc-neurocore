# SC-NEUROCORE Posner ORCA tier-2 physics pipeline

created_at_utc=2026-06-14T07:18:00Z
host=god-of-the-math
run_root=/home/anulum/sc-neurocore-orca-runs/posner_tier2_physics_20260614T0718Z
predecessor_run_root=/home/anulum/sc-neurocore-orca-runs/posner_handoff_extension_20260614T0340Z
orca_binary=/home/anulum/.local/sc-neurocore/orca-6.1.1/orca
nprocs=24
cpu_affinity=taskset -c 0-23
maxcore_mb_per_rank=4000
estimated_memory_ceiling_mb=96000
mpi_env=HWLOC_COMPONENTS=-gl OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
thread_env=OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
waits_for_lock=/home/anulum/compute-queue/active_sc_neurocore_orca_handoff_extension.lock
own_lock=/home/anulum/compute-queue/active_sc_neurocore_orca_tier2_physics.lock

## Jobs

1. 07_hydration6_opt: optimized six-water hydrated cluster geometry, B3LYP/def2-TZVP D3BJ RIJCOSX Opt.
2. 08_dimer_opt: optimized neutral dimer geometry from the current first dimer candidate, B3LYP/def2-TZVP D3BJ RIJCOSX Opt.
3. 09_bsse_counterpoise: manual counterpoise set generated after dimer optimization: full dimer, isolated monomers, and monomers with partner ghost basis. Boundary: ghost-atom syntax is explicit in generated XYZ; final binding statement must be extracted from component energies.
4. 10_dimer_distance_scan: rigid 5-point monomer translation scan at dx = 10, 12, 14, 16, 18 A using optimized dimer first fragment as monomer reference.
5. 11_hydration6_cpcm_sp: CPCM(Water) B3LYP single point on optimized hydrated geometry.
6. 12_hydration6_pbe0_sp: PBE0 single point on optimized hydrated geometry.
7. 13_dimer_pbe0_sp: PBE0 single point on optimized dimer geometry.

## Initial input hashes

7131c1233442e4bf049cbf5c076ecb1e005b37a2bd6a6b48557aecc10d75e4ce  /home/anulum/sc-neurocore-orca-runs/posner_tier2_physics_20260614T0718Z/07_hydration6_opt/run/input.xyz
b6af267015bc6c066040b9d1a650e74e488450236897c4b09fc3ca9e9767bc51  /home/anulum/sc-neurocore-orca-runs/posner_tier2_physics_20260614T0718Z/07_hydration6_opt/run/posner_hydration6_opt.inp
f474a526ec907180f85856292c243362f8be6a15022da6a110f94b28639fd2b8  /home/anulum/sc-neurocore-orca-runs/posner_tier2_physics_20260614T0718Z/08_dimer_opt/run/input.xyz
d2937b0d6033e975842ec12aba59d19ae7390489cb517efa5cc66346b9496f9b  /home/anulum/sc-neurocore-orca-runs/posner_tier2_physics_20260614T0718Z/08_dimer_opt/run/posner_dimer_opt.inp
