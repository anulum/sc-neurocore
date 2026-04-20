# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/spatial_generative_demo

module SpatialGenerativeDemoAccel

using Statistics, LinearAlgebra

function run_spatial_gen_demo()
    print("--- SPATIAL GENERATIVE MULTIMODAL DEMO ---")
    # 1. World Model & Planning
    print("\n[1] Testing World Model & Planner...")
    wm = PredictiveWorldModel(state_dim=4, action_dim=2)
    planner = SCPlanner(wm)
    curr_s = collect([0.1, 0.1, 0.1, 0.1])
    goal_s = collect([0.9, 0.9, 0.9, 0.9])
    plan = planner.plan_sequence(curr_s, goal_s, horizon=3)
    print(f"    Generated Plan (3 steps): {length(plan)} actions.")
    # 2. Spatial Stack
    print("\n[2] Testing Spatial Stack (3D)...")
    voxels = VoxelGrid(resolution=8)
    voxels.set_voxel(4, 4, 4, 1.0)  # Center occupied
    st3d = SpatialTransformer3D(resolution=8, dim_k=4)
    processed_voxels = st3d.forward(voxels.data)
    print(f"    Processed Voxel Grid Mean: {mean(processed_voxels):.4f}")
    # 3. Multimodal Generative
    print("\n[3] Testing Multimodal Generative Outputs...")
    text_gen = SCTextGenerator(vocab=["move", "fire", "detect", "human"])
    print(f"    Generated Text: '{text_gen.generate_sequence(4)}'")
    audio_syn = SCAudioSynthesizer()
    waveform = audio_syn.synthesize_tone(frequency=440, duration_ms=100, probability=0.5)
    print(f"    Audio Waveform Size: {length(waveform)} samples.")
    gen_3d = SC3DGenerator()
    gen_3d.export_point_cloud_json(rand(10, 3), rand(10), "test_points.json")
    # 4. Ensemble
    print("\n[4] Testing Agent Ensembles...")
    ens = EnsembleOrchestrator()
    ens.add_agent("Lead", CognitiveOrchestrator())
    ens.add_agent("Support", CognitiveOrchestrator())
    ens.coordinated_mission("Navigation")
end

end # module SpatialGenerativeDemoAccel
