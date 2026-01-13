
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
from sc_neurocore.world_model.planner import SCPlanner
from sc_neurocore.spatial.representations import VoxelGrid
from sc_neurocore.spatial.transformer_3d import SpatialTransformer3D
from sc_neurocore.generative.text_gen import SCTextGenerator
from sc_neurocore.generative.audio_synthesis import SCAudioSynthesizer
from sc_neurocore.generative.three_d_gen import SC3DGenerator
from sc_neurocore.pipeline.ingestion import DataIngestor
from sc_neurocore.ensembles.orchestrator import EnsembleOrchestrator
from sc_neurocore.core.orchestrator import CognitiveOrchestrator

def run_spatial_gen_demo():
    print("--- SPATIAL GENERATIVE MULTIMODAL DEMO ---")
    
    # 1. World Model & Planning
    print("\n[1] Testing World Model & Planner...")
    wm = PredictiveWorldModel(state_dim=4, action_dim=2)
    planner = SCPlanner(wm)
    curr_s = np.array([0.1, 0.1, 0.1, 0.1])
    goal_s = np.array([0.9, 0.9, 0.9, 0.9])
    plan = planner.plan_sequence(curr_s, goal_s, horizon=3)
    print(f"    Generated Plan (3 steps): {len(plan)} actions.")
    
    # 2. Spatial Stack
    print("\n[2] Testing Spatial Stack (3D)...")
    voxels = VoxelGrid(resolution=8)
    voxels.set_voxel(4, 4, 4, 1.0) # Center occupied
    st3d = SpatialTransformer3D(resolution=8, dim_k=4)
    processed_voxels = st3d.forward(voxels.data)
    print(f"    Processed Voxel Grid Mean: {np.mean(processed_voxels):.4f}")
    
    # 3. Multimodal Generative
    print("\n[3] Testing Multimodal Generative Outputs...")
    text_gen = SCTextGenerator(vocab=["move", "fire", "detect", "human"])
    print(f"    Generated Text: '{text_gen.generate_sequence(4)}'")
    
    audio_syn = SCAudioSynthesizer()
    waveform = audio_syn.synthesize_tone(frequency=440, duration_ms=100, probability=0.5)
    print(f"    Audio Waveform Size: {len(waveform)} samples.")
    
    gen_3d = SC3DGenerator()
    gen_3d.export_point_cloud_json(np.random.rand(10, 3), np.random.rand(10), "test_points.json")
    
    # 4. Ensemble
    print("\n[4] Testing Agent Ensembles...")
    ens = EnsembleOrchestrator()
    ens.add_agent("Lead", CognitiveOrchestrator())
    ens.add_agent("Support", CognitiveOrchestrator())
    ens.coordinated_mission("Navigation")

if __name__ == "__main__":
    run_spatial_gen_demo()
