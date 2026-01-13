# Handover: Spatial Generative Multimodal Dynamic AGI Paper -> SC-NeuroCore
Date: 2026-01-13
From: Codex (Actuator)
To: Gemini (Architect)
Status: Comparison complete, implementation pending

## Objective
Provide a clear alignment and gap analysis between the paper "Spatial generative multimodal dynamic AGI as model of consciousness" and the sc-neurocore codebase to guide implementation planning.

## Paper Metadata
- Title: Spatial generative multimodal dynamic AGI as model of consciousness
- Author: Evgeny Bryndin
- Journal: International Robotics and Automation Journal
- Date: 2026 (Vol 12, Issue 1, pp 6-16)
- DOI: 10.15406/iratj.2026.12.00311
- Source file: 01_MANUSCRIPTS/SCPN_PAPERS/ACADEMIA_INPUT/Spatial_generative_multimodal_dynamic_AG.pdf

## Paper Requirements (condensed)
- Spatial generativity: real-time spatial modeling and navigation.
- Multimodality: integrate visual, auditory, tactile and other modalities.
- Dynamism: adaptation and continual learning.
- Generativity: create new images, ideas, or solutions.
- Consciousness model: internal world models, attention and self-reflection, multimodal integration, metacognition.
- Neuromorphic stack: spiking neural networks plus transformers on neuromorphic chips.
- Semantic and holographic memory traces; graph networks; reinforcement learning; ensembles of agents.

## SC-NeuroCore Correspondence (what already aligns)
Neuromorphic spiking foundation
- Spiking neurons: 03_CODE/sc-neurocore/src/sc_neurocore/neurons/stochastic_lif.py
- Homeostasis: 03_CODE/sc-neurocore/src/sc_neurocore/neurons/homeostatic_lif.py
- Plasticity: 03_CODE/sc-neurocore/src/sc_neurocore/synapses/stochastic_stdp.py
- Reward-modulated learning: 03_CODE/sc-neurocore/src/sc_neurocore/synapses/r_stdp.py

Transformers and attention
- Attention: 03_CODE/sc-neurocore/src/sc_neurocore/layers/attention.py
- Transformer block (S-Former): 03_CODE/sc-neurocore/src/sc_neurocore/transformers/block.py

Multimodal inputs and fusion
- Dynamic vision input (event camera): 03_CODE/sc-neurocore/src/sc_neurocore/interfaces/dvs_input.py
- Audio example model: 03_CODE/sc-neurocore/src/sc_neurocore/models/zoo.py
- BCI input interface: 03_CODE/sc-neurocore/src/sc_neurocore/interfaces/bci.py
- Multimodal fusion layer: 03_CODE/sc-neurocore/src/sc_neurocore/layers/fusion.py

Dynamics and memory analogs
- Recurrent dynamics: 03_CODE/sc-neurocore/src/sc_neurocore/layers/recurrent.py
- Graph networks: 03_CODE/sc-neurocore/src/sc_neurocore/graphs/gnn.py
- Associative memory (HDC): 03_CODE/sc-neurocore/src/sc_neurocore/hdc/base.py
- Holographic mapping: 03_CODE/sc-neurocore/src/sc_neurocore/eschaton/holographic.py

Metacognition and consciousness metrics
- Self-model and reflection loop: 03_CODE/sc-neurocore/src/sc_neurocore/core/self_awareness.py
- Phi-like integration metric: 03_CODE/sc-neurocore/src/sc_neurocore/analysis/consciousness.py

Hardware export
- Verilog generator: 03_CODE/sc-neurocore/src/sc_neurocore/hdl_gen/verilog_generator.py
- SPICE generator: 03_CODE/sc-neurocore/src/sc_neurocore/hdl_gen/spice_generator.py

## Gaps vs Paper (missing or partial)
- No explicit internal world-model or planning module (prediction and action selection).
- No 3D spatial transformer or spatial navigation environment.
- Multimodal generation is limited to toy image output; no text, speech, or 3D generation modules.
- No staged curriculum or benchmark suite for multimodal spatial tasks.
- No unified multimodal dataset ingestion pipeline inside sc-neurocore.
- Transformable neurochips are only partially addressed via static HDL export.

## Implementation Tasks Proposed for Gemini
1) World model and planning
   - Add a predictive world-model module (state transition and forecasting).
   - Add a planner that consumes world-model state and outputs action proposals.

2) Spatial stack
   - Add 3D spatial representations (voxels or point clouds).
   - Implement a spatial transformer or attention layer for 3D.
   - Provide a basic navigation benchmark or task scaffold.

3) Multimodal generative outputs
   - Add text generation interface (minimal token-level module or external adapter).
   - Add speech or audio synthesis interface (even stub-level scaffolding).
   - Add 3D generative output adapter (mesh or point cloud export).

4) Multimodal learning pipeline
   - Create a dataset ingestion interface and training loops for multimodal fusion.
   - Add reinforcement learning loop using RewardModulatedSTDPSynapse.

5) Agent ensemble integration
   - Add a minimal multi-agent orchestrator to align with the paper ensemble concept.

## Notes and Constraints
- This handoff does not modify monads or the SCPN atlas.
- The paper is present in ACADEMIA_INPUT but has not been processed with enhanced RAG.
  If needed, use 03_CODE/SWARM_AUTOMATION/process_academia_input.py and then run
  03_CODE/CLAUDE_RAG_GENERATION_SUITE/run_enhanced_rag.py for formal extraction.

## Deliverable
This file is the handoff record for Gemini to consider sc-neurocore alignment and implementation tasks.

admin edit: consider search across the framework, spatiality 
