<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# API Reference Index

This page is the maintained entry point for the public API reference. It avoids duplicating generated module listings so stale modules do not remain visible after code moves. New users should start with the Core Runtime section, then move to Compiler/Hardware or Training/Interop only when their workflow needs those surfaces.

## API selection rule

Use the smallest public surface that proves the workflow:

- for base package work, import from `sc_neurocore`, `neurons`, `synapses`,
  `layers`, `sources`, `recorders`, `compiler`, `hdl_gen`, or documented API
  pages below;
- for optional paths, install the matching extra first and keep the dependency
  boundary in the report;
- for source-only research modules, work from a checkout and do not imply that
  the module ships in the base wheel;
- for hardware, benchmarks, or industrial claims, pair the API call with the
  generated manifest, raw benchmark artefact, or evidence bag.

## Full Generated Reference

- [Full package reference](../API_REFERENCE.md) — generated from
  `src/sc_neurocore` docstrings with `scripts/generate_docs.py`.

## Core Runtime

- [Network simulation engine](network.md)
- [Neuron model catalogue](neuron_models.md)
- [Neurons](neurons.md)
- [Synapses](synapses.md)
- [Layers](layers.md)
- [Sources](sources.md)
- [Recorders](recorders.md)
- [Spike train analysis](analysis.md)
- [Exception hierarchy](exceptions.md)
- [Named constants](constants.md)

## Compiler, Export, And Hardware

- [Compiler](compiler.md)
- [Compiler surface policy](compiler_surface.md)
- [Adaptive precision](adaptive_precision.md)
- [Export](export.md)
- [HDL generation](hdl_gen.md)
- [Formal network verification](formal_network_verification.md)
- [Hardware drivers](drivers.md)
- [Chiplet compiler](chiplet.md)
- [Analog bridge](analog_bridge.md)

## Training And Learning

- [Training API](training.md)
- [Learning facade](learning.md)
- [Delay training](delay_training.md)
- [ANN-to-SNN conversion](conversion.md)
- [Autonomous learning](autonomous_learning.md)
- [Model zoo](model_zoo.md)

## Acceleration And Interop

- [Acceleration](accel.md)
- [GPU backend](gpu.md)
- [Lava/Loihi integrations](integrations.md)
- [NIR bridge](nir_bridge.md)
- [ONNX/TVM export notes](export.md)
- [Local LLM bridge](bridges/local_llm.md)

## Domain Modules

- [Neural decoders](neural_decoders.md)
- [Transcriptomic models](transcriptomic.md)
- [Dopamine STDP synapse](dopamine_stdp.md)
- [Short-term plasticity synapse](short_term_plasticity.md)
- [World model](world_model.md)
- [Predictive model](world_model/predictive_model.md)
- [Evolutionary substrate](evo_substrate.md)
- [Fault injection](fault_injection.md)
- [Safety certification](safety_cert.md)
