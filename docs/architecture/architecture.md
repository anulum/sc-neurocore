# Architecture

## Package Structure

```
sc_neurocore/
├── Core Tier (Production-Ready)
│   ├── neurons/       113 neuron models (LIF, Izhikevich, dendritic, homeostatic, biophysical, …)
│   ├── synapses/      4 synapse types (bitstream, dot product, STDP, R-STDP)
│   ├── layers/        8 layer types (dense, conv, learning, vectorized, recurrent, ...)
│   ├── sources/       Bitstream current sources
│   ├── recorders/     Spike recording
│   ├── utils/         RNG, bitstreams, connectomes, decorrelators, fault injection
│   └── accel/         GPU backend (CuPy), JIT kernels (Numba), MPI, vector ops
│
├── Research Tier (Experimental)
│   ├── analysis/      Phi evaluation, qualia testing, Kardashev metrics
│   ├── bio/           DNA storage, gene regulatory networks, neuromodulation
│   ├── core/          Digital souls, orchestration, replication, self-awareness
│   ├── dashboard/     Terminal-based monitoring
│   ├── ensembles/     Multi-agent consensus
│   ├── export/        ONNX export
│   ├── generative/    Audio, text, and 3D generation
│   ├── hdl_gen/       Verilog and SPICE generation
│   ├── interfaces/    BCI, DVS, interstellar, real-world sensors
│   ├── learning/      Federated, lifelong, neuroevolution
│   ├── pipeline/      Data ingestion and training loops
│   ├── scpn/          7-layer SCPN consciousness model
│   ├── security/      Ethics (Asimov), immune system, watermarks, ZKP
│   ├── verification/  Formal proofs, safety verification
│   └── viz/           Web visualization, generative neuro-art
│
└── Contrib Tier (Speculative/Theoretical)
    ├── eschaton/      Heat death, holographic principle, nested universes
    ├── exotic/        Anyons, chemical RD, Dyson grids, mycelium, matrioshka
    ├── meta/          DAO governance, omega point, singularity, time crystals
    ├── post_silicon/  Claytronics, femtotech, reversible computing
    └── transcendent/  Multiverse, noetic fields, spacetime, vacuum decay
```

## Data Flow

```
Input Values (float)
    │
    ▼
BitstreamEncoder → Bitstream (uint8 array of 0/1)
    │
    ▼
SCDenseLayer / VectorizedSCLayer
    │  (packed 64-bit AND + popcount)
    ▼
Output Firing Rates (float)
```

## Hardware Path

```
Python Model ──► VerilogGenerator ──► Verilog RTL ──► FPGA Synthesis
                                         │
                                         ▼
                                   Co-Simulation
                                   (bit-true verification)
```

## GPU Acceleration

The `accel` package provides transparent GPU acceleration:

- `HAS_CUPY` flag detects CuPy availability
- `xp` module alias (CuPy or NumPy)
- `to_device()` / `to_host()` for data transfer
- Numba JIT kernels for CPU hot paths
