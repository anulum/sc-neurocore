# DNA Computing Mapper — `sc_neurocore.bridges.dna_mapper`

> **Module Version:** 1.1.0 · **Status:** Production code · **Tier:** Research substrate
> **Author:** Miroslav Šotek · **ORCID:** 0009-0009-3560-0851
> **License:** AGPL-3.0-or-later | Commercial license available
> **Contact:** www.anulum.li | protoscience@anulum.li

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Gate Library](#gate-library)
7. [Sequence Design Engine](#sequence-design-engine)
8. [Kinetic Simulation](#kinetic-simulation)
9. [Error Correction](#error-correction)
10. [Cross-Hybridization Analysis](#cross-hybridization-analysis)
11. [Noise Model & Sensitivity](#noise-model--sensitivity)
12. [Export Formats](#export-formats)
13. [Cost Estimation](#cost-estimation)
14. [Protocol Generation](#protocol-generation)
15. [Benchmarks](#benchmarks)
16. [Integration Guide](#integration-guide)
17. [Design Patterns & Conventions](#design-patterns--conventions)
18. [Known Limitations](#known-limitations)
19. [References](#references)
20. [Change Log](#change-log)

---

## 1. Overview

The `dna_mapper` compatibility façade is the **primary molecular bridge** in
the SC-NeuroCore `bridges` package. Nine acyclic responsibility modules behind
it compile stochastic computing (SC) Boolean networks into DNA strand
displacement circuits, enzymatic gate cascades, and NUPACK-compatible sequence
designs. Historical imports and serialized public object identities remain at
`sc_neurocore.bridges.dna_mapper`.

### Key Capabilities

| Capability | Component | Status |
|-----------|-----------|--------|
| 10 gate types (AND, OR, NOT, MUX, NAND, XOR, THRESHOLD, AMPLIFIER, BUFFER, CATALYTIC) | `StrandDisplacementCompiler`, `EnzymaticGateCompiler` | ✅ |
| Constraint-satisfying sequence design (GC, homopolymer, orthogonality) | `SequenceDesigner` | ✅ |
| Toehold-mediated strand displacement compilation | `StrandDisplacementCompiler` | ✅ |
| Restriction enzyme / ligase gate compilation | `EnzymaticGateCompiler` | ✅ |
| Nearest-neighbour thermodynamics (ΔG°, Tm) | `DNAStrand`, `NUPACKInterface` | ✅ |
| Kinetic simulation (mass-action ODE) | `KineticSimulator` | ✅ |
| Reed–Solomon error correction over GF(4) | `GF4ErrorCorrection` | ✅ |
| Cross-hybridization detection | `CrossHybridizationChecker` | ✅ |
| Monte Carlo noise / sensitivity analysis | `NoiseModel` | ✅ |
| Oligo synthesis cost estimation | `estimate_cost()` | ✅ |
| Wet-lab protocol generation (Markdown) | `generate_protocol()` | ✅ |
| 4 export formats (GenBank, FASTA, NUPACK, JSON) | `export_*()` functions | ✅ |
| NUPACK integration (soft import with fallback) | `NUPACKInterface` | ✅ |
| 316 focused tests; full touched-surface statement/branch coverage | DNA, wheel boundary, linked consumer, architecture, and benchmark cohorts | ✅ |
| Maintained Rust safety mirror | `src/sc_neurocore/accel/rust/safety/dna_mapper.rs` (6 focused tests) | ✅ |

### Metrics

| Metric | Value |
|--------|-------|
| **Source LOC** | 3,500 across 10 modules |
| **Façade / largest responsibility module** | 213 / 573 lines |
| **Owned test LOC** | 2,607 |
| **Benchmark LOC** | 624 |
| **Public types/classes/enums** | 23 |
| **Gate types** | 10 |
| **Focused test count** | 316 |
| **Test pass rate** | 100% |
| **Coverage** | 1,220 statements + 406 branches, 0 misses/partials |
| **Compile throughput** | ~50 gates in 196 ms |
| **Simulation throughput** | 7,200 s in 4.1 ms |

---

## 2. Theoretical Foundation

### 2.1 Stochastic Computing Meets DNA

Stochastic computing encodes real-valued probabilities `p ∈ [0, 1]` as
the frequency of `1`s in a binary bitstream of length `L`. A bitstream
with `k` ones out of `L` bits encodes `p = k/L`.

The key insight is that **standard Boolean gates perform exact
probabilistic arithmetic** on stochastic bitstreams:

| SC Operation | Gate | Formula |
|-------------|------|---------|
| Multiply | AND | `P(C) = P(A) · P(B)` |
| Complement multiply | OR | `P(C) = 1 - (1-P(A))(1-P(B))` |
| Complement | NOT | `P(C) = 1 - P(A)` |
| Scaled addition | MUX | `P(C) = S·P(A) + (1-S)·P(B)` |

### 2.2 DNA Strand Displacement

DNA strand displacement circuits implement the same Boolean operations
using toehold-mediated hybridization:

1. **Toehold binding**: a short single-stranded overhang (6 nt) initiates
   hybridization between an input signal strand and a translator complex.
2. **Branch migration**: the input strand displaces the incumbent strand
   through the recognition domain (15 nt).
3. **Output release**: the displaced strand (output signal) is released
   into solution.

The presence/absence of a DNA strand at concentration > threshold
encodes logical `1`/`0`.

### 2.3 Seesaw Gate Architecture

This module implements the seesaw gate architecture from
Qian & Winfree (2011):

```
Input A ─┐        ┌──── Output
         ├──[AND]──┤
Input B ─┘        └──── Waste
              │
           Fuel ───┘
```

Each gate consists of:

- **Translator complex**: double-stranded DNA with toehold overhangs
- **Threshold strand**: absorbs sub-threshold inputs (prevents leak)
- **Fuel strand**: drives the reaction to completion (irreversible)
- **Output strand**: released when computation succeeds

### 2.4 Nearest-Neighbour Thermodynamics

All thermodynamic calculations use the SantaLucia (1998) unified
nearest-neighbour model:

```
ΔG°₃₇ = ΔG°init + Σᵢ ΔG°ᵢ,ᵢ₊₁
```

where `ΔG°ᵢ,ᵢ₊₁` is the stacking free energy of the dinucleotide
at positions `i` and `i+1`. This determines:

- **Hybridization strength** (ΔG° < 0 = favorable)
- **Melting temperature** (Tm)
- **Leak rate estimation** (Arrhenius model)

---

## 3. Architecture

### 3.1 Responsibility Ownership

| Module | Responsibility | Physical lines |
|---|---|---:|
| `dna_mapper.py` | Stable imports, optional-backend state, compatibility identity | 213 |
| `dna_types.py` | Physical constants, enums, strand/gate/circuit contracts | 342 |
| `dna_sequences.py` | Deterministic constrained sequence generation | 163 |
| `dna_compilers.py` | Strand-displacement and enzymatic gate compilation | 573 |
| `dna_thermodynamics.py` | NUPACK adapter and deterministic fallback thermodynamics | 276 |
| `dna_simulation.py` | Kinetics, noise, concentration, precision, degradation | 491 |
| `dna_bridge.py` | High-level network and adjacency orchestration | 343 |
| `dna_encoding.py` | GF(4) error correction and dual-rail encoding | 213 |
| `dna_analysis.py` | Hybridization, topology, hairpin, and gate analysis | 338 |
| `dna_io.py` | Exports, costing, protocols, visualization, plate layout | 548 |

An architecture contract caps the façade at 250 lines and every responsibility
module at 650 lines. It also parses imports and rejects cycles or a dependency
back into the façade.

### 3.2 Component Dependency Graph

```
                    BitstreamToDNA
                    ╱            ╲
    StrandDisplacement     EnzymaticGate
        Compiler             Compiler
              ╲             ╱
            SequenceDesigner
                   │
               DNAStrand
               DNAGate
            DNACircuitDesign
                   │
         ┌─────────┼──────────┐
         │         │          │
    KineticSim  NUPACKInterface  Exporters
         │                    │
    NoiseModel          GenBank/FASTA/
                        NUPACK/JSON
         │
    GF4ErrorCorrection
    CrossHybridizationChecker
    estimate_cost()
    generate_protocol()
```

### 3.3 Data Flow

```
SC Gate Spec  ──→  SequenceDesigner  ──→  Compiler  ──→  DNACircuitDesign
     │                                                        │
     │                                                   ┌────┼────┐
     │                                                   │    │    │
     │                                              Simulate  Export  Validate
     │                                                   │    │    │
     └───────────────── Verify Logic ◄───────────────────┘    │    │
                                                              │    │
                                                        .gb .fa .json
```

### 3.4 Soft Import Pattern

External dependencies use the project-standard soft import pattern:

```python
try:
    import nupack
    _HAS_NUPACK = True
except ImportError:
    nupack = None
    _HAS_NUPACK = False
```

This ensures the module works without NUPACK installed. The fallback uses
the internal nearest-neighbour thermodynamic model plus deterministic
Watson-Crick secondary-structure dynamic programming and Boltzmann-style
base-pair probability estimates, so validation still detects likely
intramolecular hairpins instead of assuming all strands are unstructured.

The base wheel intentionally excludes the source-only `sc_neurocore.optics`
tree. `sc_neurocore.bridges` therefore publishes photonic co-design exports
only when that tree is present; DNA imports remain available in the base wheel.

### 3.5 Polyglot Authority

Python is the canonical molecular-mapper implementation. The Rust safety
module is the only maintained execution mirror in this repository; it compiles
standalone and its six focused tests pass. Generated Julia and Mojo files were
non-parsing Python-to-string scaffolds, while the generated Go package contained
only empty functions and no tests. Those false surfaces were removed rather
than presented as parity evidence. No canonical dispatcher referenced them.

---

## 4. Quick Start

### 4.1 Compile a Simple AND Gate

```python
from sc_neurocore.bridges import BitstreamToDNA

compiler = BitstreamToDNA(method="displacement", seed=42)
design = compiler.compile_network(
    gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
    input_names=["A", "B"],
    output_names=["C"],
    name="my_and_gate",
)

print(f"Total strands: {design.total_strands}")
print(f"Total nucleotides: {design.total_nucleotides}")
```

### 4.2 Simulate and Verify Logic

```python
from sc_neurocore.bridges import KineticSimulator

sim = KineticSimulator()

# AND(1, 1) → 1
result = sim.simulate(design, {"A": 200.0, "B": 200.0}, duration_s=3600.0)
print(f"Output: {result['C'][-1]:.1f} nM")  # High

# AND(1, 0) → 0
result = sim.simulate(design, {"A": 200.0, "B": 0.0}, duration_s=3600.0)
print(f"Output: {result['C'][-1]:.1f} nM")  # Low
```

### 4.3 MUX Gate (SC Weighted Addition)

```python
design = compiler.compile_network(
    gates=[{"type": "MUX", "inputs": ["S", "A", "B"], "output": "Y"}],
    input_names=["S", "A", "B"],
    output_names=["Y"],
    name="sc_mux",
)
```

### 4.4 Export for Wet Lab

```python
from sc_neurocore.bridges import (
    export_genbank, export_fasta, export_json, generate_protocol
)

export_genbank(design, "circuit.gb")
export_fasta(design, "oligos.fasta")
export_json(design, "design.json")

protocol = generate_protocol(design)
with open("protocol.md", "w") as f:
    f.write(protocol)
```

### 4.5 Cost Estimation

```python
from sc_neurocore.bridges import estimate_cost

cost = estimate_cost(design, purification="hplc")
print(f"Total cost: ${cost['total_cost_usd']:.2f}")
print(f"Unique oligos: {cost['n_unique_oligos']}")
```

---

## 5. API Reference

### 5.1 Data Classes

#### `GateType(Enum)`

Supported gate types:

| Value | Description | Compilation |
|-------|------------|-------------|
| `AND` | 2-input logical AND | Displacement |
| `OR` | 2-input logical OR | Displacement |
| `NOT` | 1-input logical NOT (inverter) | Displacement |
| `MUX` | 2:1 multiplexer (3 inputs: select, a, b) | Displacement |
| `NAND` | 2-input NOT-AND | Enzymatic |
| `XOR` | 2-input exclusive OR | Enzymatic |
| `THRESHOLD` | Concentration-dependent activation | Displacement |
| `AMPLIFIER` | Catalytic signal amplification | Displacement |
| `BUFFER` | Signal restoration buffer | Displacement |
| `CATALYTIC` | Generic catalytic cycle | Reserved |

#### `CompilationMethod(Enum)`

| Value | Description |
|-------|------------|
| `DISPLACEMENT` | Toehold-mediated strand displacement |
| `ENZYMATIC` | Restriction enzyme + ligase logic |
| `HYBRID` | Mixed (reserved for future) |

#### `DNAStrand`

Frozen dataclass representing a single-stranded DNA molecule.

| Property | Type | Description |
|----------|------|------------|
| `name` | `str` | Unique identifier |
| `sequence` | `str` | 5'→3' nucleotide sequence |
| `role` | `str` | `"signal"`, `"fuel"`, `"output"`, `"waste"`, `"toehold"`, `"translator"` |
| `concentration_nM` | `float` | Initial concentration in nanomolar |
| `length` | `int` | Sequence length (computed) |
| `gc_content` | `float` | GC fraction [0, 1] (computed) |
| `complement` | `str` | Watson-Crick complement, 3'→5' (computed) |
| `max_homopolymer_run` | `int` | Longest run of identical bases (computed) |
| `delta_g_37()` | `float` | ΔG° at 37°C in kcal/mol |
| `melting_temperature()` | `float` | Tm in °C |

#### `DNAGate`

Mutable dataclass representing a compiled DNA logic gate.

| Field | Type | Description |
|-------|------|------------|
| `gate_id` | `int` | Unique gate index |
| `gate_type` | `GateType` | Logic operation |
| `input_names` | `list[str]` | Input signal names |
| `output_name` | `str` | Output signal name |
| `strands` | `list[DNAStrand]` | All participating DNA strands |
| `threshold` | `float` | Activation threshold (threshold gates) |
| `leak_rate` | `float` | Estimated spurious activation rate (s⁻¹) |
| `strand_count` | `int` | Number of strands (computed) |

#### `DNACircuitDesign`

Complete compiled DNA circuit.

| Field | Type | Description |
|-------|------|------------|
| `name` | `str` | Circuit identifier |
| `gates` | `list[DNAGate]` | Compiled gates |
| `input_strands` | `list[DNAStrand]` | Primary input signals |
| `output_strands` | `list[DNAStrand]` | Primary output signals |
| `fuel_strands` | `list[DNAStrand]` | Fuel/helper strands |
| `method` | `CompilationMethod` | Compilation target |
| `temperature_c` | `float` | Design temperature (°C) |
| `na_concentration_M` | `float` | NaCl concentration (M) |
| `total_strands` | `int` | Total strand count (computed) |
| `total_gates` | `int` | Gate count (computed) |
| `total_nucleotides` | `int` | Total nt count (computed) |
| `validate()` | `list[str]` | Design rule check warnings |

---

## 6. Gate Library

### 6.1 Displacement Gates

#### AND Gate

**Architecture:** Two-layer seesaw cascade.
Both signal A and signal B must be present to displace the output
strand from a dual-input translator complex.

**Strands (5):**

| Strand | Role | Concentration |
|--------|------|--------------|
| `translator_top` | Translator | 200 nM |
| `translator_bot` | Translator | 200 nM |
| `output` | Output | 0 nM (released) |
| `fuel` | Fuel | 500 nM |
| `threshold` | Threshold | 50 nM |

**Truth table verified by kinetic simulation:**

| A | B | Output |
|---|---|--------|
| 0 | 0 | Low |
| 1 | 0 | Low |
| 0 | 1 | Low |
| 1 | 1 | **High** |

#### OR Gate

**Architecture:** Catalytic hairpin assembly.
Either input signal triggers hairpin opening and output release.

#### NOT Gate

**Architecture:** Strand sequestration.
Input signal sequesters the blocking strand, releasing the pre-loaded
output.

#### MUX Gate (2:1 Multiplexer)

**Architecture:** Dual-threshold cascade.
The select signal activates one of two pathways, implementing:

```
P(out) = S · P(A) + (1 - S) · P(B)
```

This is the **core SC operation** for weighted addition and is critical
for any meaningful SC circuit.

**Strands (5):**

| Strand | Role | Concentration |
|--------|------|--------------|
| `path_a` | Translator | 200 nM |
| `path_b` | Translator | 200 nM |
| `combiner` | Translator | 200 nM |
| `output` | Output | 0 nM |
| `fuel` | Fuel | 500 nM |

#### AMPLIFIER Gate

**Architecture:** Fuel-driven catalytic cycle.
One input molecule catalytically releases many output molecules.
Useful for fan-out from a single signal to multiple downstream
consumers.

#### BUFFER Gate

**Architecture:** Threshold + amplifier combination.
Passes the signal through with level restoration, cleaning up
degraded signals in long cascades (>5 gates deep).

#### THRESHOLD Gate

**Architecture:** Concentration-dependent activation.
The threshold strand absorbs input below a specified concentration;
only excess input activates the output.

### 6.2 Enzymatic Gates

#### NAND Gate

**Architecture:** Dual restriction enzyme cascade.
Both inputs (as enzymes) must be present to cleave the substrate
at two sites (EcoRI + BamHI). Only when both sites are cleaved is
the output fragment NOT produced.

#### XOR Gate

**Architecture:** Dual-substrate selective cleavage.
Each input exclusively cleaves one substrate arm; overlapping
cuts cancel the output.

---

## 7. Sequence Design Engine

### 7.1 SequenceDesigner

The `SequenceDesigner` generates DNA sequences satisfying three
simultaneous constraints:

1. **GC Content**: 40–60% (configurable)
2. **Homopolymer avoidance**: ≤3 consecutive identical bases
3. **Orthogonality**: low similarity between sequences in the same design

**Algorithm:** Rejection sampling with guided nucleotide selection.
Each position is chosen probabilistically with weights biased toward
maintaining GC content, while disallowing nucleotides that would
create excessive homopolymer runs.

**Determinism:** Fully deterministic given a seed. The same seed +
name combination always produces the same sequence.

```python
designer = SequenceDesigner(seed=42)
seq = designer.generate(20, "my_sequence")
comp = designer.generate_complement(seq)
toehold = designer.generate_toehold("gate_0")  # 6 nt
recog = designer.generate_recognition("gate_0")  # 15 nt
```

### 7.2 Design Constants

| Constant | Value | Source |
|----------|-------|--------|
| `_TOEHOLD_LENGTH` | 6 nt | Zhang & Winfree (2009) |
| `_RECOGNITION_LENGTH` | 15 nt | Qian & Winfree (2011) |
| `_CLAMP_LENGTH` | 5 nt | Standard practice |
| `_GC_TARGET_LOW` | 0.40 | SantaLucia (1998) |
| `_GC_TARGET_HIGH` | 0.60 | SantaLucia (1998) |
| `_MAX_HOMOPOLYMER` | 3 | IDT synthesis guideline |

### 7.3 Thermodynamic Model

The nearest-neighbour (NN) parameters from SantaLucia (1998) Table 2
are used for all ΔG° and Tm calculations:

| Dinucleotide | ΔG° (kcal/mol) |
|-------------|----------------|
| AA/TT | −1.00 |
| AT | −0.88 |
| TA | −0.58 |
| CA/TG | −1.45 |
| GT/AC | −1.44 |
| CT/AG | −1.28 |
| GA/TC | −1.30 |
| CG | −2.17 |
| GC | −2.24 |
| GG/CC | −1.84 |

Initiation penalty: +1.96 kcal/mol.

---

## 8. Kinetic Simulation

### 8.1 KineticSimulator

The `KineticSimulator` uses Euler-method integration of mass-action
ODE kinetics to predict strand concentration time courses.

**Rate model per gate type:**

| Gate | Rate Equation |
|------|---------------|
| AND | `d[out]/dt = k_fwd · [A] · [B] − k_rev · [out] − k_leak` |
| OR | `d[out]/dt = k_fwd · max([A], [B]) − k_rev · [out]` |
| NOT | `d[out]/dt = k_fwd · (C_max − [A]) − k_rev · [out]` |
| THRESHOLD | `d[out]/dt = k_fwd · max(0, [A] − θ) − k_rev · [out]` |

**Parameters:**

| Parameter | Default | Unit |
|-----------|---------|------|
| `k_fwd` | 1×10⁻³ | nM⁻¹·s⁻¹ |
| `k_rev` | 1×10⁻⁵ | s⁻¹ |
| `dt` | 1.0 | s |
| `duration_s` | 3600.0 | s |

### 8.2 Usage

```python
sim = KineticSimulator()
result = sim.simulate(
    design,
    {"A": 200.0, "B": 200.0},
    duration_s=3600.0,
    dt=0.5,
)
# result["time"] → np.ndarray of time points
# result["C"] → np.ndarray of output concentrations
```

### 8.3 Convergence

The simulation uses explicit Euler integration with concentration
clamping to [0, max_input_concentration]. For typical circuits,
convergence is reached within 30 minutes of simulated time.

---

## 9. Error Correction

### 9.1 GF4ErrorCorrection

Reed–Solomon-like error correction over the Galois field GF(4),
where each nucleotide maps to a GF(4) element:

| Nucleotide | GF(4) |
|-----------|-------|
| A | 0 |
| C | 1 |
| G | 2 |
| T | 3 |

**Block structure:** each block of 12 data nucleotides is followed
by 4 parity nucleotides (configurable).

```python
ec = GF4ErrorCorrection(n_parity=4, block_size=12)

encoded = ec.encode("ACGTACGTACGT")
# "ACGTACGTACGT" + 4 parity nucleotides

decoded, corrections = ec.decode(encoded)
# corrections == 0 if no errors
```

**Performance:** 576 µs to encode 1,200 nt.

---

## 10. Cross-Hybridization Analysis

### 10.1 CrossHybridizationChecker

Detects **unwanted cross-hybridization** between circuit strands by
computing pairwise longest-common-substring alignments between each
strand and the complement of every other strand.

```python
checker = CrossHybridizationChecker(max_complementary_run=8)
flags = checker.check(design)
for flag in flags:
    print(f"{flag['strand_a']} ↔ {flag['strand_b']}: "
          f"{flag['complementary_run']} bp ({flag['severity']})")
```

**Severity levels:**

| Run length | Severity |
|-----------|----------|
| ≥ 8 bp | Medium |
| ≥ 12 bp | High |

---

## 11. Noise Model & Sensitivity

### 11.1 NoiseModel

Monte Carlo sensitivity analysis that perturbs strand concentrations
and assesses output robustness.

```python
nm = NoiseModel(
    concentration_cv=0.05,  # 5% pipetting noise
    temperature_std_c=0.5,  # ±0.5°C uncertainty
    n_trials=50,
    seed=42,
)
report = nm.sensitivity_analysis(
    design,
    {"A": 200.0, "B": 200.0},
    duration_s=3600.0,
)
for key, stats in report["outputs"].items():
    print(f"{key}: mean={stats['mean']:.1f} ± {stats['std']:.1f} nM "
          f"(CV={stats['cv']:.3f}, robust={stats['robust']})")
```

**Report fields per output:**

| Field | Description |
|-------|------------|
| `mean` | Mean final concentration across trials |
| `std` | Standard deviation |
| `cv` | Coefficient of variation (std/mean) |
| `min`, `max` | Range |
| `robust` | `True` if CV < 0.15 |

---

## 12. Export Formats

### 12.1 GenBank (`.gb`)

Standard GenBank flat file format with `LOCUS`, `FEATURES`, and
`ORIGIN` sections. Compatible with SnapGene, Benchling, and APE.

```python
export_genbank(design, "circuit.gb")
```

### 12.2 FASTA (`.fasta`)

Multi-entry FASTA with one header per strand.

```python
export_fasta(design, "oligos.fasta")
```

### 12.3 NUPACK Input (`.nupack`)

Ready-to-run NUPACK input file for multi-strand thermodynamic
analysis.

```python
export_nupack_input(design, "analysis.nupack")
```

### 12.4 JSON (`.json`)

Complete machine-readable representation of the circuit, including
all strand sequences, thermodynamic properties, and gate topology.

```python
export_json(design, "design.json")
```

**JSON schema (simplified):**

```json
{
  "name": "my_circuit",
  "method": "displacement",
  "total_gates": 1,
  "total_strands": 8,
  "total_nucleotides": 185,
  "gates": [{
    "gate_type": "and",
    "input_names": ["A", "B"],
    "output_name": "C",
    "strands": [{
      "name": "g0_translator_top",
      "sequence": "ACGTAC...",
      "gc_content": 0.48,
      "delta_g_37": -15.2,
      "length": 42
    }]
  }]
}
```

---

## 13. Cost Estimation

### 13.1 `estimate_cost()`

Estimates oligo synthesis cost based on per-base pricing and
purification method.

```python
cost = estimate_cost(
    design,
    price_per_base_usd=0.10,
    fixed_per_oligo_usd=5.00,
    purification="hplc",  # "standard" | "hplc" | "page"
)
```

**Purification multipliers:**

| Method | Multiplier | Typical use |
|--------|-----------|-------------|
| Standard (desalt) | 1.0× | Screening |
| HPLC | 2.5× | Functional assays |
| PAGE | 3.0× | Precision experiments |

**Benchmark:** 13 µs per cost estimation.

---

## 14. Protocol Generation

### 14.1 `generate_protocol()`

Generates a Markdown-formatted wet-lab protocol with:

- Materials table (oligo name, length, stock concentration, volume)
- Step-by-step procedure (strand addition order, annealing, incubation)
- Expected results per gate

```python
protocol = generate_protocol(
    design,
    volume_uL=50.0,
    buffer_name="1× TAE/Mg²⁺",
)
```

**Benchmark:** 18 µs per protocol generation.

---

## 15. Benchmarks

### 15.1 Compilation Throughput

| Circuit size | Time | Strands | Nucleotides |
|-------------|------|---------|-------------|
| 1 gate | 4.5 ms | 8 | 185 |
| 5 gates | 14.9 ms | 27 | 690 |
| 10 gates | 23.4 ms | 50 | 1,265 |
| 25 gates | 84.3 ms | 120 | 3,065 |
| 50 gates | 196.0 ms | 237 | 6,090 |

### 15.2 Simulation Throughput

| Duration | Time |
|----------|------|
| 100 s | 0.1 ms |
| 1,000 s | 0.6 ms |
| 3,600 s | 2.1 ms |
| 7,200 s | 4.1 ms |

### 15.3 Error Correction

| Sequence length | Encode | Decode |
|----------------|--------|--------|
| 48 nt | 24 µs | 24 µs |
| 240 nt | 109 µs | 120 µs |
| 1,200 nt | 576 µs | 901 µs |

### 15.4 Cross-Hybridization Analysis

| Circuit | Strands | Time | Flags |
|---------|---------|------|-------|
| 5 gates | 17 | 4.6 ms | 8 |
| 10 gates | 32 | 17.4 ms | 13 |
| 25 gates | 77 | 113.5 ms | 40 |

### 15.5 Running Benchmarks

```bash
cd SC-NEUROCORE
python benchmarks/dna_mapper_benchmark.py
# Results → benchmarks/results/dna_mapper_benchmark.json
```

The architecture regression harness compares the flat parent against the
modular working tree using 30 interleaved subprocess samples after five
warmups, with ten repeated one-gate compilations per sample:

```bash
python benchmarks/bench_dna_mapper_import.py \
  --baseline-root <parent-root> \
  --candidate-root <candidate-root> \
  --output benchmarks/results/local_python_2026-07-12_dna_mapper_import.json
```

| Metric | Flat parent median | Modular median | Delta |
|---|---:|---:|---:|
| Cold import | 334.571 ms | 362.184 ms | +8.25% |
| One-gate compile | 9.689 ms | 10.082 ms | +4.06% |
| Process wall | 617.283 ms | 660.115 ms | +6.94% |
| Peak RSS | 39,948 KiB | 39,916 KiB | -0.08% |

This is local regression evidence, not a publishable throughput claim. The
capture used taskset affinity but no exclusive core, the powersave governor,
and a workstation whose one-minute load rose from 10.56 to 15.00 during the
capture. The committed JSON preserves raw samples, source hashes, affinity,
frequency, governor, and host-load limitations. A reserved-core rerun is
required before treating the positive timing deltas as a stable regression.

---

## 16. Integration Guide

### 16.1 Package Import

```python
# Direct import
from sc_neurocore.bridges import BitstreamToDNA, KineticSimulator

# Full namespace
from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    StrandDisplacementCompiler,
    EnzymaticGateCompiler,
    KineticSimulator,
    NUPACKInterface,
    GF4ErrorCorrection,
    CrossHybridizationChecker,
    NoiseModel,
    estimate_cost,
    generate_protocol,
    export_genbank,
    export_fasta,
    export_json,
    export_nupack_input,
)
```

### 16.2 End-to-End Workflow

```python
# 1. Compile
compiler = BitstreamToDNA(method="displacement", seed=42)
design = compiler.compile_network(
    gates=[
        {"type": "AND", "inputs": ["A", "B"], "output": "X"},
        {"type": "BUFFER", "inputs": ["X"], "output": "Y"},
        {"type": "NOT", "inputs": ["Y"], "output": "Z"},
    ],
    input_names=["A", "B"],
    output_names=["Z"],
)

# 2. Validate
warnings = design.validate()
xhyb = CrossHybridizationChecker().check(design)

# 3. Simulate
sim = KineticSimulator()
result = sim.simulate(design, {"A": 200.0, "B": 200.0})

# 4. Robustness analysis
noise = NoiseModel(n_trials=50)
report = noise.sensitivity_analysis(design, {"A": 200.0, "B": 200.0})

# 5. Cost
cost = estimate_cost(design, purification="hplc")

# 6. Export
export_genbank(design, "circuit.gb")
export_json(design, "design.json")
protocol = generate_protocol(design)
```

---

## 17. Design Patterns & Conventions

### 17.1 Anti-Slop Code Policy

Per project policy, no narration comments, no buzzword naming, no
AI tells. All code is written in a professional, terse, scientific
style.

### 17.2 Soft Import Pattern

All external dependencies (NUPACK, gdstk, D-Wave Ocean) use try/except
with graceful fallback:

```python
try:
    import nupack
    _HAS_NUPACK = True
except ImportError:
    nupack = None
    _HAS_NUPACK = False
```

### 17.3 Deterministic Design

All sequence generation is deterministic given a seed. This enables:

- Reproducible circuit designs across runs
- Consistent test results
- Diff-friendly design iteration

### 17.4 Frozen Dataclasses

`DNAStrand` is frozen (immutable) to prevent accidental mutation of
sequence data after compilation. `DNAGate` and `DNACircuitDesign` are
mutable to allow post-compilation modification (e.g., adding fuel
strands).

---

## 18. Known Limitations

| Limitation | Impact | Planned Fix |
|-----------|--------|-------------|
| No temperature-dependent rate constants in kinetics | Simulations are temperature-invariant | Arrhenius scaling (v1.1) |
| No salt concentration correction for Tm | Tm estimates assume 1M NaCl | SantaLucia salt correction (v1.1) |
| No mismatched base-pair model | Perfect-match only | Wobble pair penalties (v1.2) |
| No enzyme kinetics (Michaelis-Menten) | Enzymatic gates use simplified kinetics | Km/kcat parameters (v1.2) |
| No feedback loops | Can't compile recurrent SC networks | Cycle detection (v2.0) |
| No NUPACK without manual installation | Fallback uses internal NN + Watson-Crick hairpin estimates | Docker image with NUPACK for full ensemble thermodynamics |
| Euler integrator (O(dt)) | Low accuracy for stiff systems | RK4/scipy.integrate (v1.1) |

---

## 19. References

1. **Zhang, D.Y. & Winfree, E.** (2009). Control of DNA strand
   displacement kinetics using toehold exchange. *JACS*, 131(47),
   17303–17314. DOI: 10.1021/ja906987s

2. **Qian, L. & Winfree, E.** (2011). Scaling up digital circuit
   computation with DNA strand displacement cascades. *Science*,
   332(6034), 1196–1201. DOI: 10.1126/science.1200520

3. **Seelig, G. et al.** (2006). Enzyme-free nucleic acid logic
   circuits. *Science*, 314(5805), 1585–1588.
   DOI: 10.1126/science.1132493

4. **Soloveichik, D. et al.** (2010). DNA as a universal substrate
   for chemical kinetics. *PNAS*, 107(12), 5393–5398.
   DOI: 10.1073/pnas.0909380107

5. **SantaLucia, J.** (1998). A unified view of polymer, dumbbell,
   and oligonucleotide DNA nearest-neighbor thermodynamics.
   *PNAS*, 95(4), 1460–1465. DOI: 10.1073/pnas.95.4.1460

6. **Alaghi, A. & Hayes, J.P.** (2013). Survey of stochastic
   computing. *ACM TECS*, 12(2s), 1–19. DOI: 10.1145/2465787.2465794

---

## 20. Change Log

### v1.1.0 (2026-07-12)

- Replaced the 3,257-line GodFile with a 213-line compatibility façade and nine
  bounded, acyclic responsibility modules.
- Preserved documented imports, reloadable optional-backend state, and pickle
  module identity; detached-parent deterministic outputs remain equivalent.
- Reached 100% statement and branch coverage over all mapper modules and made
  missing kinetic output traces fail closed.
- Removed false Julia, Mojo, and Go acceleration scaffolds; retained and
  independently verified the maintained Rust safety mirror.
- Added hash-bound parent/candidate import and golden-path benchmark evidence.
- Made source-only photonic co-design exports conditional so the documented DNA
  import path works from the clean base wheel without the excluded optics tree.

### v1.0.0 (2026-04-16)

- Initial production release
- 14 classes, 10 gate types, 4 export formats
- 73 tests (100% pass), 5 benchmark suites
- GF(4) error correction, cross-hybridization analysis
- Monte Carlo noise model, cost estimation, protocol generation
- Full pipeline wiring via `bridges/__init__.py`
