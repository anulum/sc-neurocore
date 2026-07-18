# Neuron Model Reference — 158 Python Classes / 175 Rust PyO3 Wrappers

SC-NeuroCore currently exposes 158 lazy-loaded Python model classes across
153 Python model source modules in `src/sc_neurocore/neurons/models/`, plus
175 Rust PyO3 model wrappers in the optional engine. Matching model classes
use the same `step()` / `reset()` / `get_state()` interface shape where the
backend implements that surface.

> **Fidelity status:** every model here has a faithful, tested Python reference.
> A stricter bar — a model whose whole four-language acceleration chain
> (`accel/{rust,go,julia,mojo}`) is real and Python-parity-proven — is tracked
> separately on the [Model Fidelity & Polyglot Status](model_fidelity_status.md)
> page. Polyglot-complete today (40 models): Wang-Buzsaki, FitzHugh-Nagumo, Morris-Lecar,
> Connor-Stevens, Hodgkin-Huxley, AdEx, ExpIF, Lapicque, Perfect Integrator,
> Quadratic IF, Theta, DPI, COBA LIF, Escape Rate, Poisson, IQIF, McCulloch-Pitts, McKean, Hindmarsh-Rose, FitzHugh-Rinzel,
> Pernarowski, Terman-Wang,
> Wilson-HR, Rulkov map, GLIF, Mihalas-Niebur, Medvedev map, Cazelles map, Chialvo map, Courbage-Nekorkin map,
> Izhikevich 2007, Ibarz-Tanaka map, Ermentrout-Kopell, Sigmoid Rate,
> Threshold Linear Rate, Wilson-Cowan, Wong-Wang, Jansen-Rit,
> Montbrió-Pazó-Roxin (`ErmentroutKopellPopulation`), and Resonate-and-Fire.
> All other models are Python-faithful with their
> acceleration chain still under remediation.

## Quick Start

```python
# Python backend (default)
from sc_neurocore.neurons.models import HodgkinHuxleyNeuron
hh = HodgkinHuxleyNeuron()
spike = hh.step(current=10.0)

# Rust backend (faster, identical interface)
from sc_neurocore_engine.sc_neurocore_engine import HodgkinHuxleyNeuron
hh_rs = HodgkinHuxleyNeuron()
spike = hh_rs.step(current=10.0)
```

## Backend Selection

| Backend | Import path | Use case |
|---------|------------|----------|
| Python | `sc_neurocore.neurons.models` | Prototyping, parameter tuning, debugging |
| Rust | `sc_neurocore_engine.sc_neurocore_engine` | Production, benchmarks, batch simulation |

Backends use identical class names where parity wrappers exist (for example,
`HodgkinHuxleyNeuron`). The Rust engine provides 175 Rust PyO3 model wrappers,
162 of which are wired into the NetworkRunner pipeline.

The package-level `sc_neurocore.neurons` facade remains lazy: core neuron
symbols are available immediately, while model classes are resolved on first
attribute access. Setting `SC_NEUROCORE_NO_RUST=1` forces the pure-Python model
registry even when the optional Rust engine is installed, and the fallback path
caches the resolved Python class for later imports.

## Rust Binding Coverage Map

`tests/test_rust_python_neuron_parity.py` is the live coverage map for the
Python registry exposed by `sc_neurocore.neurons.models.__all__`. The generated
capability inventory above counts 158 static classes in
`src/sc_neurocore/neurons/models/*.py`; the registry-level map covers
160 public Python registry names because `HybridFisherPosnerLIFNeuron` and
`StochasticLIFNeuron` are re-exported (from
`sc_neurocore.quantum_cognition.fisher_posner` for population dispatch and
`sc_neurocore.neurons.stochastic_lif` for the torch-free public root import,
respectively) rather than defined as static classes under `neurons/models/`.

Current binding disposition:

The current registry map records 146 same-name Rust constructors,
9 Rust-prefixed or core-only constructors, and 5 Python-only registry names.

| Disposition | Count | Contract |
|-------------|------:|----------|
| Same-name Rust constructors | 146 | Python registry name matches the compiled `sc_neurocore_engine.sc_neurocore_engine` class name. |
| Rust-prefixed or core-only constructors | 9 | A Rust binding exists, but the generic scalar parity harness uses an explicit name map. |
| Python-only registry names | 5 | No same-name PyO3 neuron constructor is claimed; each entry below records the durable boundary. |

Rust-prefixed or core-only constructor map:

| Python registry name | Rust constructor |
|----------------------|------------------|
| `AdaptiveThresholdMoENeuron` | `RustAdaptiveThresholdMoENeuron` |
| `AstrocyteLIFNeuron` | `RustAstrocyteLIFNeuron` |
| `CochlearHairCell` | `RustCochlearHairCell` |
| `ContinuousAttractorNeuron` | `RustContinuousAttractorNeuron` |
| `DendriticNMDANeuron` | `RustDendriticNMDANeuron` |
| `DirectionSelectiveRGC` | `RustDirectionSelectiveRGC` |
| `HybridLinearAttentionNeuron` | `RustHybridLinearAttentionNeuron` |
| `MulticompartmentMCNNeuron` | `RustMulticompartmentMCNNeuron` |
| `QuantumInspiredLIFNeuron` | `RustQuantumInspiredLIFNeuron` |

Python-only boundary rationale:

| Python registry name | Boundary |
|----------------------|----------|
| `AstrocyteNeuron` | Population adapter over `AstrocyteModel`; the Rust engine already exposes `AstrocyteModel` and `RustAstrocyteLIFNeuron`, but this adapter threshold-converts cytosolic Ca²⁺ release into the population spike interface. |
| `ChayKeizerMinimalNeuron` | The reduced three-state pancreatic beta-cell model is a separate published reduction from the existing five-state Chay-Keizer Rust kernel and needs its own parameter-faithful implementation before a PyO3 constructor is claimed. |
| `HybridFisherPosnerLIFNeuron` | The population-compatible entry depends on the Python `SpinPoolMPS` quantum-metabolic state and shared-pool measurement side effects, so a same-name Rust constructor would overclaim parity until that state model is ported. |
| `Izhikevich2007Neuron` | This model already has a function-level compiled accelerator for the RK4 `simulate()` path (`py_izhikevich2007_simulate`), but no stateful PyO3 neuron constructor is claimed for the full Python object. |
| `SRM0Neuron` | The maintained Python model is an exact-flow SRM0 membrane accumulator; it is not equivalent to the older Rust `SpikeResponseNeuron` kernel, so the registry name stays Python-only until a faithful Rust SRM0 kernel lands. |

The parity test checks this map against the Python registry, committed Rust
PyO3 source declarations, and the built Rust engine when the optional engine is
installed. Stochastic models without a shared RNG remain binding-covered but
outside exact spike-train parity. `EscapeRateNeuron` and `PoissonNeuron` are the
enrolled exceptions: their Python, Rust, Julia, Go, Mojo, schema, and RTL
surfaces share one explicit seeded LFSR16 event-stream contract.
`IntegerQIFNeuron` is the deterministic integer counterpart: every maintained
runtime and Q32.0 RTL form reproduces the pinned Wu et al. source trajectory
bit-for-bit. `McCullochPittsNeuron` preserves the original 1943 logical rule:
a fixed positive active-excitatory-afferent count threshold, absolute veto by
any active inhibitory afferent, and no internal cell state; signed Q32.0 uses
`-1` only as the hardware inhibition sentinel. `ChayKeizerNeuron`
remains an expected parity divergence because the Python model is the
five-dimensional Chay-Keizer burster while the current Rust kernel is the older
reduced form.

## Reference Trace Validation

The schema-driven validation harness in `sc_neurocore.neurons.reference_traces`
loads committed JSON corpus entries, executes the `UniversalNeuron` runner, and
compares scalar trace features with explicit per-feature tolerances. The corpus
contains analytic, independently integrated, and map-iteration references tied
to each entry's stated source. For example, the Ermentrout-Kopell hand-class
enrolment independently advances the sourced theta flow with the maintained
Euler, event, and circular-wrap conventions over 2,000 steps. A corpus entry is
model-specific evidence, not a claim that every class below has external
simulator parity. The Wong-Wang enrolment separately re-derives all four
Euler/Ornstein-Uhlenbeck states and both pre-update rates from the 2006 Appendix
and pins the paper-versus-author-code timestep discrepancy.
The Jansen–Rit enrolment independently advances all six equation-(6) states,
pins the published `C1/C2/C3/C4` connectivity placement and Brian2 source
commit, and records that the maintained 0.1 ms Euler step is implementation
scope rather than a solver prescribed by the continuous paper equations.
The `ErmentroutKopellPopulation` enrolment corrects that legacy public name to
Montbrió, Pazó, and Roxin (2015), restores the printed dimensionless
equation-(12) variables through `R=tau*r` and `t'=t/tau`, independently
advances both states, and records that simultaneous explicit Euler is
maintained implementation scope rather than a solver prescribed by the
continuous paper.
The `ResonateAndFireNeuron` enrolment independently evaluates the exact
constant-input flow of Izhikevich's (2001) complex resonator, preserves `x` as
current-like and `y` as voltage-like, thresholds only an upward sampled
crossing of `y`, and checks the generalised source reset `z=i*threshold` rather
than a radius test or origin reset.

## Model Catalogue

The descriptor corpus lives under
`src/sc_neurocore/neurons/model_descriptors/<ClassName>.toml`. Public catalogue
helpers accept only public Python class identifiers such as `AdExNeuron`; dotted,
empty, private, and path-like names fail before filesystem access. Studio uses
the same `load_descriptor()` surface when it renders model detail pages, so the
descriptor guard protects the browser-facing catalogue and maintenance tools.

`tools/generate_model_descriptors.py` refreshes the corpus through
`generate_descriptor_payload()` and `merge_descriptor_payloads()`. The generator
uses the same public-class-name boundary as the catalogue: invalid identifiers
fail before registry lookup, while valid but unregistered names remain registry
misses. Missing legacy v1 schemas are allowed so new models can receive honest
empty curation fields, but malformed curated schemas abort the refresh instead
of being silently discarded.

The v2 descriptor parser accepts both compact legacy scalar forms and expanded
tables, then normalises them into `ModelDescriptor`, `ParameterSpec`,
`StateVariableSpec`, `BackendSupport`, and reproducibility records. It rejects
missing metadata, non-table sections, scalar tag fields, non-numeric values,
malformed ranges, invalid provenance years, and invalid digest formats before a
descriptor reaches the catalogue or Studio browser surface.

Tier-3 golden traces remain exact-hash contracts. When a scalar transcendental
has two measured NumPy SIMD results on heterogeneous x86 hosts, the descriptor's
`[reproducibility]` table may add `golden_trace_sha256_variants`; the typed parser
requires every entry to be a unique, non-primary, lowercase SHA-256. The
reproducibility gate accepts only that finite allowlist, reruns variant-bearing
models with AVX-512 disabled, and requires the native and portable traces to stay
within the same `1e-9` bound used for ULP-bounded backend parity. This records
platform-level rounding honestly without turning the golden gate into a broad
approximate comparison. Studio model-detail responses expose the typed variant
list so clients can verify either measured platform trace without parsing the
raw TOML descriptor.

### Trivial IF Variants (18 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `QuadraticIFNeuron` | `QuadraticIFNeuron` | Latham et al. 2000 |
| `ThetaNeuron` | `ThetaNeuron` | Ermentrout & Kopell 1986 |
| `PerfectIntegratorNeuron` | `PerfectIntegratorNeuron` | — |
| `GatedLIFNeuron` | `GatedLIFNeuron` | — |
| `NonlinearLIFNeuron` | `NonlinearLIFNeuron` | Touboul & Brette 2008 |
| `SFANeuron` | `SFANeuron` | Benda & Herz 2003 |
| `MATNeuron` | `MATNeuron` | Kobayashi et al. 2009 |
| `EscapeRateNeuron` | `EscapeRateNeuron` | Gerstner 2000 |
| `KLIFNeuron` | `KLIFNeuron` | Eshraghian et al. 2021 |
| `InhibitoryLIFNeuron` | `InhibitoryLIFNeuron` | — |
| `ComplementaryLIFNeuron` | `ComplementaryLIFNeuron` | — |
| `ParametricLIFNeuron` | `ParametricLIFNeuron` | Fang et al. 2021 |
| `NonResettingLIFNeuron` | `NonResettingLIFNeuron` | Brette 2004 |
| `AdaptiveThresholdIFNeuron` | `AdaptiveThresholdIFNeuron` | Platkiewicz & Brette 2010 |
| `SigmaDeltaNeuron` | `SigmaDeltaNeuron` | — |
| `EnergyLIFNeuron` | `EnergyLIFNeuron` | Sengupta et al. 2013 |
| `IntegerQIFNeuron` | `IntegerQIFNeuron` | Wu et al. 2021, DOI `10.1109/AICAS51828.2021.9458572` |
| `ClosedFormContinuousNeuron` | `ClosedFormContinuousNeuron` | Hasani et al. 2022 |

### Simple Spiking (21 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `FitzHughNagumoNeuron` | `FitzHughNagumoNeuron` | FitzHugh 1961 |
| `MorrisLecarNeuron` | `MorrisLecarNeuron` | Morris & Lecar 1981 |
| `HindmarshRoseNeuron` | `HindmarshRoseNeuron` | Hindmarsh & Rose 1984 |
| `ResonateAndFireNeuron` | `ResonateAndFireNeuron` | Izhikevich 2001 |
| `BalancedResonateAndFireNeuron` | `BalancedResonateAndFireNeuron` | Higuchi et al. 2024 |
| `FitzHughRinzelNeuron` | `FitzHughRinzelNeuron` | Rinzel 1987 |
| `McKeanNeuron` | `McKeanNeuron` | McKean 1970 |
| `TermanWangOscillator` | `TermanWangOscillator` | Terman & Wang 1995 |
| `BendaHerzNeuron` | `BendaHerzNeuron` | Benda & Herz 2003 |
| `AlphaNeuron` | `AlphaNeuron` | — |
| `COBALIFNeuron` | `COBALIFNeuron` | Brette et al. 2007 |
| `GutkinErmentroutNeuron` | `GutkinErmentroutNeuron` | Gutkin & Ermentrout 1998 |
| `WilsonHRNeuron` | `WilsonHRNeuron` | Wilson 1999 |
| `ChayNeuron` | `ChayNeuron` | Chay 1985 |
| `ChayKeizerNeuron` | `ChayKeizerNeuron` | Chay & Keizer 1983 |
| `ShermanRinzelKeizerNeuron` | `ShermanRinzelKeizerNeuron` | Sherman et al. 1988 |
| `ButeraRespiratoryNeuron` | `ButeraRespiratoryNeuron` | Butera et al. 1999 |
| `EPropALIFNeuron` | `EPropALIFNeuron` | Bellec et al. 2020 |
| `SuperSpikeNeuron` | `SuperSpikeNeuron` | Zenke & Ganguli 2018 |
| `LearnableNeuronModel` | `LearnableNeuronModel` | — |
| `PernarowskiNeuron` | `PernarowskiNeuron` | Pernarowski 1994 |

### Discrete Maps (7 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `ChialvoMapNeuron` | `ChialvoMapNeuron` | Chialvo 1995 |
| `RulkovMapNeuron` | `RulkovMapNeuron` | Rulkov 2001 |
| `IbarzTanakaMapNeuron` | `IbarzTanakaMapNeuron` | Ibarz, Tanaka, Sanjuan & Aihara 2007, Eqs. 2–3 |
| `MedvedevMapNeuron` | `MedvedevMapNeuron` | Medvedev 2005 slow-calcium first-return reduction |
| `CazellesMapNeuron` | `CazellesMapNeuron` | Cazelles et al. 2001 |
| `CourageNekorkinMapNeuron` | `CourageNekorkinMapNeuron` | Courbage, Nekorkin & Vdovin 2007 |
| `ErmentroutKopellMapNeuron` | `ErmentroutKopellMapNeuron` | Ermentrout & Kopell 1986 (maintained Euler map) |

### Biophysical / Conductance-Based (20 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `HodgkinHuxleyNeuron` | `HodgkinHuxleyNeuron` | Hodgkin & Huxley 1952 |
| `TraubMilesNeuron` | `TraubMilesNeuron` | Traub & Miles 1991 |
| `WangBuzsakiNeuron` | `WangBuzsakiNeuron` | Wang & Buzsáki 1996 |
| `ConnorStevensNeuron` | `ConnorStevensNeuron` | Connor et al. 1977 |
| `DestexheThalamicNeuron` | `DestexheThalamicNeuron` | Destexhe et al. 1993 |
| `HuberBraunNeuron` | `HuberBraunNeuron` | Braun et al. 1998 |
| `GolombFSNeuron` | `GolombFSNeuron` | Golomb et al. 2007 |
| `PospischilNeuron` | `PospischilNeuron` | Pospischil et al. 2008 |
| `MainenSejnowskiNeuron` | `MainenSejnowskiNeuron` | Mainen & Sejnowski 1996 |
| `DeSchutterPurkinjeNeuron` | `DeSchutterPurkinjeNeuron` | De Schutter & Bower 1994 |
| `PlantR15Neuron` | `PlantR15Neuron` | Plant & Kim 1976 |
| `PrescottNeuron` | `PrescottNeuron` | Prescott et al. 2008 |
| `MihalasNieburNeuron` | `MihalasNieburNeuron` | Mihalas & Niebur 2009 |
| `GLIFNeuron` | `GLIFNeuron` | Allen Institute GLIF5 |
| `GIFPopulationNeuron` | `GIFPopulationNeuron` | Mensi et al. 2012 |
| `AvRonCardiacNeuron` | `AvRonCardiacNeuron` | Av-Ron et al. 1991 |
| `DurstewitzDopamineNeuron` | `DurstewitzDopamineNeuron` | Durstewitz et al. 2000 |
| `HillTononiNeuron` | `HillTononiNeuron` | Hill & Tononi 2005 |
| `BertramPhantomBurster` | `BertramPhantomBurster` | Bertram et al. 2000 |
| `YamadaNeuron` | `YamadaNeuron` | Yamada et al. 1989 |

### Multi-Compartment (7 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `PinskyRinzelNeuron` | `PinskyRinzelNeuron` | Pinsky & Rinzel 1994 |
| `HayL5PyramidalNeuron` | `HayL5PyramidalNeuron` | Hay et al. 2011 |
| `MarderSTGNeuron` | `MarderSTGNeuron` | Marder & Calabrese 1996 |
| `RallCableNeuron` | `RallCableNeuron` | Rall 1964 |
| `BoothRinzelNeuron` | `BoothRinzelNeuron` | Booth et al. 1997 |
| `DendrifyNeuron` | `DendrifyNeuron` | Beniaguev et al. 2022 |
| `TwoCompartmentLIFNeuron` | `TwoCompartmentLIFNeuron` | — |

### Stochastic / Population / Neural Mass (13 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `PoissonNeuron` | `PoissonNeuron` | Gerstner et al. 2014, Sections 7.2 and 7.7 |
| `InhomogeneousPoissonNeuron` | `InhomogeneousPoissonNeuron` | — |
| `GammaRenewalNeuron` | `GammaRenewalNeuron` | — |
| `StochasticIFNeuron` | `StochasticIFNeuron` | — |
| `GalvesLocherbachNeuron` | `GalvesLocherbachNeuron` | Galves & Löcherbach 2013 |
| `SpikeResponseNeuron` | `SpikeResponseNeuron` | Gerstner 1995 (SRM0) |
| `GLMNeuron` | `GLMNeuron` | Pillow et al. 2008 |
| `WilsonCowanUnit` | `WilsonCowanUnit` | Wilson & Cowan 1972 |
| `JansenRitUnit` | `JansenRitUnit` | Jansen & Rit 1995 |
| `WongWangUnit` | `WongWangUnit` | Wong & Wang 2006 |
| `ErmentroutKopellPopulation` | `ErmentroutKopellPopulation` | Montbrió et al. 2015 |
| `WendlingNeuron` | `WendlingNeuron` | Wendling et al. 2002 |
| `LarterBreakspearNeuron` | `LarterBreakspearNeuron` | Breakspear et al. 2003 |

### Hardware Chip Emulators (9 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `LoihiCUBANeuron` | `LoihiCUBANeuron` | Davies et al. 2018 (Intel Loihi) |
| `Loihi2Neuron` | `Loihi2Neuron` | Intel Loihi 2 |
| `TrueNorthNeuron` | `TrueNorthNeuron` | Merolla et al. 2014 (IBM) |
| `BrainScaleSAdExNeuron` | `BrainScaleSAdExNeuron` | Schemmel et al. 2010 |
| `SpiNNakerLIFNeuron` | `SpiNNakerLIFNeuron` | Furber et al. 2014 |
| `SpiNNaker2Neuron` | `SpiNNaker2Neuron` | TU Dresden 2024 |
| `DPINeuron` | `DPINeuron` | Indiveri, Stefanini & Chicca 2010 |
| `AkidaNeuron` | `AkidaNeuron` | BrainChip |
| `NeuroGridNeuron` | `NeuroGridNeuron` | Boahen 2014 |

### Rate / Plasticity / Other (12 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `McCullochPittsNeuron` | `McCullochPittsNeuron` | McCulloch & Pitts 1943 |
| `SigmoidRateNeuron` | `SigmoidRateNeuron` | Wilson & Cowan 1972 |
| `ThresholdLinearRateNeuron` | `ThresholdLinearRateNeuron` | — |
| `AstrocyteModel` | `AstrocyteModel` | Li & Rinzel 1994 |
| `TsodyksMarkramNeuron` | `TsodyksMarkramNeuron` | Tsodyks & Markram 1997 |
| `LiquidTimeConstantNeuron` | `LiquidTimeConstantNeuron` | Hasani et al. 2021 |
| `CompteWMNeuron` | `CompteWMNeuron` | Compte et al. 2000 |
| `SiegertTransferFunction` | `SiegertTransferFunction` | Siegert 1951 |
| `FractionalLIFNeuron` | `FractionalLIFNeuron` | Teka et al. 2014 |
| `ParallelSpikingNeuron` | `ParallelSpikingNeuron` | Fang et al. 2023 |
| `AmariNeuralField` | `AmariNeuralField` | Amari 1977 |
| `LeakyCompeteFireNeuron` | `LeakyCompeteFireNeuron` | — |

### Core IF (Rust neuron.rs) (3 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `AdExNeuron` | `AdExNeuron` | Brette & Gerstner 2005 |
| `ExpIFNeuron` | `ExpIFNeuron` | Fourcaud-Trocmé et al. 2003 |
| `LapicqueNeuron` | `LapicqueNeuron` | Lapicque 1907 |

## Common Interface

All models share:

```python
model.step(current)   # → int (spike: 1/0) or float (firing rate)
model.reset()         # → None (restore initial conditions)
model.get_state()     # → dict of current state variables
```

Multi-input models accept additional arguments:
- `AlphaNeuron.step(exc_current, inh_current)`
- `COBALIFNeuron.step(current, delta_ge, delta_gi)`
- `PinskyRinzelNeuron.step(current_soma, current_dend)`
- `TsodyksMarkramNeuron.step(current, presynaptic_spike)`
- `CompteWMNeuron.step(current, spike_in)`

Neural mass models return `float` (firing rate or EEG potential):
- `WilsonCowanUnit`, `JansenRitUnit`, `WendlingNeuron`
- `ErmentroutKopellPopulation`, `LarterBreakspearNeuron`
- `SigmoidRateNeuron`, `SiegertTransferFunction`

`WongWangUnit.step(stim1, stim2)` returns the two continuous pre-update rates as
`(r1, r2)`. Its deterministic batch additionally accepts interleaved external
Gaussian samples and returns six complete traces plus four final states.

`JansenRitUnit.step(p_ext)` returns the post-update continuous `y1-y2` EEG
proxy. Its atomic batch accepts one external-drive sample per Euler step and
returns all six state traces, the EEG trace, and six final-state receipts.

`ErmentroutKopellPopulation.step(ext_input)` returns the post-update continuous
population firing rate from the Montbrió–Pazó–Roxin exact QIF-network mean
field. Its atomic batch returns complete `r` and `v` traces plus both final-
state receipts; the compatibility name does not denote the 1986 single-cell
Ermentrout–Kopell theta equation.

### AI-Optimized (9 models)

Novel neuron models designed for AI workloads, not biological simulation.
Located in `neurons/models/ai_optimized.py` and `neurons/models/arcane_neuron.py`.

| Python Class | Rust Class | Key Feature |
|-------------|-----------|-------------|
| `ArcaneNeuron` | `ArcaneNeuron` | 5-compartment self-referential cognition: fast (5ms), working memory (200ms), deep context (10s), attention gate, forward self-model. Identity accumulates in the deep compartment. Confidence modulates threshold and meta-learning rate. Sotek & Arcane Sapience 2026. |
| `MultiTimescaleNeuron` | — | Three-compartment (fast/medium/slow) with context-dependent threshold modulation |
| `AttentionGatedNeuron` | — | Learned sigmoid gate (key/query weights) selectively filters input |
| `PredictiveCodingNeuron` | — | Fires only on prediction errors (novel stimuli), silent on expected input |
| `SelfReferentialNeuron` | — | Introspects own spike history to auto-regulate firing dynamics |
| `CompositionalBindingNeuron` | — | Phase-coding for variable binding; in-phase = bound concepts |
| `DifferentiableSurrogateNeuron` | — | Trainable surrogate gradient parameters (alpha, beta, theta) |
| `ContinuousAttractorNeuron` | — | Ring attractor with Mexican-hat connectivity for continuous working memory |
| `MetaPlasticNeuron` | — | Self-regulating meta-learning rate based on error trace |
