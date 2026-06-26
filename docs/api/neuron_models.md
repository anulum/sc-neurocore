# Neuron Model Reference — 158 Python Classes / 175 Rust PyO3 Wrappers

SC-NeuroCore currently exposes 158 lazy-loaded Python model classes across
152 Python model source modules in `src/sc_neurocore/neurons/models/`, plus
175 Rust PyO3 model wrappers in the optional engine. Matching model classes
use the same `step()` / `reset()` / `get_state()` interface shape where the
backend implements that surface.

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
161 of which are wired into the NetworkRunner pipeline.

## Model Catalogue

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
| `IntegerQIFNeuron` | `IntegerQIFNeuron` | — |
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

### Discrete Maps (6 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `ChialvoMapNeuron` | `ChialvoMapNeuron` | Chialvo 1995 |
| `RulkovMapNeuron` | `RulkovMapNeuron` | Rulkov 2001 |
| `IbarzTanakaMapNeuron` | `IbarzTanakaMapNeuron` | Ibarz et al. 2011 |
| `MedvedevMapNeuron` | `MedvedevMapNeuron` | Medvedev 2005 |
| `CazellesMapNeuron` | `CazellesMapNeuron` | Cazelles et al. 2001 |
| `CourageNekorkinMapNeuron` | `CourageNekorkinMapNeuron` | Courbage & Nekorkin 2010 |

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
| `PoissonNeuron` | `PoissonNeuron` | — |
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
| `DPINeuron` | `DPINeuron` | Bartolozzi & Indiveri 2007 |
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
