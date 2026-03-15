# Neuron Model Reference — 111 Models

SC-NeuroCore provides 111 neuron models implemented in both Python
(dataclass) and Rust (PyO3-bound). Both backends expose identical
`step()` / `reset()` / `get_state()` interfaces.

## Quick Start

```python
# Python backend (default)
from sc_neurocore.neurons.models import HodgkinHuxleyNeuron
hh = HodgkinHuxleyNeuron()
spike = hh.step(current=10.0)

# Rust backend (faster, identical interface)
from sc_neurocore_engine.sc_neurocore_engine import PyHodgkinHuxleyNeuron
hh_rs = PyHodgkinHuxleyNeuron()
spike = hh_rs.step(current=10.0)
```

## Backend Selection

| Backend | Import path | Use case |
|---------|------------|----------|
| Python | `sc_neurocore.neurons.models` | Prototyping, parameter tuning, debugging |
| Rust | `sc_neurocore_engine.sc_neurocore_engine` | Production, benchmarks, batch simulation |

Rust names are prefixed with `Py` (e.g., `PyHodgkinHuxleyNeuron`).
The Python names have no prefix (`HodgkinHuxleyNeuron`).

## Model Catalogue

### Trivial IF Variants (18 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `QuadraticIFNeuron` | `PyQuadraticIFNeuron` | Latham et al. 2000 |
| `ThetaNeuron` | `PyThetaNeuron` | Ermentrout & Kopell 1986 |
| `PerfectIntegratorNeuron` | `PyPerfectIntegratorNeuron` | — |
| `GatedLIFNeuron` | `PyGatedLIFNeuron` | — |
| `NonlinearLIFNeuron` | `PyNonlinearLIFNeuron` | Touboul & Brette 2008 |
| `SFANeuron` | `PySFANeuron` | Benda & Herz 2003 |
| `MATNeuron` | `PyMATNeuron` | Kobayashi et al. 2009 |
| `EscapeRateNeuron` | `PyEscapeRateNeuron` | Gerstner 2000 |
| `KLIFNeuron` | `PyKLIFNeuron` | Eshraghian et al. 2021 |
| `InhibitoryLIFNeuron` | `PyInhibitoryLIFNeuron` | — |
| `ComplementaryLIFNeuron` | `PyComplementaryLIFNeuron` | — |
| `ParametricLIFNeuron` | `PyParametricLIFNeuron` | Fang et al. 2021 |
| `NonResettingLIFNeuron` | `PyNonResettingLIFNeuron` | Brette 2004 |
| `AdaptiveThresholdIFNeuron` | `PyAdaptiveThresholdIFNeuron` | Platkiewicz & Brette 2010 |
| `SigmaDeltaNeuron` | `PySigmaDeltaNeuron` | — |
| `EnergyLIFNeuron` | `PyEnergyLIFNeuron` | Sengupta et al. 2013 |
| `IntegerQIFNeuron` | `PyIntegerQIFNeuron` | — |
| `ClosedFormContinuousNeuron` | `PyClosedFormContinuousNeuron` | Hasani et al. 2022 |

### Simple Spiking (20 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `FitzHughNagumoNeuron` | `PyFitzHughNagumoNeuron` | FitzHugh 1961 |
| `MorrisLecarNeuron` | `PyMorrisLecarNeuron` | Morris & Lecar 1981 |
| `HindmarshRoseNeuron` | `PyHindmarshRoseNeuron` | Hindmarsh & Rose 1984 |
| `ResonateAndFireNeuron` | `PyResonateAndFireNeuron` | Izhikevich 2001 |
| `FitzHughRinzelNeuron` | `PyFitzHughRinzelNeuron` | Rinzel 1987 |
| `McKeanNeuron` | `PyMcKeanNeuron` | McKean 1970 |
| `TermanWangOscillator` | `PyTermanWangOscillator` | Terman & Wang 1995 |
| `BendaHerzNeuron` | `PyBendaHerzNeuron` | Benda & Herz 2003 |
| `AlphaNeuron` | `PyAlphaNeuron` | — |
| `COBALIFNeuron` | `PyCOBALIFNeuron` | Brette et al. 2007 |
| `GutkinErmentroutNeuron` | `PyGutkinErmentroutNeuron` | Gutkin & Ermentrout 1998 |
| `WilsonHRNeuron` | `PyWilsonHRNeuron` | Wilson 1999 |
| `ChayNeuron` | `PyChayNeuron` | Chay 1985 |
| `ChayKeizerNeuron` | `PyChayKeizerNeuron` | Chay & Keizer 1983 |
| `ShermanRinzelKeizerNeuron` | `PyShermanRinzelKeizerNeuron` | Sherman et al. 1988 |
| `ButeraRespiratoryNeuron` | `PyButeraRespiratoryNeuron` | Butera et al. 1999 |
| `EPropALIFNeuron` | `PyEPropALIFNeuron` | Bellec et al. 2020 |
| `SuperSpikeNeuron` | `PySuperSpikeNeuron` | Zenke & Ganguli 2018 |
| `LearnableNeuronModel` | `PyLearnableNeuronModel` | — |
| `PernarowskiNeuron` | `PyPernarowskiNeuron` | Pernarowski 1994 |

### Discrete Maps (6 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `ChialvoMapNeuron` | `PyChialvoMapNeuron` | Chialvo 1995 |
| `RulkovMapNeuron` | `PyRulkovMapNeuron` | Rulkov 2001 |
| `IbarzTanakaMapNeuron` | `PyIbarzTanakaMapNeuron` | Ibarz et al. 2011 |
| `MedvedevMapNeuron` | `PyMedvedevMapNeuron` | Medvedev 2005 |
| `CazellesMapNeuron` | `PyCazellesMapNeuron` | Cazelles et al. 2001 |
| `CourageNekorkinMapNeuron` | `PyCourageNekorkinMapNeuron` | Courbage & Nekorkin 2010 |

### Biophysical / Conductance-Based (20 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `HodgkinHuxleyNeuron` | `PyHodgkinHuxleyNeuron` | Hodgkin & Huxley 1952 |
| `TraubMilesNeuron` | `PyTraubMilesNeuron` | Traub & Miles 1991 |
| `WangBuzsakiNeuron` | `PyWangBuzsakiNeuron` | Wang & Buzsáki 1996 |
| `ConnorStevensNeuron` | `PyConnorStevensNeuron` | Connor et al. 1977 |
| `DestexheThalamicNeuron` | `PyDestexheThalamicNeuron` | Destexhe et al. 1993 |
| `HuberBraunNeuron` | `PyHuberBraunNeuron` | Braun et al. 1998 |
| `GolombFSNeuron` | `PyGolombFSNeuron` | Golomb et al. 2007 |
| `PospischilNeuron` | `PyPospischilNeuron` | Pospischil et al. 2008 |
| `MainenSejnowskiNeuron` | `PyMainenSejnowskiNeuron` | Mainen & Sejnowski 1996 |
| `DeSchutterPurkinjeNeuron` | `PyDeSchutterPurkinjeNeuron` | De Schutter & Bower 1994 |
| `PlantR15Neuron` | `PyPlantR15Neuron` | Plant & Kim 1976 |
| `PrescottNeuron` | `PyPrescottNeuron` | Prescott et al. 2008 |
| `MihalasNieburNeuron` | `PyMihalasNieburNeuron` | Mihalas & Niebur 2009 |
| `GLIFNeuron` | `PyGLIFNeuron` | Allen Institute GLIF5 |
| `GIFPopulationNeuron` | `PyGIFPopulationNeuron` | Mensi et al. 2012 |
| `AvRonCardiacNeuron` | `PyAvRonCardiacNeuron` | Av-Ron et al. 1991 |
| `DurstewitzDopamineNeuron` | `PyDurstewitzDopamineNeuron` | Durstewitz et al. 2000 |
| `HillTononiNeuron` | `PyHillTononiNeuron` | Hill & Tononi 2005 |
| `BertramPhantomBurster` | `PyBertramPhantomBurster` | Bertram et al. 2000 |
| `YamadaNeuron` | `PyYamadaNeuron` | Yamada et al. 1989 |

### Multi-Compartment (7 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `PinskyRinzelNeuron` | `PyPinskyRinzelNeuron` | Pinsky & Rinzel 1994 |
| `HayL5PyramidalNeuron` | `PyHayL5PyramidalNeuron` | Hay et al. 2011 |
| `MarderSTGNeuron` | `PyMarderSTGNeuron` | Marder & Calabrese 1996 |
| `RallCableNeuron` | `PyRallCableNeuron` | Rall 1964 |
| `BoothRinzelNeuron` | `PyBoothRinzelNeuron` | Booth et al. 1997 |
| `DendrifyNeuron` | `PyDendrifyNeuron` | Beniaguev et al. 2022 |
| `TwoCompartmentLIFNeuron` | `PyTwoCompartmentLIFNeuron` | — |

### Stochastic / Population / Neural Mass (13 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `PoissonNeuron` | `PyPoissonNeuron` | — |
| `InhomogeneousPoissonNeuron` | `PyInhomogeneousPoissonNeuron` | — |
| `GammaRenewalNeuron` | `PyGammaRenewalNeuron` | — |
| `StochasticIFNeuron` | `PyStochasticIFNeuron` | — |
| `GalvesLocherbachNeuron` | `PyGalvesLocherbachNeuron` | Galves & Löcherbach 2013 |
| `SpikeResponseNeuron` | `PySpikeResponseNeuron` | Gerstner 1995 (SRM0) |
| `GLMNeuron` | `PyGLMNeuron` | Pillow et al. 2008 |
| `WilsonCowanUnit` | `PyWilsonCowanUnit` | Wilson & Cowan 1972 |
| `JansenRitUnit` | `PyJansenRitUnit` | Jansen & Rit 1995 |
| `WongWangUnit` | `PyWongWangUnit` | Wong & Wang 2006 |
| `ErmentroutKopellPopulation` | `PyErmentroutKopellPopulation` | Montbrió et al. 2015 |
| `WendlingNeuron` | `PyWendlingNeuron` | Wendling et al. 2002 |
| `LarterBreakspearNeuron` | `PyLarterBreakspearNeuron` | Breakspear et al. 2003 |

### Hardware Chip Emulators (9 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `LoihiCUBANeuron` | `PyLoihiCUBANeuron` | Davies et al. 2018 (Intel Loihi) |
| `Loihi2Neuron` | `PyLoihi2Neuron` | Intel Loihi 2 |
| `TrueNorthNeuron` | `PyTrueNorthNeuron` | Merolla et al. 2014 (IBM) |
| `BrainScaleSAdExNeuron` | `PyBrainScaleSAdExNeuron` | Schemmel et al. 2010 |
| `SpiNNakerLIFNeuron` | `PySpiNNakerLIFNeuron` | Furber et al. 2014 |
| `SpiNNaker2Neuron` | `PySpiNNaker2Neuron` | TU Dresden 2024 |
| `DPINeuron` | `PyDPINeuron` | Bartolozzi & Indiveri 2007 |
| `AkidaNeuron` | `PyAkidaNeuron` | BrainChip |
| `NeuroGridNeuron` | `PyNeuroGridNeuron` | Boahen 2014 |

### Rate / Plasticity / Other (11 models)

| Python Class | Rust Class | Reference |
|-------------|-----------|-----------|
| `McCullochPittsNeuron` | `PyMcCullochPittsNeuron` | McCulloch & Pitts 1943 |
| `SigmoidRateNeuron` | `PySigmoidRateNeuron` | Wilson & Cowan 1972 |
| `ThresholdLinearRateNeuron` | `PyThresholdLinearRateNeuron` | — |
| `AstrocyteModel` | `PyAstrocyteModel` | Li & Rinzel 1994 |
| `TsodyksMarkramNeuron` | `PyTsodyksMarkramNeuron` | Tsodyks & Markram 1997 |
| `LiquidTimeConstantNeuron` | `PyLiquidTimeConstantNeuron` | Hasani et al. 2021 |
| `CompteWMNeuron` | `PyCompteWMNeuron` | Compte et al. 2000 |
| `SiegertTransferFunction` | `PySiegertTransferFunction` | Siegert 1951 |
| `FractionalLIFNeuron` | (Python only) | Teka et al. 2014 |
| `ParallelSpikingNeuron` | (Python only) | Fang et al. 2023 |
| `AmariNeuralField` | (Python only) | Amari 1977 |

Note: FractionalLIF, PSN, and AmariNeuralField use dynamically-sized
buffers; Rust implementations exist but PyO3 wrappers are deferred
pending array-return API design.

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
