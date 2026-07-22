<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 22: Choosing the Right Neuron Model

SC-NeuroCore currently exposes 158 lazy-loaded Python model classes across
153 Python model source modules, with 181 Rust PyO3 model wrappers in the
optional engine. This guide helps you pick the right model for your
application.

## Decision Tree

```
What is your goal?
│
├─ FPGA deployment → FixedPointLIFNeuron (bit-true Q8.8 RTL match)
│
├─ GPU training (PyTorch) → LIFCell, AdExCell, or ConvSpikingNet
│
├─ Biophysical realism
│  ├─ Ion channels → HodgkinHuxleyNeuron (4 ODEs, Nobel 1963)
│  ├─ Cortical L5 → HayL5PyramidalNeuron (3-compartment, BAC firing)
│  ├─ Fast-spiking → WangBuzsakiNeuron (gamma oscillations)
│  ├─ Thalamic relay → DestexheThalamicNeuron (T-current, burst/tonic)
│  ├─ Hippocampal → TraubMilesNeuron (CA3 pyramidal)
│  └─ Cerebellar → DeSchutterPurkinjeNeuron (P-type Ca)
│
├─ Bursting behaviour
│  ├─ Chaotic → HindmarshRoseNeuron (3D, chaotic bursting)
│  ├─ Parabolic → PlantR15Neuron (Aplysia)
│  ├─ Square-wave → ShermanRinzelKeizerNeuron (pancreatic beta)
│  └─ Respiratory → ButeraRespiratoryNeuron (pre-Botzinger)
│
├─ Simple + fast
│  ├─ 1 ODE → LapicqueNeuron, QuadraticIFNeuron
│  ├─ 2 ODEs → Izhikevich, AdExNeuron, FitzHughNagumoNeuron
│  └─ Map (O(1)) → RulkovMapNeuron, ChialvoMapNeuron
│
├─ Adaptive firing rate
│  ├─ Threshold adaptation → HomeostaticLIFNeuron, MATNeuron, GLIFNeuron
│  ├─ Spike-frequency adaptation → SFANeuron, BendaHerzNeuron
│  └─ Reward-modulated → EPropALIFNeuron (with eligibility traces)
│
├─ Population / neural mass (EEG, fMRI)
│  ├─ EEG → JansenRitUnit, WendlingNeuron (epilepsy)
│  ├─ Decision-making → WongWangUnit (attractor dynamics)
│  ├─ Whole-brain → LarterBreakspearNeuron (TVB compatible)
│  └─ Exact mean-field → ErmentroutKopellPopulation
│
├─ Hardware-specific
│  ├─ Intel Loihi → LoihiCUBANeuron, Loihi2Neuron
│  ├─ IBM TrueNorth → TrueNorthNeuron
│  ├─ SpiNNaker → SpiNNakerLIFNeuron, SpiNNaker2Neuron
│  ├─ BrainScaleS → BrainScaleSAdExNeuron
│  ├─ DYNAP-SE → DPINeuron
│  └─ BrainChip Akida → AkidaNeuron
│
├─ ML / training-oriented
│  ├─ Learnable tau → ParametricLIFNeuron (PLIF, Fang 2021)
│  ├─ Learnable gates → GatedLIFNeuron (NeurIPS 2022)
│  ├─ Fully learnable → LearnableNeuronModel (all params learnable)
│  ├─ Liquid networks → LiquidTimeConstantNeuron, ClosedFormContinuousNeuron
│  └─ Parallel processing → ParallelSpikingNeuron (no sequential dependency)
│
├─ Stochastic / statistical
│  ├─ Rate-based input → PoissonNeuron, InhomogeneousPoissonNeuron
│  ├─ Stochastic threshold → EscapeRateNeuron, GalvesLocherbachNeuron
│  ├─ Noise-driven → StochasticIFNeuron (Ornstein-Uhlenbeck)
│  ├─ GLM encoding → GLMNeuron (stimulus + history filters)
│  └─ Regular spiking → GammaRenewalNeuron
│
└─ Glial / non-neuronal
   └─ Astrocyte Ca2+ → AstrocyteModel (IP3 receptor dynamics)
```

## Quick Comparison by Speed

| Speed | Models | Use case |
|-------|--------|----------|
| O(1) per step | McCullochPitts, Rulkov, Chialvo, Medvedev | Largest networks |
| 1 ODE | LIF, IF, QIF, ExpIF, Lapicque, Theta | Standard SNN |
| 2 ODEs | Izhikevich, AdEx, FHN, Morris-Lecar, SRM | Rich dynamics |
| 3 ODEs | Hindmarsh-Rose, FitzHugh-Rinzel, Pernarowski | Bursting |
| 4+ ODEs | HH, Connor-Stevens, Destexhe | Biophysical detail |
| 6+ ODEs | Hay L5, Jansen-Rit, Marder STG | Multi-compartment / neural mass |

## Quick Start Examples

### Simple LIF network
```python
from sc_neurocore.neurons import StochasticLIFNeuron

neuron = StochasticLIFNeuron()
for t in range(1000):
    spike = neuron.step(input_current=1.5)
```

### Biophysical Hodgkin-Huxley
```python
from sc_neurocore.neurons import HodgkinHuxleyNeuron

hh = HodgkinHuxleyNeuron()
for t in range(1000):
    spike = hh.step(current=10.0)  # 10 uA/cm^2
```

### Cortical L5 pyramidal with BAC firing
```python
from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron

l5 = HayL5PyramidalNeuron()
for t in range(1000):
    spike = l5.step(current_soma=5.0, current_tuft=2.0)
```

### Hardware deployment (Intel Loihi)
```python
from sc_neurocore.neurons import LoihiCUBANeuron

loihi = LoihiCUBANeuron(v_threshold=500)
spike = loihi.step(weighted_input=200)
```

### Population-level EEG simulation
```python
from sc_neurocore.neurons import JansenRitUnit

jr = JansenRitUnit()
eeg_signal = [jr.step(p_ext=220.0) for _ in range(10000)]
```

## Model Categories

### 1. Integrate-and-Fire Family (26 models)
The workhorse of SNN simulation. Fast, analytically tractable.

LIF, FixedPointLIF, HomeostaticLIF, IF, ALIF, Lapicque, ExpIF, QIF,
SecondOrderLIF, FractionalLIF, GLIF, MAT, SFA, StochasticIF, EscapeRate,
PerfectIntegrator, NLIF, COBA, AdaptiveThreshold, PLIF, NonResetting,
GatedLIF, SigmaDelta, KLIF, ILIF, IntegerQIF

### 2. Biophysical / Conductance-Based (12 models)
Ion-channel-level detail. For understanding spike generation mechanisms.

HodgkinHuxley, ConnorStevens, WangBuzsaki, PinskyRinzel, Destexhe,
HuberBraun, GutkinErmentrout, TraubMiles, GolombFS, MainenSejnowski,
Pospischil, DeSchutterPurkinje

### 3. Multi-Compartment (4 models)
Separate soma and dendrite dynamics. For dendritic computation.

HayL5Pyramidal, BoothRinzel, Dendrify, TwoCompartmentLIF

### 4. Adaptive (4 models)
Spike-frequency adaptation and threshold dynamics.

AdEx, Izhikevich, MihalasNiebur, BendaHerz

### 5. Oscillatory / Qualitative (7 models)
Phase-plane dynamics, bifurcation analysis.

FitzHughNagumo, MorrisLecar, HindmarshRose, ResonateAndFire, Theta,
FitzHughRinzel, TermanWang

### 6. Bursting (6 models)
Periodic bursts of spikes followed by quiescence.

Chay, ButeraRespiratory, ShermanRinzelKeizer, PlantR15,
BertramPhantom, Pernarowski

### 7. Map-Based / Discrete (6 models)
Discrete-time iteration, no ODE solver needed. Fastest option.

Rulkov, Chialvo, CourageNekorkin, Medvedev, IbarzTanaka, Cazelles

### 8. Population / Neural Mass (6 models)
Model whole brain regions, not individual neurons.

WilsonCowan, JansenRit, WongWang, ErmentroutKopell, AmariField,
Wendling, LarterBreakspear

### 9. Hardware-Specific (9 models)
Match the behaviour of specific neuromorphic chips.

LoihiCUBA, Loihi2, TrueNorth, BrainScaleSAdEx, SpiNNakerLIF,
SpiNNaker2, DPI, Akida, SigmaDelta

### 10. Stochastic / Statistical (5 models)
Probabilistic spike generation.

Poisson, InhomogeneousPoisson, GalvesLocherbach, GLM, GammaRenewal

### 11. Synaptic (3 models)
Detailed synaptic current dynamics.

Alpha, Synaptic (dual-exponential), TsodyksMarkram

### 12. ML / Modern (8 models)
Designed for gradient-based training.

ParametricLIF, GatedLIF, LearnableNeuronModel, LiquidTimeConstant,
ClosedFormContinuous, ParallelSpiking, EPropALIF, SuperSpike

### 13. Rate / Abstract (4 models)
Non-spiking, continuous-valued models.

McCullochPitts, SigmoidRate, ThresholdLinearRate, WilsonHR

### 14. Other (2 models)
McKean (piecewise-linear FHN), Astrocyte (glial Ca2+)

## Next Steps

- [Tutorial 01: SC Fundamentals](01_stochastic_computing_fundamentals.md) — bitstream encoding
- [Tutorial 17: Custom Neuron Models](17_custom_neuron_models.md) — extend BaseNeuron
- [Tutorial 20: Adapter Ecosystem](20_adapter_ecosystem.md) — SCPN layer mapping
- [API Reference: Neurons](../api/neurons.md) — full class signatures
