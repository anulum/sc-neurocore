# Neurons

Python-facing neuron models spanning classical integrate-and-fire dynamics,
conductance-based cells, neural-mass models, maps, hardware-specific neurons,
and differentiable training cells. Use the source tree and benchmark inventory
as the authority for exact model and backend counts.

## Quick Start

```python
# Flat import (any model)
from sc_neurocore.neurons import HodgkinHuxleyNeuron, AdExNeuron

# Individual file import
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
```

## Reference Trace Validation

Schema-driven models can be checked against committed reference-trace feature
contracts through `sc_neurocore.neurons.reference_traces`. The current corpus
covers deterministic `lif`, `lapicque`, and `quadratic_if` schema entries with
analytic feature references; external NEST, Brian2, NEURON, and published-figure
traces remain a separate corpus-expansion task.

```python
from sc_neurocore.neurons.reference_traces import validate_all_reference_traces

reports = validate_all_reference_traces()
assert all(report.passed for report in reports)
```

See [Reference Trace Harness](../validation/reference_traces.md) for corpus
format, validation commands, and remaining WC-A1 scope.

::: sc_neurocore.neurons.reference_trace_contracts

::: sc_neurocore.neurons.reference_trace_io

::: sc_neurocore.neurons.reference_trace_runner

## Core SC Neurons (bitstream-capable)

| Class | Domain |
|-------|--------|
| `StochasticLIFNeuron` | Software simulation (fast) |
| `FixedPointLIFNeuron` | Bit-true Q8.8 hardware model |
| `HomeostaticLIFNeuron` | Self-regulating firing rate |
| `SCIzhikevichNeuron` | Rich dynamics (bursting, chattering) |
| `StochasticDendriticNeuron` | XOR dendritic processing |

::: sc_neurocore.neurons.base.BaseNeuron

::: sc_neurocore.neurons.stochastic_lif.StochasticLIFNeuron

::: sc_neurocore.neurons.fixed_point_lif.FixedPointLIFNeuron

::: sc_neurocore.neurons.sc_izhikevich.SCIzhikevichNeuron

::: sc_neurocore.neurons.homeostatic_lif.HomeostaticLIFNeuron

::: sc_neurocore.neurons.dendritic.StochasticDendriticNeuron

## Extended Model Library (`neurons/models/`)

### Integrate-and-Fire Variants (21)

| Model | File | Reference |
|-------|------|-----------|
| AdEx | `adex.py` | Brette & Gerstner 2005 |
| ExpIF | `expif.py` | Fourcaud-Trocme 2003 |
| Lapicque | `lapicque.py` | Lapicque 1907 |
| QIF | `quadratic_if.py` | Latham 2000 |
| GLIF (5 levels) | `glif.py` | Teeter 2018, Allen Institute |
| MAT | `mat.py` | Kobayashi 2009 |
| SFA | `sfa.py` | Benda & Herz 2003 |
| Stochastic IF | `stochastic_if.py` | Brunel & Hakim 1999 |
| Escape-rate | `escape_rate.py` | Gerstner 2000 |
| Fractional LIF | `fractional_lif.py` | Lundstrom 2008 |
| COBA LIF | `coba_lif.py` | Conductance-based |
| Perfect Integrator | `perfect_integrator.py` | Non-leaky IF |
| NLIF | `nlif.py` | Cubic nonlinearity |
| Adaptive Threshold | `adaptive_threshold_if.py` | Dynamic threshold |
| PLIF | `plif.py` | Fang 2021, learnable tau |
| Non-Resetting LIF | `non_resetting_lif.py` | Kobayashi 2009 |
| Gated LIF | `gated_lif.py` | Yao 2022, NeurIPS |
| Sigma-Delta | `sigma_delta.py` | Yoon 2017 |
| TC-LIF | `tc_lif.py` | AAAI 2024 |
| Benda-Herz | `benda_herz.py` | Benda 2003 |
| Integer QIF | `iqif.py` | Lo 2021, fixed-point |
| Complementary LIF | `clif.py` | ICML 2024, dual paths |
| K-LIF | `klif.py` | Learnable scaling |
| Inhibitory LIF | `ilif.py` | 2025, temporal inhibition |
| E-prop ALIF | `e_prop_alif.py` | Bellec 2020, eligibility |
| Izhikevich 2007 | `izhikevich2007.py` | Izhikevich 2007 biophysical |
| Energy LIF | `energy_lif.py` | Fardet 2020 |

### Biophysical / Conductance-Based (11)

| Model | File | Reference |
|-------|------|-----------|
| Hodgkin-Huxley | `hodgkin_huxley.py` | HH 1952 (Nobel Prize) |
| Connor-Stevens | `connor_stevens.py` | Connor 1977, A-type K+ |
| Wang-Buzsaki | `wang_buzsaki.py` | Wang 1996, FS interneuron |
| Pinsky-Rinzel | `pinsky_rinzel.py` | Pinsky 1994, 2-compartment |
| Destexhe | `destexhe_thalamic.py` | Destexhe 1993, T-current |
| Huber-Braun | `huber_braun.py` | Braun 1998, cold receptor |
| Gutkin-Ermentrout | `gutkin_ermentrout.py` | Gutkin 1998 |
| Traub-Miles | `traub_miles.py` | Traub 1991, hippocampal |
| Golomb FS | `golomb_fs.py` | Golomb 2007, Kv3 channels |
| Mainen-Sejnowski | `mainen_sejnowski.py` | Mainen 1996, axonal Na |
| Pospischil | `pospischil.py` | Pospischil 2008, 5 types |

### Oscillatory / Qualitative (7)

| Model | File | Reference |
|-------|------|-----------|
| FitzHugh-Nagumo | `fitzhugh_nagumo.py` | FitzHugh 1961 |
| Morris-Lecar | `morris_lecar.py` | Morris 1981 |
| Hindmarsh-Rose | `hindmarsh_rose.py` | HR 1984, chaotic bursting |
| Resonate-and-Fire | `resonate_and_fire.py` | Izhikevich 2001 |
| Balanced Resonate-and-Fire | `balanced_resonate_and_fire.py` | Higuchi et al. 2024 |
| Theta | `theta.py` | Ermentrout 1986 |
| FitzHugh-Rinzel | `fitzhugh_rinzel.py` | FitzHugh 1976, 3D |
| Terman-Wang | `terman_wang.py` | Terman 1995, LEGION |

### Bursting (5)

| Model | File | Reference |
|-------|------|-----------|
| Chay | `chay.py` | Chay 1985, pancreatic beta |
| Butera | `butera_respiratory.py` | Butera 1999, respiratory |
| Sherman-Rinzel-Keizer | `sherman_rinzel_keizer.py` | Sherman 1988 |
| Plant R15 | `plant_r15.py` | Plant 1981, Aplysia |
| Bertram Phantom | `bertram_phantom.py` | Bertram 2008 |
| Pernarowski | `pernarowski.py` | Pernarowski 1994 |

### Multi-Compartment (4)

| Model | File | Reference |
|-------|------|-----------|
| Hay L5 Pyramidal | `hay_l5.py` | Hay 2011, 3-compartment BAC firing |
| Booth-Rinzel | `booth_rinzel.py` | Booth 1995, bistable motoneuron |
| Dendrify | `dendrify.py` | Beniaguev 2022, active dendrite |
| TC-LIF | `tc_lif.py` | AAAI 2024, soma+dendrite |

### Synaptic (3)

Alpha, Synaptic (dual-exp), Tsodyks-Markram (STP)

### Map-Based / Discrete (6)

Rulkov, Chialvo, Courbage-Nekorkin, Medvedev, Ibarz-Tanaka, Cazelles

### Stochastic (4)

Poisson, Inhomogeneous Poisson, Galves-Locherbach, GLM (Pillow 2008)

### Population / Neural Mass (7)

Wilson-Cowan, Jansen-Rit (EEG), Wong-Wang (decision), Ermentrout-Kopell (exact mean-field), Amari (neural field), Wendling (extended JR, epilepsy EEG), Larter-Breakspear (TVB whole-brain)

### Hardware-Specific (9)

Loihi CUBA, Loihi 2, TrueNorth, BrainScaleS AdEx, SpiNNaker LIF, SpiNNaker2, DPI/DYNAP-SE, Akida, Sigma-Delta

### Rate Models (3)

McCulloch-Pitts (1943), Sigmoid Rate, Threshold-Linear (ReLU)

### Other (5)

SRM/SRM0 (kernel), McKean (piecewise FHN), Leaky-Compete-Fire (WTA), Prescott (Type I/II/III), Compte (NMDA working memory)

### Multi-Compartment (3)

Pinsky-Rinzel (2-comp), Booth-Rinzel (motoneuron), TC-LIF (soma+dendrite)

## PyTorch Training Cells (10)

Differentiable spiking neurons for surrogate gradient training:

| Cell | Module | Reference |
|------|--------|-----------|
| LIFCell | `training.snn_modules` | Standard LIF |
| IFCell | `training.snn_modules` | No leak |
| SynapticCell | `training.snn_modules` | Dual-exponential |
| ALIFCell | `training.snn_modules` | Bellec 2020 |
| RecurrentLIFCell | `training.snn_modules` | Orthogonal init |
| ExpIFCell | `training.snn_modules` | Exponential |
| AdExCell | `training.snn_modules` | Adaptive exponential |
| LapicqueCell | `training.snn_modules` | RC circuit |
| AlphaCell | `training.snn_modules` | Alpha synapse |
| SecondOrderLIFCell | `training.snn_modules` | Inertial term |
