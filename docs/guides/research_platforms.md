<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Research Hardware Platforms Guide

SC-NeuroCore currently registers **194 hardware profiles** across **38
platform classes**. The registry spans stable deployment targets, research
accelerators, and explicitly experimental substrate profiles, from FPGAs and
ASICs to organoid, molecular, and quantum-neuromorphic research targets.

This guide covers the research platform classes, their physical principles,
compilation considerations, and code examples.

## Table of Contents

1. [Cryogenic and Non-Volatile Platforms](#cryogenic-and-non-volatile-platforms)
2. [Biological, Electrochemical, and Wafer-Scale Platforms](#biological-electrochemical-and-wafer-scale-platforms)
3. [Memory-Centric and Quantum-Inspired Platforms](#memory-centric-and-quantum-inspired-platforms)
4. [Interconnect, Acoustic, Fluidic, and Space Platforms](#interconnect-acoustic-fluidic-and-space-platforms)
5. [Sovereign, Organic, and Magnonic Platforms](#sovereign-organic-and-magnonic-platforms)
6. [Adaptive Reliability Platforms](#adaptive-reliability-platforms)
7. [Custom Profile Registration](#custom-profile-registration)
8. [Platform Class Reference](#platform-class-reference)

---

## Cryogenic and Non-Volatile Platforms

### Superconducting / Cryogenic (3 profiles)

**Physics**: Single Flux Quantum (SFQ) logic operates at 4K with
Josephson junctions producing picosecond voltage pulses. Clock speeds
reach 100+ GHz with near-zero switching energy (~10⁻¹⁹ J per gate).

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `nist_sfq` | NIST | SFQ | 8-bit |
| `northrop_aqfp` | Northrop Grumman | AQFP | 8-bit |
| `mit_josephson` | MIT-LL | Josephson-JJ | 8-bit |

**Compilation notes**:
- Use 8-bit or narrower data widths (gate budgets are small)
- Overflow: saturate (no wrap — metastability risk at 4K)
- SFQ pulse timing requires careful pipeline depth

```python
from sc_neurocore.compiler.platforms import get_profile

p = get_profile("nist_sfq")
print(f"Class: {p.platform_class}")  # superconducting
print(f"Width: Q{p.data_width - p.fraction}.{p.fraction}")  # Q4.4
```

---

### Spintronic (2 profiles)

**Physics**: Spin-Transfer Torque (STT) and Spin-Orbit Torque (SOT)
MRAM use electron spin states for non-volatile storage. Enables
in-memory computation with ns switching and near-infinite endurance.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `everspin_stt` | Everspin | STT-MRAM | 8-bit |
| `samsung_sot` | Samsung | SOT-MRAM | 8-bit |

**Key advantage**: Non-volatile synaptic weights survive power cycles,
enabling instant-on neural inference without weight reload.

---

### Ferroelectric (2 profiles)

**Physics**: Ferroelectric FETs (FeFET) and FeRAM use polarisation
switching in hafnium oxide (HfO₂) for non-volatile CIM with
CMOS-compatible process integration.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `gf_fefet` | GlobalFoundries | FeFET-22FDX | 8-bit |
| `sk_hynix_feram` | SK Hynix | FeRAM | 8-bit |

---

### CGRA (3 profiles)

**Physics**: Coarse-Grained Reconfigurable Arrays provide FPGA-like
flexibility with ASIC-like efficiency. Reconfiguration happens at the
PE (Processing Element) level, not the LUT level.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `samsung_cgra` | Samsung | CGRA-v2 | 16-bit | 1 GHz |
| `qualcomm_cgra` | Qualcomm | NPU-CGRA | 8-bit | 2 GHz |
| `cadence_xtensa` | Cadence | Xtensa-NX | 16-bit | 1.5 GHz |

---

### 3D-Stacked (3 profiles)

**Physics**: 3D integration via through-silicon vias (TSVs), hybrid
bonding, or micro-bumps. Dramatically reduces interconnect length and
enables heterogeneous integration (logic + memory on same stack).

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `tsmc_soic` | TSMC | SoIC-3D | 16-bit | 2 GHz |
| `intel_foveros` | Intel | Foveros-3D | 16-bit | 3 GHz |
| `amd_3d_vcache` | AMD | 3D-V-Cache | 16-bit | 4.5 GHz |

---

### Edge MCU / TinyML (5 profiles)

**Physics**: Standard ARM Cortex-M and RISC-V microcontrollers with
hardware neural accelerators. Target: <1mW inference at the sensor edge.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `rp2040` | Raspberry Pi | RP2040 | 16-bit | 133 MHz |
| `esp32_s3` | Espressif | ESP32-S3 | 16-bit | 240 MHz |
| `stm32_h7` | STMicro | STM32H7 | 16-bit | 480 MHz |
| `nrf5340` | Nordic | nRF5340 | 16-bit | 128 MHz |
| `max78000` | Maxim/ADI | MAX78000 | 8-bit | 100 MHz |

```python
from sc_neurocore.compiler.platforms import get_profile

p = get_profile("max78000")
print(f"Built-in CNN accelerator: {p.notes}")
```

---

## Biological, Electrochemical, and Wafer-Scale Platforms

### Biological / Wetware (2 profiles)

**Physics**: Living organoid co-processors. FinalSpark uses human iPSC
neuron cultures as computational substrates. Cortical Labs (DishBrain)
demonstrated game-playing biological neural networks.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `finalspark_neuroplatform` | FinalSpark | Neuroplatform | 8-bit |
| `cortical_labs_dishbrain` | Cortical Labs | DishBrain | 8-bit |

**Compilation notes**:
- Output is stimulation protocol, not RTL
- Time constants are biological (ms–s scale)
- Stochastic behaviour — no deterministic guarantee

---

### Electrochemical / Memristive (3 profiles)

**Physics**: Electrochemical RAM (ECRAM) and Phase-Change RAM (PCRAM)
enable analog synaptic weight storage with tuneable conductance states.
Ideal for in-situ training.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `ibm_ecram` | IBM | ECRAM-AnalogAI | 8-bit |
| `samsung_pcram` | Samsung | PCRAM | 8-bit |
| `stanford_ecram` | Stanford | ECRAM-Research | 8-bit |

---

### Wafer-Scale (3 profiles)

**Physics**: Entire silicon wafers used as single chips. Cerebras WSE-3
has 4 trillion transistors, 900,000 cores. Tesla Dojo uses custom
die-to-die interconnect for distributed neural training.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `cerebras_wse3_ws` | Cerebras | WSE-3-WS | 16-bit | 1 GHz |
| `tesla_dojo3` | Tesla | Dojo-3 | 16-bit | 2 GHz |
| `tachyum_prodigy` | Tachyum | Prodigy-2nm | 16-bit | 5.5 GHz |

---

### Analog Mixed-Signal (2 profiles)

**Physics**: Compute-at-sensor architectures. Aspinity AML100 performs
analog feature extraction before ADC, eliminating >90% of data movement.
Renesas AnalogAI integrates analog MAC arrays.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `aspinity_aml100` | Aspinity | AML100 | 8-bit |
| `renesas_analog_ai` | Renesas | AnalogAI | 8-bit |

---

## Memory-Centric and Quantum-Inspired Platforms

### RRAM / Memristive Crossbar (3 profiles)

**Physics**: Resistive RAM crossbar arrays perform in-situ matrix-vector
multiplication using Ohm's law (V=IR) and Kirchhoff's current law.
Weebit Nano demonstrates 200 TOPS/W, licensed to TI and Onsemi.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `weebit_reram` | Weebit Nano | ReRAM-ACiM | 8-bit |
| `crossbar_rram` | Crossbar | ReRAM-1T1R | 8-bit |
| `adesto_cbram` | Adesto | CBRAM | 8-bit |

**Compilation notes**:
- Weights stored as conductance states (analog)
- Use drift compensation (§32) for long-term reliability
- 8-bit is typical due to conductance resolution limits

---

### SRAM Compute-in-Memory (2 profiles)

**Physics**: Digital CIM using standard 6T SRAM cells modified for
in-memory Boolean/MAC operations. TSMC and Samsung offer foundry-
qualified CIM macro IP at N7 and SF3 nodes.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `tsmc_cim_n7` | TSMC | CIM-N7 | 8-bit | 1 GHz |
| `samsung_cim_sf3` | Samsung | CIM-SF3 | 8-bit | 900 MHz |

---

### Cryogenic CMOS (2 profiles)

**Physics**: Standard CMOS operated at 4K for quantum chip control.
Device physics change significantly at cryogenic temperatures
(threshold shift, carrier freeze-out, reduced leakage).

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `intel_horse_ridge` | Intel | Horse-Ridge-II | 16-bit | 6 GHz |
| `google_cryo_ctrl` | Google | Cryo-Controller | 16-bit | 4 GHz |

---

### DNA / Molecular (2 profiles)

**Physics**: DNA base-pair gated logic and perovskite-DNA hybrid
synaptic devices. Microsoft demonstrated enzymatic DNA synthesis for
archival compute. ASU achieved CMOS-DNA integration.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `microsoft_dna_store` | Microsoft | DNA-Storage | 8-bit |
| `asu_dna_perovskite` | ASU | DNA-Perovskite | 8-bit |

---

### Quantum Neuromorphic (2 profiles)

**Physics**: Quantum reservoir computing and quantum SNNs using
superconducting transmon qubits (IBM) or trapped-ion all-to-all
connectivity (IonQ).

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `ibm_qnn` | IBM | Quantum-NN | 16-bit |
| `ionq_trapped_ion` | IonQ | Trapped-Ion-QNN | 16-bit |

---

## Interconnect, Acoustic, Fluidic, and Space Platforms

### Optical Interconnect / CPO (2 profiles)

**Physics**: Silicon photonic I/O chiplets replacing electrical
interconnect with optical. Ayar Labs TeraPHY delivers 8 Tbps
bidirectional, UCIe-compatible. Enables rack-scale SNN.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `ayar_teraphy` | Ayar Labs | TeraPHY | 16-bit | 25 GHz |
| `intel_cpo` | Intel | CPO | 16-bit | 20 GHz |

---

### Acoustic / Phononic (2 profiles)

**Physics**: Acoustic wave reservoir computing using MEMS resonator
arrays. Mechanical nonlinearity provides natural activation functions.
Zero digital power consumption during inference.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `mit_phononic` | MIT | Phononic-NN | 8-bit |
| `caltech_mems_nn` | Caltech | MEMS-NN | 8-bit |

---

### Fluidic / Microfluidic (2 profiles)

**Physics**: Droplet-based and pressure-driven bistable logic gates for
chemical/biological neural computation. Lab-on-chip applications for
in-vivo diagnostic neural inference.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `stanford_microfluidic` | Stanford | µFluidic-NN | 8-bit |
| `eth_fluidic_logic` | ETH Zurich | Fluidic-Logic | 8-bit |

---

### Space-Qualified (4 profiles)

**Physics**: Radiation-hardened processors qualified for TID (Total
Ionizing Dose) and SEE (Single Event Effects) tolerance. Deployed on
ISS, Mars rovers, and deep-space missions.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `bae_rad750_sq` | BAE Systems | RAD750 | 32-bit | 200 MHz |
| `seakr_sbc` | SEAKR | SBC-SpaceAI | 16-bit | 400 MHz |
| `vorago_va10820` | Vorago | VA10820 | 16-bit | 100 MHz |
| `frontgrade_leon5` | Frontgrade | LEON5-FT | 32-bit | 250 MHz |

**Compilation notes**:
- Use TMR (Triple Modular Redundancy) via §7 SEU hardening
- Combine with §50 fault tree for DO-254 Level A certification
- Use supply chain risk scorer (§36) for ITAR compliance

```python
from sc_neurocore.compiler.platforms import get_profile
from sc_neurocore.compiler.intelligence import (
    score_supply_chain_risk, generate_fault_tree,
)

p = get_profile("bae_rad750_sq")
risk = score_supply_chain_risk("bae_rad750_sq")
ft = generate_fault_tree("sc_lif", {"v": "-(v)/tau + I"})
print(f"Risk: {risk.overall_risk}")
print(f"Fault tree events: {len(ft.basic_events)}")
```

---

## Sovereign, Organic, and Magnonic Platforms

### Magnonic / Skyrmion (3 profiles)

**Physics**: Magnetic skyrmions are topologically protected spin textures
that behave as quasi-particles. Their nonlinear dynamics, ultra-low
switching energy (~10⁻²⁰ J), and emergent collective behaviour make them
ideal reservoir computing substrates. Spin-wave (magnon) interference
provides natural nonlinear activation without transistors.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `tum_skyrmion` | TU Munich | SkyANN-v1 | 8-bit |
| `kaist_spinwave` | KAIST | SpinWave-RC | 8-bit |
| `imec_mtj_reservoir` | imec | MTJ-Reservoir | 8-bit |

**Compilation notes**:
- Reservoir computing mode: only readout layer trained
- Use 8-bit widths (sub-fJ switching limits precision)
- Ideal for temporal signal processing (EEG, vibration)

```python
from sc_neurocore.compiler.platforms import get_profile

p = get_profile("tum_skyrmion")
print(f"Class: {p.platform_class}")  # magnonic
```

---

### Organic Bioelectronic (2 profiles)

**Physics**: Organic Electrochemical Transistors (OECTs) use mixed
ionic/electronic conduction in PEDOT:PSS to create artificial synapses
that operate in aqueous environments. Enables direct neural tissue
interfacing for in-vivo bioelectronic medicine.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `cambridge_oect` | Cambridge | OECT-Synapse | 8-bit |
| `linkoping_organic` | Linköping | Organic-NN | 8-bit |

**Compilation notes**:
- Output is stimulation/recording protocol, not RTL
- Wet computing: ionic time constants (100 ms – 10 s)
- Biocompatible — no heavy metals
- Use HIL calibration (§62) for analog drift compensation

---

### RISC-V Sovereign AI (5 profiles)

**Physics**: Open-ISA RISC-V processors with vector AI extensions.
No ITAR/EAR restrictions. Enables data-sovereign and export-control-free
deployment for government, defence, and critical infrastructure.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `sifive_x280_ai` | SiFive | X280-AI | 16-bit | 2 GHz |
| `esperanto_et_soc` | Esperanto | ET-SoC-1 | 8-bit | 1 GHz |
| `ventana_veyron_ai` | Ventana | Veyron-V2 | 16-bit | 3.6 GHz |
| `tenstorrent_ascalon` | Tenstorrent | Ascalon | 16-bit | 4 GHz |
| `andes_ax45mpv` | Andes | AX45MPV | 16-bit | 1.5 GHz |

**Compilation notes**:
- Open ISA: no license fees or export restrictions
- Use UCIe protocol mapper (§64) for chiplet-based designs
- Combine with SBOM generator (§61) for EU CRA compliance
- Supply chain risk score (§36) will be LOW for all RISC-V

```python
from sc_neurocore.compiler.platforms import get_profile
from sc_neurocore.compiler.intelligence import score_supply_chain_risk

p = get_profile("sifive_x280_ai")
risk = score_supply_chain_risk("sifive_x280_ai")
print(f"Open ISA — Risk: {risk.overall_risk}")
```

---

## Custom Profile Registration

### Method 1: TOML File (Zero Code Changes)

Create a `.toml` file with your custom profiles:

```toml
# my_lab_profiles.toml
[[profile]]
name = "my_custom_asic"
vendor = "UniversityLab"
family = "ASIC-v1"
platform_class = "asic"
data_width = 24
fraction = 12
overflow = "saturate"
rounding = "nearest"
max_freq_mhz = 500
notes = "Custom 24-bit ASIC for cortical simulation."

[[profile]]
name = "my_fpga_board"
vendor = "UniversityLab"
family = "Custom-FPGA"
platform_class = "fpga"
data_width = 16
fraction = 8
overflow = "wrap"
rounding = "truncate"
dsp_block = "DSP48E2"
dsp_mult_a = 18
dsp_mult_b = 27
max_freq_mhz = 250
notes = "Custom FPGA board with Xilinx UltraScale+."
```

```python
from sc_neurocore.compiler.intelligence import load_profiles_from_toml

loaded = load_profiles_from_toml("my_lab_profiles.toml")
print(f"Loaded: {loaded}")  # ['my_custom_asic', 'my_fpga_board']
```

### Method 2: Runtime Discovery Hook (Vendor SDK)

```python
from sc_neurocore.compiler.intelligence import (
    register_platform_hook, discover_platforms,
)
from sc_neurocore.compiler.platforms import HardwareProfile

def vendor_discovery():
    """Auto-detect connected hardware and return profiles."""
    return [HardwareProfile(
        name="detected_board", vendor="AutoDetect",
        family="Board-v1", platform_class="fpga",
        data_width=16, fraction=8,
        overflow="saturate", rounding="nearest",
    )]

register_platform_hook(vendor_discovery)
discovered = discover_platforms()
```

### Method 3: Auto-Construct from Spec Sheet

```python
from sc_neurocore.compiler.platforms import HardwareProfile

# Any future chip — one function call:
p = HardwareProfile.from_constraints(
    "my_2030_chip",
    vendor="FutureVendor",
    platform_class="custom",
    max_power_budget_mw=5,
    min_precision_bits=8,
)
# Automatically registered and ready to compile against
```

---

## Frontier Paradigm Platforms

### Wetware / Biological (2 profiles)

**Physics**: Living organoid co-processors interfaced via high-density Multi-Electrode Arrays (MEAs). FinalSpark uses living spherical brain organoids for closed-loop biocomputing. Cortical Labs (DishBrain) demonstrates embodied intelligence via active biological neural cultures.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `finalspark_neuroplatform` | FinalSpark | Neuroplatform | 8-bit |
| `cortical_labs_dishbrain` | Cortical Labs | DishBrain | 8-bit |

**Compilation notes**:
- Uses the `map_wetware_mea` (§79) intelligence feature to translate SNN topology into spatio-temporal stimulations.

---

### Molecular / Chemical (2 profiles)

**Physics**: Computation and storage within synthetic DNA base pairs and enzymatic reactions. Biomemory maps ultra-high-density data (such as billion-parameter network weights) into physical DNA storage at near-zero static energy. Catalog utilizes parallel search within DNA liquid solutions.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `biomemory_dna` | Biomemory | DNA-Storage | 8-bit |
| `catalog_dna_compute` | Catalog | Shannon | 8-bit |

---

### Reversible / Adiabatic (2 profiles)

**Physics**: Operating at the Landauer limit of energy dissipation. Logic gates (like Toffoli and Fredkin) preserve information perfectly, allowing energy to be recovered rather than dissipated as heat. Requires multi-phase trapezoidal resonant clocking (§82).

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `superconducting_aqfp` | Yokohama Univ | AQFP | 16-bit |
| `scrl_logic` | Generic | SCRL | 16-bit |

---

### Microfluidic / Mechanical (2 profiles)

**Physics**: Fluid dynamics and nonlinear mechanical oscillators acting as physical computational substrates. Nanofluidic 2D ionic channels physically emulate biological ion exchange using water/ion flow.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `nanofluidic_logic` | EPFL | Ion-Channel | 8-bit |
| `mems_neuromorphic` | Generic | MEMS-Resonator | 8-bit |

---

Representative platform groups documented in this guide:

| ID | Class Name | Profiles | Description | Target |
|---|---|---|---|---|
| 1 | `fpga` | 27 | Traditional FPGAs (Xilinx/Intel) | Production |
| 2 | `asic` | 13 | Custom silicon | Production |
| 3 | `neuromorphic` | 11 | SNN chips (Loihi, TrueNorth) | Production |
| 4 | `pim` | 8 | Processing-in-memory | Production |
| 5 | `quantum` | 6 | Superconducting/Ion/Optical | Research |
| 6 | `optical` | 6 | Silicon photonics | Pre-production |
| 7 | `analog` | 6 | Continuous-time analog | Research |
| 8 | `memristive` | 6 | Crossbar arrays | Research |
| 9 | `emerging` | 3 | Hybrid/novel | Research |
| 10 | `superconducting` | 3 | SFQ/AQFP at 4K | Research |
| 11 | `spintronic` | 2 | STT/SOT-MRAM | Pre-production |
| 12 | `ferroelectric` | 2 | FeFET/FeRAM | Pre-production |
| 13 | `cgra` | 3 | Reconfigurable array | Production |
| 27 | `fluidic` | 2 | Microfluidic logic | Research |
| 28 | `space_qualified` | 4 | Rad-hard processors | Deployed |
| 29 | `magnonic` | 3 | Skyrmion/spin-wave | Research |
| 30 | `organic_bioelectronic` | 2 | OECT wet computing | Research |
| 31 | `risc_v_sovereign` | 5 | Open ISA AI cores | Production |
| 32 | `thermodynamic` | 2 | EBM thermal equilibration | Research |
| 33 | `probabilistic` | 2 | p-Bit sMTJ | Research |
| 34 | `polariton` | 2 | Bose-Einstein condensate | Research |
| 35 | `metamaterial` | 2 | Passive wave propagation | Research |
| 36 | `wetware` | 2 | Living organoids (MEA) | Research |
| 37 | `molecular` | 2 | DNA / Enzymatic compute | Research |
| 38 | `reversible` | 2 | Adiabatic / Landauer limit | Research |
| 39 | `microfluidic` | 2 | Nanofluidic / MEMS | Research |
| | **Total** | **191** | | |

## Further Reading

- [Hardware Profiles Guide](hardware_profiles.md) — full profile tables
- [Compiler Intelligence Guide](compiler_intelligence.md) — all 76 features
- [Platform Extensibility Guide](platform_extensibility.md) — TOML + hook + from_constraints
- [Precision Modes Guide](precision_modes.md) — Q-format modes
- [Deployment Guide](deployment_guide.md) — constraints, TCL, bitstream
