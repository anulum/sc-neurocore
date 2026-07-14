<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Hardware Platform Profiles

SC-NeuroCore ships with **194 pre-configured hardware profiles** across **38
platform classes** and **121 vendors**, covering FPGA vendors (including
rad-hard and eFPGA), AI accelerator, DSP processor, neuromorphic chip, ASIC
target, photonic/optical compute, processing-in-memory, chiplet/UCIe,
automotive/edge AI, sovereign/defence, superconducting/cryogenic, spintronic,
ferroelectric compute-in-memory, CGRA, 3D-stacked, edge MCU/TinyML, RISC-V AI,
emerging compute paradigm, and simulation reference. Each profile encodes the
optimal fixed-point configuration (bit width, fraction, overflow handling,
rounding semantics) for its target platform.

## Registry Architecture

The public API remains `sc_neurocore.compiler.platforms`: callers use
`HardwareProfile`, `get_profile()`, `list_profiles()`, and
`list_profile_names()` without depending on the source layout. The historical
`cmos_profiles` module is a registration facade. Five private modules own the
FPGA, conventional-accelerator, embedded-processor, architecture-paradigm, and
ASIC/simulation profile families.

The facade invokes those private registrars in the established sequence. This
preserves all 72 CMOS-family profile values and registry insertion order while
the registry continues to reject accidental duplicate names. Importing or
reloading a private definition module alone has no registration side effect.

## Quick Start

```bash
# List all 194 hardware profiles
python -m sc_neurocore.neurons platforms

# Compile a LIF neuron for Intel Loihi 2 (auto-selects Q12.12, wrap overflow)
python -m sc_neurocore.neurons compile lif --target loihi2 -o sc_lif_loihi.v

# Compile for Xilinx Artix-7 with IEEE 754 banker's rounding
python -m sc_neurocore.neurons compile lif --target artix7 --rounding bankers -o sc_lif.v

# Compile for safety-critical ASIC with overflow trapping
python -m sc_neurocore.neurons compile lif --target asic_custom -o sc_lif_safe.v
```

## Python API

```python
from sc_neurocore.compiler.platforms import get_profile, list_profiles
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

# Look up a hardware profile
profile = get_profile("loihi2")
print(profile.vendor)          # "Intel"
print(profile.family)          # "Loihi 2"
print(profile.q_format_label)  # "Q11.12"
print(profile.data_width)      # 24
print(profile.fraction)        # 12
print(profile.overflow)        # "wrap"
print(profile.rounding)        # "truncate"

# Compile with profile settings
neuron = UniversalNeuron.from_schema("lif")
verilog = neuron.to_verilog(
    module_name="sc_lif",
    data_width=profile.data_width,
    fraction=profile.fraction,
    overflow=profile.overflow,
    rounding=profile.rounding,
)

# List all FPGA profiles
for p in list_profiles(platform_class="fpga"):
    print(f"{p.name:18s} {p.q_format_label:10s} {p.vendor}")

# Filter by vendor
for p in list_profiles(vendor="Xilinx"):
    print(f"{p.name:18s} {p.family}")
```

## Supported Platforms

### CMOS/FPGA Platforms (31 profiles)

| Profile | Vendor | Family | Format | Bits | DSP Block | DSP Width |
|---------|--------|--------|--------|:----:|-----------|-----------|
| `spartan6` | Xilinx | Spartan-6 | Q9.9 | 18 | DSP48A1 | 18×18 |
| `artix7` | Xilinx | Artix-7 | Q9.9 | 18 | DSP48E1 | 25×18 |
| `kintex7` | Xilinx | Kintex-7 | Q9.9 | 18 | DSP48E1 | 25×18 |
| `ultrascale` | Xilinx | UltraScale | Q18.18 | 36 | DSP48E2 | 27×18 |
| `ultrascale_plus` | Xilinx | UltraScale+ | Q18.18 | 36 | DSP48E2 | 27×18 |
| `versal` | Xilinx | Versal | Q12.12 | 24 | DSP58 | 27×24 |
| `cyclone_v` | Intel | Cyclone V | Q9.9 | 18 | Variable | 18×18 |
| `cyclone_10` | Intel | Cyclone 10 | Q9.9 | 18 | Variable | 18×18 |
| `arria10` | Intel | Arria 10 | Q14.13 | 27 | Variable | 27×27 |
| `stratix10` | Intel | Stratix 10 | Q14.13 | 27 | Variable | 27×27 |
| `agilex` | Intel | Agilex | Q14.13 | 27 | Variable | 27×27 |
| `ecp5` | Lattice | ECP5 | Q9.9 | 18 | MULT18X18D | 18×18 |
| `crosslink_nx` | Lattice | CrossLink-NX | Q9.9 | 18 | DSP | 18×18 |
| `certuspro_nx` | Lattice | CertusPro-NX | Q9.9 | 18 | DSP | 18×18 |
| `ice40` | Lattice | iCE40 | Q8.8 | 16 | SB_MAC16 | 16×16 |
| `gowin` | Gowin | GW1N/GW2A | Q9.9 | 18 | MULT18X18 | 18×18 |
| `efinix` | Efinix | Trion/Titanium | Q5.5 | 10 | MULT | 10×9 |
| `polarfire` | Microchip | PolarFire | Q9.9 | 18 | MACC | 18×18 |
| `smartfusion2` | Microchip | SmartFusion2 | Q9.9 | 18 | MACC | 18×18 |
| `achronix` | Achronix | Speedster7t | Q12.12 | 24 | MLP | 24×24 |
| `quicklogic` | QuickLogic | EOS S3 | Q4.4 | 8 | LUT-based | N/A |
| `alveo` | AMD/Xilinx | Alveo U50/U200 | Q18.18 | 36 | DSP48E2 | 27×18 |
| `nexus` | Lattice | Nexus (LIFCL) | Q9.9 | 18 | DSP | 18×18 |
| `polarfire_soc` | Microchip | PolarFire SoC | Q9.9 | 18 | MACC | 18×18 |
| `avant` | Lattice | Avant | Q9.9 | 18 | DSP | 18×18 |
| `nanoxplore` | NanoXplore | NG-Ultra | Q9.9 | 18 | DSP | 18×18 |
| `rtg4` | Microchip | RTG4 | Q9.9 | 18 | MACC | 18×18 |
| `kintex_us_rt` | Xilinx | Kintex US RT | Q18.18 | 36 | DSP48E2 | 27×18 |
| `speedcore` | Achronix | Speedcore eFPGA | Q12.12 | 24 | MLP | 24×24 |
| `eflx` | Flex Logix | EFLX eFPGA | Q8.8 | 16 | LUT-based | N/A |
| `menta_efpga` | Menta | Origami eFPGA | Q8.8 | 16 | LUT-based | N/A |

### Neuromorphic Chips (7 profiles)

| Profile | Vendor | Chip | Format | Bits | Overflow | Rounding | Notes |
|---------|--------|------|--------|:----:|----------|----------|-------|
| `loihi2` | Intel | Loihi 2 | Q12.12 | 24 | **wrap** | truncate | 24-bit membrane; hardware wraps on overflow |
| `truenorth` | IBM | TrueNorth | Q1.7 | 8 | saturate | truncate | 1-bit stochastic neurons |
| `brainscales2` | Heidelberg | BrainScaleS-2 | Q3.4 | 8 | **wrap** | truncate | Analog accelerated neuro; 1000× bio realtime |
| `spinnaker2` | SpiNNcloud | SpiNNaker 2 | Q15.16 | 32 | **wrap** | truncate | ARM-based massively parallel GALS |
| `akida` | BrainChip | Akida 2.0 | Q1.7 | 8 | saturate | truncate | Event-driven neural processor |
| `dynap_se2` | SynSense | DYNAP-SE2 | Q8.8 | 16 | saturate | truncate | Mixed-signal neuromorphic |
| `xylo` | SynSense | Xylo | Q8.8 | 16 | saturate | truncate | Digital SNN processor |

### ASIC Targets (3 profiles)

| Profile | Description | Format | Overflow | Rounding |
|---------|-------------|--------|----------|----------|
| `asic_16` | Generic 16-bit standard cell | Q8.8 | saturate | nearest |
| `asic_32` | Generic 32-bit standard cell | Q16.16 | saturate | nearest |
| `asic_custom` | Safety-critical (DO-254/IEC 61508) | Q12.12 | **trap** | **bankers** |

### Simulation References (2 profiles)

| Profile | Description | Format |
|---------|-------------|--------|
| `sim_q88` | Icarus Verilog Q8.8 reference | Q8.8 |
| `sim_q1616` | Icarus Verilog Q16.16 gold standard | Q16.16 |

### AI / ML Accelerators (15 profiles)

| Profile | Vendor | Family | Format | Bits | Overflow | Rounding |
|---------|--------|--------|--------|:----:|----------|----------|
| `tpu` | Google | TPU v4/v5 | Q8.8 | 16 | saturate | nearest |
| `cerebras_wse` | Cerebras | WSE-3 | Q8.8 | 16 | saturate | nearest |
| `graphcore_ipu` | Graphcore | IPU Mk2/Bow | Q8.8 | 16 | saturate | nearest |
| `tenstorrent` | Tenstorrent | Grayskull | Q8.8 | 16 | saturate | truncate |
| `ethos_u` | ARM | Ethos-U55/U65 | Q4.4 | 8 | saturate | nearest |
| `hexagon` | Qualcomm | Hexagon HVX | Q8.8 | 16 | saturate | nearest |
| `apple_ane` | Apple | Neural Engine | Q8.8 | 16 | saturate | nearest |
| `hailo8` | Hailo | Hailo-8/8L | Q4.4 | 8 | saturate | nearest |
| `kneron` | Kneron | KL730 | Q4.4 | 8 | saturate | nearest |
| `groq_tsp` | Groq | TSP (LPU) | Q8.8 | 16 | saturate | nearest |
| `jetson` | NVIDIA | Jetson Orin | Q4.4 | 8 | saturate | nearest |
| `habana_gaudi` | Intel | Habana Gaudi 2/3 | Q8.8 | 16 | saturate | nearest |
| `drp_ai` | Renesas | RZ/V2H DRP-AI | Q4.4 | 8 | saturate | nearest |
| `imx500` | Sony | IMX500/IMX501 | Q4.4 | 8 | saturate | truncate |
| `samsung_npu` | Samsung | Exynos NPU | Q8.8 | 16 | saturate | nearest |

### DSP Processors (3 profiles)

| Profile | Vendor | Family | Format | Bits |
|---------|--------|--------|--------|:----:|
| `sharc` | Analog Devices | SHARC SC5xx | Q16.16 | 32 |
| `c6000` | Texas Instruments | C66x/C67x | Q16.16 | 32 |
| `ceva_xc` | CEVA | CEVA-XC/X2 | Q8.8 | 16 |

### Emerging Compute (4 profiles)

| Profile | Vendor | Paradigm | Format | Bits |
|---------|--------|----------|--------|:----:|
| `photonic` | Lightmatter | Optical matrix multiply | Q4.4 | 8 |
| `riscv_fpga` | SiFive | RISC-V + FPGA fabric | Q8.8 | 16 |
| `in_memory` | Mythic/Syntiant | Analog in-memory | Q4.4 | 8 |
| `quantum_hybrid` | IBM/Custom | Quantum-classical | Q16.16 | 32 |

## Overflow Modes

The compiler supports three overflow handling strategies, configured per-profile
or via `--overflow` on the CLI:

### Saturate (default)

Clamps next-state values to the representable range `[min, max]`:

```verilog
// Generated for overflow="saturate"
wire signed [16:0] v_raw = v_reg + dv;
wire signed [15:0] v_next =
    (v_raw > 17'sd32767)  ? 16'sd32767 :
    (v_raw < (-17'sd32768)) ? (-16'sd32768) :
    v_raw[15:0];
```

**Use for:** Most designs. Prevents catastrophic wrap-around spikes.

### Wrap

Two's complement wrap-around — the hardware simply takes the lower bits:

```verilog
// Generated for overflow="wrap"
wire signed [16:0] v_raw = v_reg + dv;
wire signed [15:0] v_next = v_raw[15:0];
```

**Use for:** Intel Loihi 2 compatibility (their hardware wraps natively).
The model must be designed to stay within range; wrapping is undefined behaviour
for neuron dynamics.

### Trap

Emits a `$fatal` assertion guarded by `synthesis translate_off`, meaning:
- **Simulation**: halts immediately on overflow with a diagnostic message
- **Synthesis**: zero hardware cost (assertion is stripped)

```verilog
// Generated for overflow="trap"
wire signed [15:0] v_next = v_raw[15:0];
// synthesis translate_off
always @(*) if (v_raw > 17'sd32767 || v_raw < (-17'sd32768))
    $fatal(1, "OVERFLOW TRAP: v_raw=%0d", v_raw);
// synthesis translate_on
```

**Use for:** Safety-critical systems (DO-254, IEC 61508) where overflow
must be detected during verification.

## Rounding Modes

The compiler supports four rounding strategies for fixed-point multiplication
truncation:

### Truncate (default)

Simple arithmetic right shift — floors towards -∞:

```verilog
wire signed [15:0] _t0 = (_mul0 >>> 8);
```

**Properties:** Zero additional logic. Systematic negative bias of 0.5 LSB.

### Nearest

Adds 0.5 LSB before truncation — round half away from zero:

```verilog
wire signed [31:0] _rnd_half1 = _mul0 + 128;  // 1 << (frac-1)
wire signed [15:0] _t0 = (_rnd_half1 >>> 8);
```

**Properties:** ±0.5 LSB maximum error. Small additional logic (1 adder).

### Bankers (IEEE 754)

Round half to even — eliminates statistical bias when accumulating many values:

```verilog
wire signed [31:0] _rnd_biased1 = _mul0 + 128;
wire _rnd_guard1 = (_mul0[7:0] == 128);  // exact half?
wire signed [15:0] _t0 =
    (_rnd_biased1 >>> 8) & ((_rnd_guard1) ? ~16'd1 : {16{1'b1}});
```

**Properties:** Zero expected bias over many operations. Small logic overhead.
Required by IEEE 754 and recommended for ASIC designs.

### Stochastic

Adds random fractional bits (from an LFSR) before truncation:

```verilog
wire signed [31:0] _rnd_stoch1 = _mul0 + {{24{1'b0}}, _lfsr[7:0]};
wire signed [15:0] _t0 = (_rnd_stoch1 >>> 8);
```

**Properties:** Zero expected bias. Each truncation independently random.
Requires an external LFSR module named `_lfsr`. Provably optimal for
long-running simulations (eliminates systematic drift).

## DSP Block Utilisation Guide

The key insight behind hardware profiles is **DSP-native format selection**.
Every FPGA has hard multiplier blocks with specific widths. Choosing a Q-format
that matches the native width means:

- **Zero wasted DSP bits** — every multiplier bit carries signal, not padding
- **1 DSP per multiply** — no multi-cycle or multi-DSP decomposition needed
- **Maximum clock frequency** — combinational paths stay within one DSP

| DSP Width | Optimal Format | Used by |
|:---------:|:-------------:|---------|
| 10×9 | Q5.5 (10-bit) | Efinix Trion/Titanium |
| 16×16 | Q8.8 (16-bit) | Lattice iCE40 |
| 18×18 | **Q9.9 (18-bit)** | Xilinx 7-series, Intel Cyclone, Lattice ECP5, Gowin, Microchip |
| 24×24 | Q12.12 (24-bit) | Achronix Speedster7t, Xilinx Versal (DSP58) |
| 27×27 | Q14.13 (27-bit) | Intel Arria 10, Stratix 10, Agilex |
| 27×18 | Q18.18 (36-bit) | Xilinx UltraScale/UltraScale+ (DSP48E2) |

> **Q9.9 is the universal format**: it runs natively on Xilinx, Intel, Lattice,
> Gowin, and Microchip — covering >95% of the FPGA market.

## HardwareProfile Dataclass Reference

```python
@dataclass(frozen=True)
class HardwareProfile:
    name: str                 # e.g. "loihi2"
    vendor: str               # e.g. "Intel"
    family: str               # e.g. "Loihi 2"
    platform_class: str       # "fpga" | "neuromorphic" | "accelerator" | "dsp" | "asic" | "emerging" | "simulation"
    data_width: int           # Total bit width
    fraction: int             # Fractional bits
    signed: bool = True       # True=signed, False=unsigned Q-format
    overflow: str = "saturate"  # "saturate" | "wrap" | "trap"
    rounding: str = "truncate"  # "truncate" | "nearest" | "bankers" | "stochastic"
    dsp_block: str = ""       # DSP macro name (e.g. "DSP48E2")
    dsp_mult_a: int = 0       # A-port width
    dsp_mult_b: int = 0       # B-port width
    max_freq_mhz: int = 0     # Typical max clock
    notes: str = ""           # Human-readable rationale
```

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `int_bits` | `int` | Integer bits (excluding sign if signed) |
| `q_format_label` | `str` | e.g. `"Q9.9"` or `"UQ8.8"` |
| `max_value` | `float` | Maximum representable value |
| `min_value` | `float` | Minimum representable value |
| `resolution` | `float` | Smallest step (1/2^fraction) |

## CLI Reference

### `platforms` — List all targets

```bash
python -m sc_neurocore.neurons platforms
```

### `compile --target` — Target-specific compilation

```bash
# --target sets defaults; --precision/--overflow/--rounding override
python -m sc_neurocore.neurons compile lif --target artix7 -o lif.v
python -m sc_neurocore.neurons compile lif --target loihi2 --overflow saturate -o lif.v
python -m sc_neurocore.neurons compile lif --target asic_custom --rounding stochastic -o lif.v
```

### Precedence

1. Explicit `--overflow` / `--rounding` flags (highest priority)
2. Target profile defaults
3. Global defaults (`saturate`, `truncate`)

## Bus Interface Generation

Auto-generate SoC bus wrappers for instant integration with Zynq, Nios, or
RISC-V SoCs. Parameters become memory-mapped registers; spike output becomes
an interrupt line.

```python
from sc_neurocore.hdl_gen.bus_interface import generate_bus_wrapper

# AXI4-Lite for Xilinx Zynq
axi = generate_bus_wrapper("sc_lif",
    params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
    bus="axi_lite",
)

# Wishbone B4 for LiteX / RISC-V
wb = generate_bus_wrapper("sc_lif",
    params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
    bus="wishbone",
)
```

## Mixed-Precision Per Variable

Different state variables can use different bit widths within the same module:

```python
from sc_neurocore.compiler.mixed_precision import (
    MixedPrecisionSpec, PrecisionConfig, solve_precision, from_preset,
)

# Dict API — explicit control
spec = MixedPrecisionSpec({
    "v": PrecisionConfig(16, 8),   # Q8.8 for membrane voltage
    "u": PrecisionConfig(8, 4),    # Q4.4 for recovery variable
})
print(spec.total_bits)  # 24 (vs 32 for uniform Q8.8)

# Constraint solver — automatic from bounds
spec = solve_precision(
    bounds={"v": (-128, 127), "u": (-10, 10)},
    min_resolution={"v": 0.01, "u": 0.1},
    max_total_bits=24,
)

# Preset shorthand
spec = from_preset({"v": "q88", "u": "q44"})
```

## Static Analysis (Guard Bits, Overflow Proof, SVA)

```python
from sc_neurocore.compiler.static_analysis import (
    compute_guard_bits, prove_no_overflow, generate_sva,
)

# Guard bits needed for safe accumulation
bits = compute_guard_bits("a + b + c + d")  # → 2

# Formal overflow proof via interval arithmetic
result = prove_no_overflow(
    "-(v - v_rest) / tau_m + R * I / C",
    bounds={"v": (-128, 127), "v_rest": (-65, -65),
            "tau_m": (10, 10), "R": (1, 1), "I": (0, 100), "C": (1, 1)},
    data_width=16, fraction=8,
)
assert result.proven_safe  # Mathematical guarantee

# SystemVerilog Assertions for DO-254 certification
sva = generate_sva(["v", "u"], data_width=16, fraction=8,
                   module_name="sc_izh")
```

## Cryogenic and Non-Volatile Platform Classes

### Superconducting / Cryogenic

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `nist_sfq` | NIST | SFQ | 8-bit | 100 GHz |
| `northrop_aqfp` | Northrop Grumman | AQFP | 8-bit | 5 GHz |
| `josephson_jj` | Research | Josephson | 8-bit | 50 GHz |

### Spintronic / MRAM

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `everspin_stt_mram` | Everspin | STT-MRAM | 8-bit |
| `samsung_sot_mram` | Samsung | SOT-MRAM | 8-bit |

### Ferroelectric Compute-in-Memory

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `gf_fefet` | GlobalFoundries | FeFET-22FDX | 8-bit |
| `sk_hynix_feram` | SK Hynix | FeRAM | 8-bit |

### CGRA (Coarse-Grained Reconfigurable)

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `samsung_cgra` | Samsung | CGRA-NPU | 16-bit | 1 GHz |
| `qualcomm_npu_cgra` | Qualcomm | NPU-CGRA | 8-bit | 1.2 GHz |
| `pact_xtensa` | Cadence | Xtensa-CGRA | 16-bit | 800 MHz |

### 3D-Stacked / Monolithic 3D

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `tsmc_soic` | TSMC | SoIC-3D | 16-bit | 2 GHz |
| `intel_foveros` | Intel | Foveros-Direct | 16-bit | 1.8 GHz |
| `amd_3dv` | AMD | 3D V-Cache | 16-bit | 2.2 GHz |

### Edge MCU / TinyML

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `rp2040` | Raspberry Pi | RP2040 | 16-bit | 133 MHz |
| `esp32_s3` | Espressif | ESP32-S3 | 16-bit | 240 MHz |
| `stm32h7` | STMicroelectronics | STM32H7 | 32-bit | 480 MHz |
| `nrf5340` | Nordic | nRF5340 | 16-bit | 128 MHz |
| `max78000` | Analog Devices | MAX78000 | 8-bit | 100 MHz |

### RISC-V AI Accelerators

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `sifive_x280` | SiFive | X280 | 16-bit | 2 GHz |
| `qualcomm_ventana` | Qualcomm/Ventana | Veyron-V2 | 16-bit | 3 GHz |
| `ainekko_rv` | Ainekko | ET-Mirai | 8-bit | 1 GHz |

## Biological, Electrochemical, and Wafer-Scale Platform Classes

### Biological / Wetware

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `finalspark_neuroplatform` | FinalSpark | Neuroplatform | 8-bit |
| `cortical_labs_dishbrain` | Cortical Labs | DishBrain | 8-bit |

### Electrochemical / Memristive

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `ibm_ecram` | IBM | ECRAM-AnalogAI | 8-bit |
| `samsung_pcram` | Samsung | PCRAM | 8-bit |
| `stanford_ecram` | Stanford | ECRAM-Research | 8-bit |

### Wafer-Scale

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `cerebras_wse3_ws` | Cerebras | WSE-3-WS | 16-bit | 1 GHz |
| `tesla_dojo3` | Tesla | Dojo-3 | 16-bit | 2 GHz |
| `tachyum_prodigy` | Tachyum | Prodigy-2nm | 16-bit | 5.5 GHz |

### Analog Mixed-Signal

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `aspinity_aml100` | Aspinity | AML100 | 8-bit |
| `renesas_analog_ai` | Renesas | AnalogAI | 8-bit |

## Memory-Centric and Quantum-Inspired Platform Classes

### RRAM / Memristive Crossbar

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `weebit_reram` | Weebit Nano | ReRAM-ACiM | 8-bit |
| `crossbar_rram` | Crossbar | ReRAM-1T1R | 8-bit |
| `adesto_cbram` | Adesto | CBRAM | 8-bit |

### SRAM Compute-in-Memory

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `tsmc_cim_n7` | TSMC | CIM-N7 | 8-bit | 1 GHz |
| `samsung_cim_sf3` | Samsung | CIM-SF3 | 8-bit | 900 MHz |

### Cryogenic CMOS

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `intel_horse_ridge` | Intel | Horse-Ridge-II | 16-bit | 6 GHz |
| `google_cryo_ctrl` | Google | Cryo-Controller | 16-bit | 4 GHz |

### DNA / Molecular

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `microsoft_dna_store` | Microsoft | DNA-Storage | 8-bit |
| `asu_dna_perovskite` | ASU | DNA-Perovskite | 8-bit |

### Quantum Neuromorphic

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `ibm_qnn` | IBM | Quantum-NN | 16-bit |
| `ionq_trapped_ion` | IonQ | Trapped-Ion-QNN | 16-bit |

## Further Reading

- [Precision Modes Guide](precision_modes.md) — all 11 Q-format modes
- [Static Analysis Guide](static_analysis_guide.md) — guard bits, overflow proof, SVA
- [SoC Integration Guide](soc_integration_guide.md) — bus wrappers, drivers, IP-XACT
- [Deployment Guide](deployment_guide.md) — constraints, TCL, Cocotb, bitstream
- [Co-Simulation Guide](cosimulation_guide.md) — Python↔Verilog verification
- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features (§26–§67)
- [Research Platforms Guide](research_platforms.md) — 38 platform classes deep-dive
- [Platform Extensibility Guide](platform_extensibility.md) — TOML loader + discovery hook
- [Safety Certification Guide](safety_certification.md) — DO-254, IEC 61508, ISO 26262
- [Tutorial 33: Equation-to-Verilog](../tutorials/33_equation_to_verilog.md)
- [SC for Hardware Engineers](SC_FOR_HARDWARE_ENGINEERS.md)
