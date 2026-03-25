<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Pricing

SC-NeuroCore is dual-licensed: open-source for research and education,
commercial licenses for proprietary integration.

<style>
.pricing-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}
.pricing-card {
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
}
.pricing-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.pricing-card.featured {
  border-color: #2d6a4f;
  position: relative;
}
.pricing-card.featured::before {
  content: "RECOMMENDED";
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  background: #2d6a4f;
  color: white;
  padding: 4px 16px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.pricing-card h3 {
  margin-top: 0;
  font-size: 1.4rem;
}
.pricing-price {
  font-size: 2.5rem;
  font-weight: 700;
  margin: 1rem 0 0.25rem;
}
.pricing-period {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
}
.pricing-features {
  text-align: left;
  list-style: none;
  padding: 0;
  margin: 1.5rem 0;
}
.pricing-features li {
  padding: 0.4rem 0;
  border-bottom: 1px solid #f0f0f0;
}
.pricing-features li::before {
  content: "✓ ";
  color: #2d6a4f;
  font-weight: bold;
}
.pricing-btn {
  display: inline-block;
  padding: 12px 32px;
  border-radius: 8px;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.2s;
}
.pricing-btn-primary {
  background: #2d6a4f;
  color: white !important;
}
.pricing-btn-primary:hover {
  background: #1b4332;
}
.pricing-btn-outline {
  border: 2px solid #2d6a4f;
  color: #2d6a4f !important;
}
.pricing-btn-outline:hover {
  background: #f0f7f4;
}
.pricing-btn-enterprise {
  background: #1a237e;
  color: white !important;
}
.pricing-btn-enterprise:hover {
  background: #0d1642;
}
.pricing-btn-founding {
  background: #b45309;
  color: white !important;
}
.pricing-btn-founding:hover {
  background: #92400e;
}
</style>

<div class="pricing-grid">

<div class="pricing-card">
<h3>Community</h3>
<div class="pricing-price">Free</div>
<div class="pricing-period">Open Source — AGPL-3.0</div>
<ul class="pricing-features">
<li>122 neuron models (113 biophysical + 9 AI-optimized)</li>
<li>Full Python + Rust SIMD engine (41.3 Gbit/s)</li>
<li>19 Verilog HDL modules + 7 formal verification files</li>
<li>Equation → Verilog compiler (<code>sc-neurocore compile</code>)</li>
<li>PyTorch training (7 surrogates, 10 cells, SpikingNet)</li>
<li>6-codec neural data compression library</li>
<li>125-function spike analysis toolkit</li>
<li>Quantum hybrid (Qiskit/PennyLane)</li>
<li>84 tutorials + full documentation</li>
<li>Community support (GitHub Discussions)</li>
<li>Source modifications must remain open (AGPL)</li>
</ul>
<a href="https://pypi.org/project/sc-neurocore/" class="pricing-btn pricing-btn-outline">pip install sc-neurocore</a>
</div>

<div class="pricing-card featured">
<h3>Professional</h3>
<div class="pricing-price">CHF 490</div>
<div class="pricing-period">per seat / year</div>
<ul class="pricing-features">
<li>Everything in Community</li>
<li>Closed-source integration permitted</li>
<li>Priority email support (48h SLA)</li>
<li>Pre-built FPGA bitstreams (ice40, ECP5, Artix-7)</li>
<li>Visual SNN Design Studio (when available)</li>
<li>Custom neuron model development (2 models/year)</li>
<li>Quarterly security advisories</li>
<li>No AGPL copyleft obligation</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_A6s1rmJQVQX6SFLqTuqyVFK2Y9wzigjSs5x193JWV1F" class="pricing-btn pricing-btn-primary">Buy Professional</a>
</div>

<div class="pricing-card">
<h3>Enterprise</h3>
<div class="pricing-price">CHF 4,900</div>
<div class="pricing-period">site license / year</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>Unlimited seats across organization</li>
<li>Dedicated support engineer (24h SLA)</li>
<li>On-premise deployment assistance</li>
<li>Custom FPGA/ASIC integration + target-specific RTL</li>
<li>Safety-critical certification support (ISO 26262)</li>
<li>Formal verification reports for your design</li>
<li>White-label and OEM licensing</li>
<li>Joint development agreements</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_LYpNQdmXAtYGZDU7BgiuD0p16dCO7a6Hz9PsP4cTsIh" class="pricing-btn pricing-btn-enterprise">Buy Enterprise</a>
</div>

</div>

<div style="text-align: center; margin: 1.5rem 0;">
<div class="pricing-card" style="display: inline-block; max-width: 400px; border-color: #b45309;">
<h3>Founding Member</h3>
<div class="pricing-price">CHF 290</div>
<div class="pricing-period">per seat / year — 10 spots only</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>50% lifetime discount (locked in forever)</li>
<li>Direct access to lead developer</li>
<li>Input on roadmap priorities</li>
<li>Name in CONTRIBUTORS.md + release notes</li>
<li>Free 30-day pilot before commitment</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_u6tIXv44uMXqtILxY3KIk9epbMK3AHZJEwllq0MpHut" class="pricing-btn pricing-btn-founding">Claim Your Spot</a>
</div>
</div>

---

## What You Get

### 122 Neuron Models — 82 Years of Computational Neuroscience

Every published neuron model from McCulloch-Pitts 1943 to ArcaneNeuron
2026. Biophysical (Hodgkin-Huxley, Izhikevich, AdEx, FitzHugh-Nagumo),
hardware emulators (Loihi, TrueNorth, BrainScaleS, SpiNNaker, Akida),
and 9 AI-optimized models. All with Rust SIMD acceleration.

### ODE → FPGA in One Command

```bash
sc-neurocore compile "dv/dt = -(v-E_L)/tau + I/C" \
    --threshold "v > -50" --reset "v = -65" \
    --params "E_L=-65,tau_m=10,C=1" --init "v=-65" \
    --target ice40 --testbench --synthesize
```

Transcendental functions (exp, log, tanh, sigmoid, sin, cos) via Q8.8
lookup tables. Saturating arithmetic prevents overflow bugs. Auto testbench
generation. One-click Yosys synthesis when toolchain is installed.

### PyTorch Training with SC Export

7 surrogate gradient functions, 10 differentiable neuron cells, SpikingNet
and ConvSpikingNet architectures. Train on GPU, export to stochastic
computing bitstreams via `to_sc_weights()`. Trainable per-synapse delays
(DelayLinear).

### 6-Codec Neural Data Compression

ISI+Huffman, Predictive (4 learnable predictors), Delta, Streaming, AER,
and WaveformCodec (24x on 1024-channel Neuralink-scale data). Unified API:
`get_codec(name)`, `recommend_codec()`. Rust backend (780x speedup).

### Python → Verilog Bit-True

Python simulation matches synthesisable RTL bit-for-bit (deterministic
LFSR seeds, Q8.8 fixed-point, cycle-exact co-simulation). Formal
verification with 67 SymbiYosys properties across 7 modules.

### Rust SIMD Engine — 41.3 Gbit/s AVX-512

111 Rust neuron models with PyO3 bindings, 81-model NetworkRunner with
Rayon-parallel populations. AVX-512, AVX2, NEON, SVE, RISC-V Vector
dispatch. 224 Mstep/s LIF neuron throughput.

---

## Feature Comparison

| Feature | Community | Professional | Enterprise |
|---------|:---------:|:------------:|:----------:|
| 122 neuron models | Yes | Yes | Yes |
| Rust SIMD engine (41.3 Gbit/s) | Yes | Yes | Yes |
| `sc-neurocore compile` (ODE → Verilog) | Yes | Yes | Yes |
| PyTorch training (7 surrogates, 10 cells) | Yes | Yes | Yes |
| 6-codec neural compression | Yes | Yes | Yes |
| 125-function spike analysis | Yes | Yes | Yes |
| 19 Verilog modules + formal verification | Yes | Yes | Yes |
| Quantum hybrid (Qiskit/PennyLane) | Yes | Yes | Yes |
| 84 tutorials + full docs | Yes | Yes | Yes |
| Closed-source use | No (AGPL) | **Yes** | **Yes** |
| Priority support | Community | 48h SLA | **24h SLA** |
| Pre-built FPGA bitstreams | — | ice40/ECP5/Artix-7 | **Custom targets** |
| Visual SNN Design Studio | — | **Yes** | **Yes** |
| Custom neuron models | — | 2/year | **Unlimited** |
| Safety certification (ISO 26262) | — | — | **Yes** |
| Formal verification reports | — | — | **Yes** |
| OEM / white-label | — | — | **Yes** |
| On-premise deployment | — | — | **Yes** |

---

## Academic Pricing

Free Professional license for .edu email addresses. Includes closed-source
rights for thesis work and research prototypes. Apply with your
institutional email:

<a href="mailto:protoscience@anulum.li?subject=SC-NeuroCore%20Academic%20License&body=Institution:%0AResearch%20group:%0AUse%20case:" class="pricing-btn pricing-btn-outline">Apply for Academic License</a>

---

## FAQ

**Can I use the Community edition for commercial research?**
Yes, as long as your modifications are also released under AGPL-3.0.
If you need to keep your code proprietary, choose the Professional license.

**What FPGA targets are supported?**
The RTL is vendor-agnostic (standard Verilog-2005). The `compile` CLI
targets ice40, ECP5, Artix-7, and Zynq. Professional includes pre-built
bitstreams. Enterprise adds custom ASIC targets.

**What is the Founding Member program?**
10 spots at 50% lifetime discount (CHF 290/seat/year instead of CHF 490).
Includes a free 30-day pilot, direct developer access, and roadmap input.
Once 10 spots are filled, the program closes permanently.

**Can I evaluate before purchasing?**
The Community edition is fully functional — all 122 neuron models,
Rust engine, Verilog RTL, compiler, training stack, and quantum modules.
The Professional license adds closed-source rights, support, and
pre-built bitstreams. Founding Members get 30 days free.

**What is the Visual SNN Design Studio?**
A web-based IDE for visual SNN design — ODE equation editor, network
canvas, training monitor, compiler inspector, and synthesis dashboard.
Currently in development. Professional and Enterprise licensees get
access when it launches.

**Do you offer volume discounts?**
Enterprise site licenses cover unlimited seats. For teams of 3-9,
contact us for multi-seat Professional pricing.

**What payment methods do you accept?**
Bank transfer (IBAN), Polar.sh (credit card), or invoice (NET-30 for
Enterprise). All prices in CHF. EUR and GBP accepted at daily exchange rate.

---

<p style="text-align: center; margin-top: 3rem;">
<strong>Ready to deploy neuromorphic AI on silicon?</strong><br>
<a href="mailto:protoscience@anulum.li?subject=SC-NeuroCore%20Inquiry" class="pricing-btn pricing-btn-primary" style="margin-top: 1rem;">Contact Us</a>
</p>

---

*SC-NeuroCore is developed by [ANULUM](https://www.anulum.li) — advancing
neuromorphic computing from simulation to silicon.*

*Contact: [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[www.anulum.li](https://www.anulum.li) |
ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)*
