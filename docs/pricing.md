---
title: Pricing
description: SC-NeuroCore licensing and pricing
---

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
  border-color: #b45309;
  position: relative;
}
.pricing-card.featured::before {
  content: "EARLY ADOPTER";
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  background: #b45309;
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
.pricing-price-full {
  text-decoration: line-through;
  color: #999;
  font-size: 1.2rem;
}
.pricing-period {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}
.pricing-savings {
  color: #b45309;
  font-weight: 700;
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
.pricing-badge {
  display: inline-block;
  background: #fef3c7;
  color: #92400e;
  padding: 2px 10px;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
</style>

!!! warning "Early Adopter Pricing — First 25 Customers"
    SC-NeuroCore commercial licenses are available at **introductory pricing**
    for the first 25 customers. These rates lock in permanently — when the
    program closes, standard pricing applies to new customers.

    **Spots remaining: 25 of 25**

<div class="pricing-grid">

<div class="pricing-card">
<h3>Community</h3>
<div class="pricing-price">Free</div>
<div class="pricing-period">Open Source — AGPL-3.0</div>
<div class="pricing-savings">&nbsp;</div>
<ul class="pricing-features">
<li>173 neuron models (113 biophysical + 9 AI-optimized)</li>
<li>Rust SIMD engine (113 Gbit/s AVX-512)</li>
<li>20 Verilog HDL modules + 72 formal properties</li>
<li>ODE → Verilog compiler (<code>sc-neurocore compile</code>)</li>
<li>PyTorch training (6 surrogates, 12 cells)</li>
<li>6-codec neural data compression</li>
<li>127-function spike analysis toolkit</li>
<li>Quantum hybrid (Qiskit/PennyLane)</li>
<li>87 tutorials + full API documentation</li>
<li>Community support (GitHub Discussions)</li>
<li>Source modifications must remain open (AGPL)</li>
</ul>
<a href="https://pypi.org/project/sc-neurocore/" class="pricing-btn pricing-btn-outline">pip install sc-neurocore</a>
</div>

<div class="pricing-card featured">
<h3>Professional</h3>
<span class="pricing-badge">First 25 customers</span>
<div class="pricing-price-full">CHF 1,490 /yr</div>
<div class="pricing-price">CHF 490</div>
<div class="pricing-period">per seat / year — locked permanently</div>
<div class="pricing-savings">Save CHF 1,000/yr (67% off)</div>
<ul class="pricing-features">
<li>Everything in Community</li>
<li>Closed-source integration permitted</li>
<li>Priority email support (48h business hours)</li>
<li>FPGA synthesis support + build guidance</li>
<li>Visual SNN Design Studio (coming soon)</li>
<li>Custom neuron model development (2 models/year)</li>
<li>Quarterly security advisories</li>
<li>No AGPL copyleft obligation</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_KhZxFeUlgfnriOpNWcUAMs39o7102otpmSO1l2ZRpZG" class="pricing-btn pricing-btn-founding">Buy Now — CHF 490/yr</a>
</div>

<div class="pricing-card">
<h3>Enterprise</h3>
<span class="pricing-badge">First 25 customers</span>
<div class="pricing-price-full">CHF 14,900 /yr</div>
<div class="pricing-price">CHF 4,900</div>
<div class="pricing-period">site license / year — locked permanently</div>
<div class="pricing-savings">Save CHF 10,000/yr (67% off)</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>Unlimited seats across organization</li>
<li>Priority email support (24h business hours)</li>
<li>On-premise deployment assistance</li>
<li>Custom FPGA target integration + RTL adaptation</li>
<li>Formal verification reports for your design</li>
<li>White-label and OEM licensing</li>
<li>Joint development agreements</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_YQvtDeDz3nGObr4llMZIPvGZj7QAoQ8phL6BY0tTU5Z" class="pricing-btn pricing-btn-enterprise">Buy Now — CHF 4,900/yr</a>
</div>

</div>

<div style="text-align: center; margin: 1.5rem 0;">
<div class="pricing-card" style="display: inline-block; max-width: 420px; border-color: #dc2626; border-width: 3px;">
<span class="pricing-badge" style="background: #fecaca; color: #991b1b;">10 spots — never again</span>
<h3>Founding Member</h3>
<div class="pricing-price-full">CHF 1,490 /yr</div>
<div class="pricing-price">CHF 290</div>
<div class="pricing-period">per seat / year — locked for life</div>
<div class="pricing-savings">Save CHF 1,200/yr (81% off standard price — forever)</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>Lifetime price lock (CHF 290/yr — even when standard is CHF 1,490)</li>
<li>Direct access to lead developer (email + video calls)</li>
<li>Input on roadmap priorities</li>
<li>Name in CONTRIBUTORS.md + release notes</li>
<li>Free 30-day evaluation before commitment</li>
<li>Early access to Visual SNN Design Studio</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_Umb7XhLU0HsWaKdhWDHOdQNYXrauBHFtmYT2R0TJ9NU" class="pricing-btn pricing-btn-founding">Claim Founding Spot — CHF 290/yr</a>
<p style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
Spots remaining: 10 of 10. When gone, this tier closes permanently.
</p>
</div>
</div>

---

## What You Get

### ODE → FPGA in One Command

```bash
sc-neurocore compile "dv/dt = -(v-E_L)/tau + I/C" \
    --threshold "v > -50" --reset "v = -65" \
    --params "E_L=-65,tau_m=10,C=1" --init "v=-65" \
    --target ice40 --testbench --synthesize
```

Write neuron equations as strings. The compiler produces synthesizable
Q8.8 Verilog RTL with saturating arithmetic, transcendental function
LUTs (exp, log, tanh, sigmoid, sin, cos), auto-generated testbenches,
and one-click Yosys synthesis. Python simulation matches Verilog
bit-for-bit (72 formal properties verified by SymbiYosys).

### PyTorch Training → SC Bitstream → FPGA

6 surrogate gradient functions, 12 differentiable neuron cells
(`nn.Module`), SpikingNet and ConvSpikingNet architectures. Train on
GPU with standard PyTorch optimizers, export weights to stochastic
computing bitstreams via `to_sc_weights()`, compile to Verilog RTL.
Trainable per-synapse delays (DelayLinear).

### 6-Codec Neural Data Compression

ISI+Huffman, Predictive (4 learnable predictors), Delta, Streaming, AER,
and WaveformCodec (24x compression on 1024-channel Neuralink-scale data).
Unified API: `get_codec(name)`, `recommend_codec()`. Rust backend (780x).

### 173 Neuron Models + Rust SIMD Engine

82 years of computational neuroscience (1943-2026). 173 Rust models with
PyO3 bindings, 160-model NetworkRunner with Rayon-parallel populations.
113 Gbit/s bitstream packing (AVX-512). 456 Mstep/s LIF throughput.

### 127-Function Spike Analysis Toolkit

CV, Fano factor, cross-correlation, Victor-Purpura distance, SPIKE-sync,
Granger causality, GPFA, SPADE pattern detection. Matches Elephant +
PySpike combined. Pure NumPy — no additional dependencies.

---

## Feature Comparison

| Feature | Community | Professional | Enterprise |
|---------|:---------:|:------------:|:----------:|
| 173 neuron models + Rust engine | Yes | Yes | Yes |
| `sc-neurocore compile` (ODE → Verilog) | Yes | Yes | Yes |
| PyTorch training (6 surrogates, 12 cells) | Yes | Yes | Yes |
| 6-codec neural compression | Yes | Yes | Yes |
| 127-function spike analysis | Yes | Yes | Yes |
| 20 Verilog modules + 72 formal properties | Yes | Yes | Yes |
| Quantum hybrid (Qiskit/PennyLane) | Yes | Yes | Yes |
| 87 tutorials + full docs | Yes | Yes | Yes |
| Closed-source use | No (AGPL) | **Yes** | **Yes** |
| Priority support | Community | **48h** | **24h** |
| FPGA synthesis support | Self-serve | **Guided** | **Custom targets** |
| Visual SNN Design Studio | — | **Yes** (coming) | **Yes** (coming) |
| Custom neuron models | — | 2/year | **Unlimited** |
| Formal verification reports | — | — | **Yes** |
| OEM / white-label | — | — | **Yes** |
| On-premise deployment | — | — | **Yes** |

---

## Why Buy Now?

```text
Standard pricing (after early adopter program closes):

    Professional:     CHF 1,490 /yr per seat
    Enterprise:       CHF 14,900 /yr site license

Early adopter pricing (first 25 customers — locked permanently):

    Professional:     CHF 490 /yr per seat      ← you save CHF 1,000/yr
    Enterprise:       CHF 4,900 /yr site license ← you save CHF 10,000/yr
    Founding Member:  CHF 290 /yr per seat       ← you save CHF 1,200/yr

    Example: Year 1 you pay CHF 490.
             Year 5 new customers pay CHF 1,490.
             You still pay CHF 490. Every year. Forever.
```

SC-NeuroCore is the only open-source framework with a complete
ODE → PyTorch training → SC bitstream → Verilog RTL → FPGA bitstream
pipeline. The standard pricing reflects this — CHF 1,490/yr is still
less than a single Vivado license (CHF 3,000+/yr), and SC-NeuroCore
includes the full SNN training stack, 173 neuron models, and formal
verification that Vivado doesn't offer.

Early adopters get this at 67-81% off. Permanently.

---

## Academic Pricing

Free Professional license for .edu email addresses. Includes closed-source
rights for thesis work and research prototypes. No strings attached.

<a href="mailto:protoscience@anulum.li?subject=SC-NeuroCore%20Academic%20License&body=Institution:%0AResearch%20group:%0AUse%20case:" class="pricing-btn pricing-btn-outline">Apply for Academic License</a>

---

## FAQ

**Can I use the Community edition for commercial research?**
Yes, as long as your modifications are also released under AGPL-3.0.
If you need to keep your code proprietary, choose Professional.

**What FPGA targets are supported?**
The RTL is vendor-agnostic (Verilog-2005). The `compile` CLI targets
ice40, ECP5, Artix-7, and Zynq. Professional includes guided synthesis
support. Enterprise adds custom ASIC targets.

**What is the early adopter program?**
The first 25 paying customers lock in current pricing permanently.
When the 25th customer signs up, prices increase to standard rates
for all new customers. Existing customers keep their locked rate.

**What is the Founding Member tier?**
10 spots at 81% off standard pricing — CHF 290/yr instead of CHF 1,490.
Includes direct developer access, roadmap input, and a free 30-day
evaluation. Once 10 spots fill, the tier closes permanently.

**Can I evaluate before purchasing?**
The Community edition is fully functional. Founding Members additionally
get 30 days free before their first payment.

**What is the Visual SNN Design Studio?**
A web-based IDE for visual SNN design — equation editor, network canvas,
training monitor, compiler inspector, synthesis dashboard. Currently in
development. Professional and Enterprise licensees get access at launch.

**What payment methods do you accept?**
Credit card via Polar.sh, bank transfer (IBAN), or invoice (NET-30 for
Enterprise). All prices in CHF. EUR and GBP accepted at daily exchange rate.

**Do you offer refunds?**
30-day money-back guarantee on all tiers. No questions asked.

---

<p style="text-align: center; margin-top: 3rem;">
<strong>Ready to deploy neuromorphic AI on silicon?</strong><br>
<a href="https://polar.sh/checkout/polar_c_KhZxFeUlgfnriOpNWcUAMs39o7102otpmSO1l2ZRpZG" class="pricing-btn pricing-btn-primary" style="margin-top: 1rem;">Get Started — CHF 490/yr</a>
&nbsp;&nbsp;
<a href="mailto:protoscience@anulum.li?subject=SC-NeuroCore%20Inquiry" class="pricing-btn pricing-btn-outline" style="margin-top: 1rem;">Talk to Us</a>
</p>

---

*SC-NeuroCore is developed by [ANULUM](https://www.anulum.li) — advancing
neuromorphic computing from simulation to silicon.*

*Contact: [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[www.anulum.li](https://www.anulum.li) |
ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)*
