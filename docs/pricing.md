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
  content: "MOST POPULAR";
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
</style>

<div class="pricing-grid">

<div class="pricing-card">
<h3>Community</h3>
<div class="pricing-price">Free</div>
<div class="pricing-period">Open Source — AGPL-3.0</div>
<ul class="pricing-features">
<li>122 neuron models</li>
<li>Full Python + Rust SIMD engine</li>
<li>10 Verilog HDL modules</li>
<li>IR compiler (SystemVerilog + MLIR)</li>
<li>Surrogate gradient training</li>
<li>Quantum hybrid (Qiskit/PennyLane)</li>
<li>38 tutorials + full documentation</li>
<li>Community support (GitHub Discussions)</li>
<li>Source code must remain open (AGPL)</li>
</ul>
<a href="https://pypi.org/project/sc-neurocore/" class="pricing-btn pricing-btn-outline">pip install sc-neurocore</a>
</div>

<div class="pricing-card featured">
<h3>Professional</h3>
<div class="pricing-price">Contact Us</div>
<div class="pricing-period">Commercial License — Per Seat</div>
<ul class="pricing-features">
<li>Everything in Community</li>
<li>Closed-source integration permitted</li>
<li>Priority email support (48h SLA)</li>
<li>Private GitHub repository access</li>
<li>Quarterly security advisories</li>
<li>Pre-built FPGA bitstreams (Artix-7, Zynq)</li>
<li>Custom neuron model development</li>
<li>ONNX + TensorRT export pipeline</li>
<li>No AGPL copyleft obligation</li>
</ul>
<a href="mailto:protoscience@anulum.li?subject=SC-NeuroCore%20Professional%20License" class="pricing-btn pricing-btn-primary">Request Quote</a>
</div>

<div class="pricing-card">
<h3>Enterprise</h3>
<div class="pricing-price">Custom</div>
<div class="pricing-period">Site License + Dedicated Support</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>Unlimited seats across organization</li>
<li>Dedicated support engineer (24h SLA)</li>
<li>On-premise deployment assistance</li>
<li>Custom FPGA/ASIC integration</li>
<li>Safety-critical certification support (ISO 26262)</li>
<li>Formal verification reports for your design</li>
<li>White-label and OEM licensing</li>
<li>Joint development agreements</li>
</ul>
<a href="mailto:protoscience@anulum.li?subject=SC-NeuroCore%20Enterprise%20License" class="pricing-btn pricing-btn-enterprise">Contact Sales</a>
</div>

</div>

---

## What You Get

### 122 Neuron Models — The World's Largest Library

Every published neuron model from computational neuroscience, spanning
82 years (McCulloch-Pitts 1943 to ArcaneNeuron 2026). From simple LIF
to Hodgkin-Huxley ion channels, Hay Layer-5 pyramidal cells, hardware
chip emulators (Loihi, TrueNorth, BrainScaleS, SpiNNaker, Akida), and
9 AI-optimized models for cognitive workloads.

### 99.49% MNIST Accuracy

State-of-the-art SNN classification with ConvSpikingNet. Learnable
neuron parameters, surrogate gradient training, and direct export
to stochastic bitstream weights for FPGA deployment.

### Python → FPGA in One Pipeline

```
Train in PyTorch → Export SC weights → IR Compiler → SystemVerilog / MLIR → FPGA Bitstream
```

The only framework where the Python simulation matches synthesisable
RTL bit-for-bit. Formal verification with 64 SymbiYosys properties across 7 modules.

### Rust SIMD Engine — 41.3 Gbit/s AVX-512

100% Python parity with AVX-512, AVX2, NEON, SVE, and RISC-V Vector
dispatch. 224 Mstep/s LIF neuron throughput. 41.3 Gbit/s bitstream packing.

---

## Comparison

| Feature | Community | Professional | Enterprise |
|---------|:---------:|:------------:|:----------:|
| 122 neuron models | Yes | Yes | Yes |
| Rust SIMD engine | Yes | Yes | Yes |
| Verilog RTL | Yes | Yes | Yes |
| Formal verification | Yes | Yes | Yes |
| Quantum hybrid | Yes | Yes | Yes |
| Closed-source use | No (AGPL) | **Yes** | **Yes** |
| Priority support | — | 48h SLA | 24h SLA |
| FPGA bitstreams | — | Pre-built | Custom |
| Safety certification | — | — | **ISO 26262** |
| OEM / white-label | — | — | **Yes** |
| Custom development | — | Available | Included |

---

## Trusted By

*Research institutions and companies using SC-NeuroCore for
neuromorphic computing, edge AI, and quantum-classical hybrid systems.*

---

## FAQ

**Can I use the Community edition for commercial research?**
Yes, as long as your modifications are also released under AGPL-3.0.
If you need to keep your code proprietary, choose the Professional license.

**What FPGA targets are supported?**
The RTL is vendor-agnostic (standard Verilog-2005). Pre-built bitstreams
are available for Xilinx Artix-7 and Zynq series. Custom targets
available under Enterprise.

**Do you offer academic discounts?**
Yes. Contact us with your institution details for special academic pricing.

**Can I evaluate before purchasing?**
The Community edition is fully functional. All 122 neuron models,
the Rust engine, Verilog RTL, and quantum modules are included.
The Professional license adds closed-source rights and support.

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
