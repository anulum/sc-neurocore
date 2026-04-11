# Email Draft: Neuralink — DJ Seo

**Status:** READY TO SEND (mailserver port 993 down, use webmail)
**Date prepared:** 2026-04-07
**From:** neurocore@anulum.li
**To:** dj@neuralink.com
**Subject:** SNN-to-Verilog compiler — accelerating on-chip neural processing design

---

DJ,

I lead a private research programme (Anulum, est. 2009, Switzerland/Liechtenstein) that has built what appears to be the only framework capable of compiling trained spiking neural networks to synthesisable Verilog with bit-true Python-to-hardware co-simulation.

The core tool: You give it an ODE system as a string. It returns a synthesisable Verilog module — fixed-point arithmetic, deterministic LFSR-based stochastic encoding, pipelined, formally verified. No manual RTL. Iteration cycle drops from weeks to hours.

We also have:

- Event-driven RTL primitives (AER encoder, event neuron, spike router) — 15–39x fewer register toggles than clock-driven at typical BCI spike rates (<1%). Direct power savings for implantable hardware.

- Formal verification — 72 properties proven for all possible inputs via bounded model checking and k-induction (SymbiYosys). Not simulation coverage — mathematical proof of correctness. Relevant for FDA-class device verification.

- Neural compression library — three modes on raw 1024-channel, 30 kHz electrode data: spike timing only at 4,500x (fits Bluetooth at 16,384 electrodes), waveform-preserving at 1,700x, or full LFP at 137x. Spike timing is lossless.

- Population decoders — four publication-exact implementations (POSSM/S4D for causal O(1)-per-step decoding, NDT3, POYO+, CEBRA), all with Rust-accelerated kernels.

No other publicly or commercially available SNN framework generates synthesisable RTL from trained models. snnTorch, Norse, and BindsNET are GPU-only. Lava is locked to Loihi. Brian2 generates C++ but no hardware. We have verified this is a competition-free position.

The framework is under dual license (AGPL + commercial). Repository is private — full access available for evaluation under NDA. A targeted technical brief covering only BCI-relevant modules is available on request.

I would welcome the opportunity to discuss whether this toolchain could be useful to your hardware team, and under what terms.

Best regards,

Miroslav Šotek
Anulum Research
protoscience@anulum.li
www.anulum.li
