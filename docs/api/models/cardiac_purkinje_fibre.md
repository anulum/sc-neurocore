# CardiacPurkinjeFibre

**Module:** `engine/src/neurons/misc.rs`
**Reference:** DiFrancesco & Noble, Phil Trans R Soc Lond B 307:353, 1985
**Family:** Cardiac conduction cell with pacemaker capability
**State variables:** `v`, `m`, `h` (Na), `d`, `f` (CaL), `x_r` (IKr), `y` (If/HCN)

---

## Biological Context

Cardiac Purkinje fibres are specialised conduction cells in the ventricular conduction system. They have:
- Long action potentials (~300 ms) with distinct phases (0-4)
- Intrinsic pacemaker capability via If (funny current / HCN channels)
- High conduction velocity for rapid ventricular activation

The DiFrancesco-Noble model includes 6 major ionic currents covering all AP phases.

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{Na} + I_{CaL} + I_{Kr} + I_{K1} + I_f + I_L) + I_{ext}$$

| Current | Gating | Phase | Role |
|---------|--------|-------|------|
| INa | m³h | 0 | Rapid depolarisation |
| ICaL | d·f | 2 | Plateau maintenance |
| IKr | x_r | 3 | Repolarisation |
| IK1 | Boltzmann(V) | 4/rest | Resting potential |
| If | y | 4 | Pacemaker depolarisation |
| IL | — | all | Background leak |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v, d, f, y) |
| NetworkRunner wired | `NeuronVariant::CardiacPurkinje` |
| `create_neuron("CardiacPurkinjeFibre")` | Yes |
| `supported_models()` | Includes "CardiacPurkinjeFibre" |
| STRONG tests | 10 |
| Benchmark | `cardiac_purkinje_1k_steps`: **586.7 µs** (586.7 ns/step), i5-11600K |
