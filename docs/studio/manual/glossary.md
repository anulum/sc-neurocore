<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Glossary

| Term | Meaning in Studio |
|---|---|
| SNN | Spiking neural network: units communicate through discrete spike events |
| Neuron model | Equations/state and a numerical implementation for one neuron's behaviour |
| Catalogue | The models and metadata available in the current installation |
| ODE | Ordinary differential equation; in ODE mode you enter the equations to run |
| Parameter | A configured model value, such as a time constant; distinct from changing state |
| State / initial state | Values advanced by the model / values before its first step |
| `dt` | Numerical timestep; inspect the model's units and method |
| Protocol / drive | Input applied over time, such as a constant, step or sine current |
| Threshold / reset | Condition producing a spike and the state update after that event |
| Seed | Input identifying a supported pseudo-random stream; not applicable to every model |
| Replay / fresh trial | Reuse the effective random experiment / draw and report a new seed |
| Experiment specification | Resolved source, inputs, numerical/randomness/backend and runtime contract |
| Digest / SHA-256 | Content fingerprint used to identify/check bytes or canonical documents; not a signature |
| Raw trace | Full-resolution recorded samples, subject to explicit custody budgets |
| Display projection | Bounded samples selected to draw a result; not a replacement for raw data |
| ISI | Inter-spike interval: elapsed time between consecutive spikes |
| f-I curve | Firing-rate response as input current varies under a stated measurement protocol |
| Rheobase | Threshold-current estimate under a particular duration/protocol/definition |
| Nullcline | State-space points where one derivative is zero; inspect the plotted domain |
| Population | A graph node containing neurons of one supported model |
| Projection | Directed connection from a source population to a target population |
| Autapse | A neuron connected to itself; the graph controls whether these are admitted |
| NIR | Neural Intermediate Representation used for exchange; imports still require supported operations |
| Surrogate gradient | Differentiable approximation used when training through spike events |
| Epoch / batch | One training-data pass / a group of samples processed together |
| Checkpoint | Recorded training configuration and, where declared, artifacts/state needed for continuation |
| Warm start / exact resume | Start a new run from compatible weights / continue under a qualified complete-state contract |
| Hold-out | Data kept out of optimisation for evaluation under a declared split |
| Candidate | Proposed model package; not an admitted catalogue entry |
| IR | Intermediate representation used by a compiler; structural validity does not establish source fidelity |
| RTL | Register-transfer-level hardware source, such as Verilog/SystemVerilog |
| Q-format | Fixed-point numeric format defining width/fraction; inspect rounding and overflow policy |
| Bit-true kernel | Reference integer computation intended to match a declared hardware numeric contract |
| Co-simulation | Running implementations together and comparing recorded signals for bounded stimuli |
| Synthesis | Mapping RTL to a target netlist and reporting counted resources |
| Place and route | Assigning physical device resources and connections under a tool/constraint configuration |
| LUT / FF / BRAM / DSP | FPGA logic look-up table / flip-flop / block RAM / arithmetic resource |
| Target profile | Named tool/device-capacity configuration, not automatic physical-board qualification |
| Workspace / revision | Server-saved project / immutable saved state in its history |
| Session | Browser-local named experiment snapshot; separate from server workspace storage |
| Origin | Scheme, host and port that identify browser storage and request-policy context |
| Job / ledger | Tracked operation / durable records of its status, events and artifacts |
| Receipt / evidence bundle | Scoped record of inputs/outcomes / exported collection with manifests and artifacts |
| Declared / verified tier | Descriptor claim / evidence-backed tier for the stated source or build seal |
| Capability | Whether a deployment/model admits an operation, with its prerequisites and reasons |
| Federation | Embedding/integrating Studio through a host's plugin and backend contract |
| Loopback | Local-machine network addresses such as `127.0.0.1` |

Return to the [manual contents](index.md) or [first experiment](first-experiment.md).
