# Applications and Market

SC-NeuroCore addresses teams that need neuromorphic and stochastic-computing evidence, not just simulation demos. Its practical value is strongest where energy, reliability, latency, or hardware auditability matter.

## Addressable application lanes

| Lane | Problem | SC-NeuroCore contribution |
| --- | --- | --- |
| Edge neuromorphic inference | Low-power inference for sensors and embedded systems. | SNN cells, stochastic encoders, quantisation/export paths, and hardware-oriented artefacts. |
| FPGA and ASIC exploration | Early design-space exploration before committing to silicon. | RTL generation, synthesis reports, fixed-point paths, and benchmark-driven trade-off analysis. |
| Neuroscience and computational modelling | Compare neuron families and network dynamics reproducibly. | Broad neuron catalogue, numerical guardrails, parity tests, and notebook evidence. |
| BCI and spike-codec prototyping | Compress and transform neural events under latency and bandwidth constraints. | Spike codecs, AER paths, waveform evidence notebooks, and readiness boundaries. |
| Safety and industrial readiness | Convert research claims into evidence categories and gap lists. | Evidence bags, industrial profiles, fail-closed readiness arithmetic, and documentation traceability. |
| Framework interoperability | Move models between SNN ecosystems and hardware-oriented flows. | NIR bridge, cross-framework benchmarks, and documented optional dependency profiles. |

## Market value thesis

The project is valuable because it sits between three markets that are usually served by separate tools:

- neuromorphic and SNN research frameworks;
- hardware/EDA prototyping and FPGA deployment flows;
- industrial evidence, safety-case, and readiness tooling.

A buyer or partner does not only get another simulator. They get a route for asking whether a stochastic or spiking model can be measured, compared, exported, and defended with artefacts. That makes the software relevant for research labs, hardware start-ups, industrial R&D groups, and organisations evaluating neuromorphic edge systems.

## Commercial evaluation sequence

1. **Fit:** choose an application lane and identify the minimum useful workflow
   from modelling, training, interop, hardware, or evidence tooling.
2. **Evidence:** run only the relevant tutorials, notebooks, tests, and
   benchmarks, then keep the raw artefacts named in the report.
3. **Gap review:** classify missing timing, power, hardware, clinical,
   cybersecurity, regulatory, or external-dataset evidence explicitly.
4. **Pilot:** scope a target-specific proof of concept around the missing
   evidence, not around broad feature-count claims.
5. **License:** use AGPL for open research or request a commercial license for
   closed-source evaluation, embedding, OEM, or white-label use.

## Differentiation

SC-NeuroCore is differentiated by the combination of stochastic bitstream arithmetic, spiking neural models, optional accelerated execution, generated hardware artefacts, and evidence-indexed documentation. Many SNN frameworks support training or biological simulation. Few make stochastic-computing arithmetic and hardware evidence central to the workflow.

The distinct public claim is therefore not that every experimental module is deployment-ready. The distinct claim is that the repository provides a broad, auditable path from stochastic neural modelling to hardware-oriented evidence, while marking missing evidence explicitly.

## Commercial boundaries

SC-NeuroCore can support commercial evaluation, prototypes, and evidence planning. Regulated deployment still requires target-specific validation, independent safety assessment, hardware timing/power reports, cybersecurity review, and domain authority acceptance. The industrial profiles describe those missing evidence categories rather than hiding them.

## Buyer-facing entry points

- Technical due diligence: [Product Overview](product_overview.md), [Architecture](architecture/architecture.md), and [API Reference Index](api/API_REFERENCE.md).
- Performance review: [Benchmarks](benchmarks/BENCHMARKS.md) and [Cross-Framework Benchmark Evidence](benchmarks/cross_framework.md).
- Hardware feasibility: [Hardware Guide](hardware/HARDWARE_GUIDE.md) and [FPGA Toolchain Guide](hardware/FPGA_TOOLCHAIN_GUIDE.md).
- Research evaluation: [Learning Path](LEARNING_PATH.md), [Notebook Guide](guides/notebook_guide.md), and evidence notebooks.
- Industrial readiness: [Industrial Applications](api/industrial_applications.md) and safety-certification docs.
