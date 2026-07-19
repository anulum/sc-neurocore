# SC-NeuroCore Notebooks

The notebooks are executable documentation for onboarding, feature exploration, and evidence review. They are not a substitute for the test suite or benchmark artefacts; use them to understand workflows and to reproduce documented evidence boundaries.

## Recommended order

1. `quickstart_colab.ipynb` for a zero-local-setup introduction.
2. `03_end_to_end_pipeline.ipynb` for encode -> synapse -> layer workflow intuition.
3. `05_nir_bridge.ipynb` if cross-framework interoperability matters.
4. `08_equation_to_verilog.ipynb` and `27_python_to_proven_silicon.ipynb` for hardware-oriented flows.
5. Evidence notebooks `29` and later when reviewing claims, readiness gates, or local artefacts.

## Audience map

| Audience | Notebook path |
| --- | --- |
| Research lab | `04_neuron_explorer`, `10_spike_train_analysis`, `16_neuron_atlas`, then the relevant evidence notebook. |
| Hardware team | `08_equation_to_verilog`, `13_quantisation_pipeline`, `27_python_to_proven_silicon`, `29_golden_path_evidence`. |
| Commercial evaluator | `29_golden_path_evidence`, `34_industrial_readiness_evidence`, `36_fault_resilience_evidence`, `38_formal_snn_verification_standard_evidence`. |
| Interop reviewer | `05_nir_bridge`, `28_domain_bridge`, and the cross-framework benchmark documentation. |
| High-fidelity demos | `41_one_model_five_views` (HH / Morris–Lecar / AdEx), `42_fault_tolerance_theatre` (Perfect Integrator + SC BER), `43_studio_evidence_cart_lab` (AdEx digests). |

## Reproducibility rules

- Install only the extras required by the notebook category.
- Keep generated artefacts in ignored build/result locations unless they are intentional benchmark or evidence outputs.
- Do not promote notebook output to README, release, or market claims unless the raw artefact is committed and named in the docs.
- Treat hardware, clinical, regulatory, power, and energy statements as gaps unless the notebook points to the exact committed report.

## High-fidelity demo track (41–43)

Followable programme (polyglot-complete neurons only):
monorepo plan
`PLAN_2026-07-19T2301_notebook_demo_programme_high_fidelity_neurons.md`.

| Notebook | Models | Story |
| --- | --- | --- |
| `41_one_model_five_views.ipynb` | Hodgkin–Huxley, Morris–Lecar, AdEx | Dynamics → f–I → SC rate pedagogy → quant demo → hardware pointers |
| `42_fault_tolerance_theatre.ipynb` | Perfect Integrator | SC product vs float under bit flips |
| `43_studio_evidence_cart_lab.ipynb` | AdEx | Pedagogical ledger digests (Studio cart companion) |

For the full notebook map, see `docs/guides/notebook_guide.md`.
