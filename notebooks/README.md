# SC-NeuroCore Notebooks

The notebooks are executable documentation for onboarding, feature exploration, and evidence review. They are not a substitute for the test suite or benchmark artefacts; use them to understand workflows and to reproduce documented evidence boundaries.

## Recommended order

1. `quickstart_colab.ipynb` for a zero-local-setup introduction.
2. `03_end_to_end_pipeline.ipynb` for encode -> synapse -> layer workflow intuition.
3. `05_nir_bridge.ipynb` if cross-framework interoperability matters.
4. `08_equation_to_verilog.ipynb` and `27_python_to_proven_silicon.ipynb` for hardware-oriented flows.
5. Evidence notebooks `29` and later when reviewing claims, readiness gates, or local artefacts.

## Reproducibility rules

- Install only the extras required by the notebook category.
- Keep generated artefacts in ignored build/result locations unless they are intentional benchmark or evidence outputs.
- Do not promote notebook output to README, release, or market claims unless the raw artefact is committed and named in the docs.
- Treat hardware, clinical, regulatory, power, and energy statements as gaps unless the notebook points to the exact committed report.

For the full notebook map, see `docs/guides/notebook_guide.md`.
