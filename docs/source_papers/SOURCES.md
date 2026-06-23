# Source papers

Reference literature we transcribe parameters from or that is within our modelling
scope. **The full publisher PDFs are kept locally under `pdfs/` (git-ignored) — they
are copyright-bound and are never committed.** What is tracked here is only our own
transcribed parameter tables (`parameters/`, facts and attributed) and this index.

To reproduce our parameters, obtain each paper from its DOI/URL below and compare
against the matching note in `parameters/`.

| File (`pdfs/`) | Citation | DOI / source | License | We use it for |
|---|---|---|---|---|
| `Chay_Keizer_1983_minimal_model_pancreatic_beta_cell_BiophysJ_42_181-190.pdf` | Chay, T.R. & Keizer, J. (1983). Minimal model for membrane oscillations in the pancreatic beta-cell. *Biophys. J.* 42:181–190. | [10.1016/S0006-3495(83)84384-7](https://doi.org/10.1016/S0006-3495(83)84384-7) | © Biophysical Society | `ChayKeizerNeuron` (5-D) — Table I + Eqs 1–9. See `parameters/chay_keizer_1983.md`. |
| `Demonstration-Chay-Keizer-Model-...-definition.nb` | Wolfram Demonstrations Project — Chay–Keizer model for electrical activity of the pancreatic beta-cell. | [demonstrations.wolfram.com](https://demonstrations.wolfram.com/) | CC BY-NC-SA 3.0 | Independent reference implementation; cross-checked the 5-D equations, the cell-radius units and the temperature factor. |
| `Sherman_Bertram_Integrative_modeling_pancreatic_beta_cell_encyclopedia_NIDDK.pdf` | Sherman, A. & Bertram, R. Integrative modeling of the pancreatic β-cell. *Wiley Encyclopedia of Genetics, Genomics, Proteomics and Bioinformatics.* | [10.1002/047001153X.g308213](https://doi.org/10.1002/047001153X.g308213) | © Wiley | Equation structure of the reduced "Chay–Keizer-like" minimal model (planned `ChayKeizerMinimalNeuron`). |
| `0702010v1.pdf` | The electrophysiology of the beta-cell based on single transmembrane protein characteristics. | [arXiv:q-bio/0702010](https://arxiv.org/abs/q-bio/0702010) | arXiv preprint | Beta-cell electrophysiology background. |
| `1-s2.0-S0025556423001256-main.pdf` | Deconstructing the integrated oscillator model for pancreatic β-cells. *Mathematical Biosciences.* | [10.1016/j.mbs.2023.109015](https://doi.org/10.1016/j.mbs.2023.109015) | © Elsevier | Beta-cell oscillator-model context. |
| `MBS_23.pdf` | *Mathematical Biosciences* article (beta-cell modelling). | — | © Elsevier | Beta-cell modelling context. |
| `Programmable_neuromorphic_circuits_for_s.pdf` | Programmable neuromorphic circuits (neuromorphic hardware). | — | — | Neuromorphic-hardware background. |

## Adding a source

1. Drop the PDF into `pdfs/` (it is git-ignored automatically).
2. Add a row above with the full citation, DOI/URL and license.
3. If we transcribe parameters from it, add a `parameters/<slug>.md` note citing it.
