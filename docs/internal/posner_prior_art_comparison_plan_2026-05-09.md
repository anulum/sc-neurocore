<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Posner Prior-Art Comparison Plan - 2026-05-09

This is an internal acquisition note for the SC-NeuroCore Posner ORCA lane. It
records which published Posner data can be used as comparison evidence, which
data must be fetched before use, and what must remain excluded from runtime
parameters until SC-NeuroCore derives and validates its own ORCA tensors.

## Current Answer

Yes, other groups have published Posner data we can use. The correct use is as
provenance-tracked comparison fixtures, not as silent replacements for our ORCA
outputs. The active r6 neutral geometry still has to converge before any final
SC-NeuroCore `hf.json` or `extended.json` values are generated.

## Source Matrix

| Source | What it gives | Use in SC-NeuroCore | Gate before use |
|---|---|---|---|
| Swift, Van de Walle, Fisher 2018, DOI `10.1039/C7CP07720C` | First-principles Posner structure, vibrational spectra, cation interactions, pair binding, nuclear-spin context | Primary structural benchmark for final neutral ORCA geometry, P-P distances, P-O/Ca-O statistics, and high-symmetry baseline | Enter exact numeric values only from article/SI with units and extraction notes |
| Fisher 2015, DOI `10.1016/j.aop.2015.08.020` | Mechanistic hypothesis for phosphate nuclear-spin processing | Background only; not a molecular-data fixture | Do not use as runtime constants |
| Player and Hore 2018, DOI `10.1098/rsif.2018.0494` | Spin-dynamics model, scalar-coupling assumptions, 37-minute idealised lifetime upper bound | Downstream simulator/lifetime comparison once our molecular tensors exist | Preserve atom ordering and coupling sign conventions |
| Agarwal, Aiello, Kattnig, Banerjee 2021, DOI `10.1021/acs.jpclett.1c02796` | Ab initio MD/relaxation evidence that room-temperature Posner ensembles are predominantly low-symmetry | Required control against forcing the r6 geometry into a high-symmetry story | Compare symmetry residuals and document if r6 relaxes to a low-symmetry basin |
| Agarwal, Kattnig, Aiello, Banerjee 2023, DOI `10.1021/acs.jpclett.2c03945` | Asymmetric Posner spin dynamics and calcium-phosphate dimer alternative | Negative/alternative-structure benchmark for entanglement-decay claims | Keep separate from trimer runtime data unless an explicit dimer model is implemented |
| Entanglement and coherence in pure and doped Posner molecules, Scientific Reports 2025, DOI `10.1038/s41598-025-96487-5` | ORCA-derived J-couplings for pure and lithium-doped Posner models, spin coherence/concurrence analysis | Direct comparison target for J-coupling and doped-Posner branches | Fetch author data or tables first; article states data is available from the corresponding author |

## Data-Use Policy

Published numbers can become comparison fixtures when all of the following are
true:

1. The source DOI, paper title, authors, and table/figure/SI location are
   recorded.
2. Units, sign conventions, atom indexing, charge, multiplicity, method,
   basis set, and geometry state are recorded.
3. The extract is small, factual, and copyright-compliant.
4. A parser or fixture test proves the local JSON/table has the expected
   shape, finite numeric values, and declared units.
5. The file is stored under `results/posner_external_data/` or an internal
   provenance directory with a README that says it is comparison data.

Published numbers must not become runtime constants unless they are explicitly
declared external inputs and pass the same runtime validation as SC-NeuroCore
ORCA outputs. Runtime Posner verification should prefer our own accepted ORCA
geometry, radical EPR outputs, and parsed full tensors.

## Required Comparison Reports After r6

1. `geometry_report.json`: atom count, charge/multiplicity, final energy,
   convergence markers, P-P distance matrix, P-O/Ca-O summaries, point-group
   residual, and RMSD against the Swift baseline if exact baseline coordinates
   are available.
2. `symmetry_control.md`: high-symmetry Swift comparison versus low-symmetry
   Agarwal framing, with a plain statement of where the r6 structure lands.
3. `epr_tensor_report.json`: full 31P hyperfine tensors, calcium tensors where
   available, units, principal values, eigenvectors, and parser provenance.
4. `coupling_comparison.md`: Swift/Player-Hore/pure-doped-ORCA J-coupling
   comparison with atom-order mapping and sign conventions.
5. `runtime_gate.md`: statement of which values are permitted in `hf.json` and
   `extended.json`, and which remain comparison-only.

## Immediate Follow-Up

- Fetch Swift supporting information or source tables needed for exact
  structural comparison.
- Contact or retrieve the Scientific Reports 2025 corresponding-author data
  package before using its ORCA J-coupling constants as local fixtures.
- After r6 converges, generate the geometry report before launching radical EPR
  so the neutral structure is either accepted or rejected on evidence.
