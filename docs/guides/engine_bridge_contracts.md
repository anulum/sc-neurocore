<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
# SC-NeuroCore — Stable Engine Bridge Contracts

Stable engine consumers should use explicit wrapper modules under
`bridge/sc_neurocore_engine/`, not the top-level `sc_neurocore_engine`
namespace as a generic export bag.

Current maintained wrapper surfaces:

- `sc_neurocore_engine.network`
- `sc_neurocore_engine.studio`
- `sc_neurocore_engine.world_model`
- `sc_neurocore_engine.photonics`
- `sc_neurocore_engine.dna`
- `sc_neurocore_engine.quantum`

Why:

- explicit import points are easier to test
- capability failure becomes local and diagnosable
- consumers stop depending on unrelated re-export churn in
  `bridge/sc_neurocore_engine/__init__.py`
- mypy can be paid down module by module instead of treating the entire
  top-level bridge as one dynamic surface

Rule:

- runtime code should import the narrow wrapper it actually needs
- top-level re-exports remain compatibility surface, not the preferred
  implementation contract
