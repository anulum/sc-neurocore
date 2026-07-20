<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Recurring audit cadence
-->

# Audit Cadence

The `Audit Cadence` workflow runs on the first day of each month and can also
be started manually from GitHub Actions. It is a lightweight inventory audit,
not a full test-suite substitute.

The workflow performs five checks:

1. Install pinned Go and Mojo toolchains.
2. Install the development test environment with `python tools/ci_install_dev.py`.
3. Build the Go and Mojo `rk4_neurons` and `wilson_cowan` shared libraries whose
   parity modules gate collection on real native availability.
4. Run `python -m pytest tests/ --collect-only -q`.
5. Run `python tools/test_inventory_audit.py` against the collection transcript.

The audit compares tracked `test_*.py` files with the files observed in
pytest's collect-only output. A tracked file may be absent from base collection
only when it declares a module-level `pytest.importorskip(...)` optional
dependency gate or the standard `__test__ = False` marker used by an explicit
compatibility or shared-support module. Both categories remain visible in the
JSON artefact. Any other uncollected tracked test file fails the workflow.

The workflow uploads two artefacts:

| Artefact | Purpose |
| --- | --- |
| `audit-collect-only.txt` | Raw pytest collection transcript. |
| `audit-inventory.json` | Stable JSON summary of tracked files, collected files, collected tests, optional import-skip files, explicit non-test modules, and unexpected gaps. |

Use the same command locally when adding or moving tests:

```bash
PYTHONPATH=src:. python -m pytest tests/ --collect-only -q | tee audit-collect-only.txt
python tools/test_inventory_audit.py \
  --repo . \
  --collect-output audit-collect-only.txt \
  --output audit-inventory.json
```

This cadence detects drift in the test inventory between larger audit sweeps.
Runtime correctness remains owned by the normal CI matrix, optional-dependency
lanes, perf-gated selector, and focused tests attached to each code change.
