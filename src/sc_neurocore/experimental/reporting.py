# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Report helpers for the experimental alternative-path harness

"""JSON report helpers for experimental alternative-path batch runs.

The module keeps report file naming deterministic and writes
``AlternativePathBatchSummary`` payloads using the same sorted, indented JSON
shape consumed by benchmark evidence and documentation examples.
"""

from __future__ import annotations

import json
from pathlib import Path

from .alternative_path import AlternativePathBatchSummary


def default_report_path(route_name: str) -> Path:
    """Return the default JSON report path for a route."""
    safe_name = route_name.replace(".", "_").replace("-", "_")
    return Path("benchmarks/results") / f"experimental_{safe_name}.json"


def write_batch_report(summary: AlternativePathBatchSummary, path: str | Path) -> Path:
    """Write a batch summary to a JSON report file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_report(), indent=2, sort_keys=True) + "\n")
    return out_path
