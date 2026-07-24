# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_snn_memory_discipline_audit.py

from __future__ import annotations

"""Support extracted from test_snn_memory_discipline_audit.py."""

import json


import runpy


import subprocess


import sys


from pathlib import Path


import pytest


from tools import snn_memory_discipline_audit as audit_tool


def _canonical_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "actor": "codex",
        "content": "SC-NEUROCORE canonical memory discipline fixture.",
        "entities": ["SC-NEUROCORE"],
        "kind": "event",
        "project": "SC-NEUROCORE",
        "source_ref": "tests/test_tools/test_snn_memory_discipline_audit.py",
        "timestamp": 1783617021,
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _actor_detail() -> str:
    return f"actor must be one of {sorted(audit_tool.CONTROLLED_ACTORS)}"



__all__ = ['json', 'runpy', 'subprocess', 'sys', 'Path', 'pytest', 'audit_tool', '_canonical_payload', '_write_json', '_actor_detail']
