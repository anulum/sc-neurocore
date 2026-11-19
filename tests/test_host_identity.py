# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One host CPU model rule for dispatch, conversion and Studio

"""Hold dispatch, the dense IF comparison and Studio's contribution to one host CPU model rule.

The ``model name`` line of ``/proc/cpuinfo``, else :func:`platform.processor`, else ``"unknown"``;
an unreadable or undecodable file falls back the same way. The dense IF comparison used to raise
``StopIteration`` on a host without a ``model name`` line, and reading the rule must not pull the
acceleration package into the conversion module.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore import _host_identity
from sc_neurocore.accel import backend_selection
from sc_neurocore.conversion import if_benchmark_order
from sc_neurocore.studio import benchmark_contribution


def _cpuinfo(tmp_path: Path, text: str | bytes) -> Path:
    """Write a ``/proc/cpuinfo`` stand-in and return its path."""
    path = tmp_path / "cpuinfo"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_the_model_name_line_names_the_cpu(tmp_path: Path) -> None:
    """The first ``model name`` value, stripped, is the model."""
    info = "processor\t: 0\nmodel name\t: AMD EPYC 7302 16-Core Processor\nmodel name\t: other\n"
    assert _host_identity.cpu_model(_cpuinfo(tmp_path, info)) == "AMD EPYC 7302 16-Core Processor"


@pytest.mark.parametrize(
    "text",
    [
        "processor\t: 0\nvendor_id\t: GenuineIntel\n",
        "model name\n",
        "model name\t:   \n",
        b"\xff\xfe",
    ],
    ids=["no-model-line", "no-colon", "empty-value", "not-utf8"],
)
def test_a_missing_or_unusable_model_line_falls_back_to_the_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str | bytes
) -> None:
    """No line, a line without a value, or undecodable bytes give ``platform.processor``."""
    monkeypatch.setattr(platform, "processor", lambda: "portable-cpu")
    assert _host_identity.cpu_model(_cpuinfo(tmp_path, text)) == "portable-cpu"


def test_a_bad_byte_elsewhere_keeps_the_model_line(tmp_path: Path) -> None:
    """One undecodable byte in another line does not cost the model name."""
    info = b"processor\t: 0\nflags\t: \xff\xfe\nmodel name\t: Ampere Altra\n"
    assert _host_identity.cpu_model(_cpuinfo(tmp_path, info)) == "Ampere Altra"


def test_a_file_that_vanishes_after_a_bad_read_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the byte-level reread fails, the platform answers, as for an unreadable file."""

    def vanished(_path: Path) -> bytes:
        raise OSError("cpuinfo vanished")

    path = _cpuinfo(tmp_path, b"model name\t: \xff\n")
    monkeypatch.setattr(Path, "read_bytes", vanished)
    monkeypatch.setattr(platform, "processor", lambda: "portable-cpu")
    assert _host_identity.cpu_model(path) == "portable-cpu"


def test_an_absent_file_and_an_empty_platform_give_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing to read anywhere, the model is ``"unknown"``."""
    monkeypatch.setattr(platform, "processor", lambda: "")
    assert _host_identity.cpu_model(tmp_path / "absent") == "unknown"


def test_dispatch_studio_and_conversion_read_the_same_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patching the one rule changes what dispatch and Studio report."""
    monkeypatch.setattr(_host_identity, "cpu_model", lambda cpuinfo=None: "one-rule-cpu")
    assert backend_selection.current_cpu() == "one-rule-cpu"
    assert benchmark_contribution.safe_environment()["cpu"] == "one-rule-cpu"


def test_the_dense_if_comparison_uses_the_static_order_on_another_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host whose model differs, including one with no ``model name`` line, keeps the static order."""
    record = tmp_path / "comparison.json"
    record.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.dense-if-comparison.v1",
                "kernel": "dense-if-f64-sequential-v1",
                "meta": {"cpu": "the measuring host"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SC_NEUROCORE_IF_BENCHMARK", str(record))
    monkeypatch.setattr(_host_identity, "cpu_model", lambda cpuinfo=None: "portable-cpu")
    assert if_benchmark_order.measured_order() == if_benchmark_order.STATIC_ORDER


def test_reading_the_rule_does_not_import_the_acceleration_package() -> None:
    """The dense IF comparison promises no native runtime import; the rule keeps that true."""
    probe = (
        "import sys\n"
        "import sc_neurocore.conversion.if_benchmark_order\n"
        "print('sc_neurocore.accel' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "False"
