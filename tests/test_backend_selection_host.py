# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused backend selection contracts

"""Focused data-driven backend selection contracts."""

from .backend_selection_support import *


def test_current_cpu_is_nonempty_string() -> None:
    cpu = bs.current_cpu()
    assert isinstance(cpu, str) and cpu


def test_current_cpu_uses_platform_processor_without_proc_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host detection falls back to ``platform.processor`` without model-name data."""

    def read_cpuinfo_without_model(
        _path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        assert encoding == "utf-8"
        assert errors is None
        return "processor\t: 0\nvendor_id\t: GenuineIntel\n"

    monkeypatch.setattr(Path, "read_text", read_cpuinfo_without_model)
    monkeypatch.setattr(platform, "processor", lambda: "portable-cpu")

    assert bs.current_cpu() == "portable-cpu"


def test_current_cpu_returns_unknown_when_proc_and_platform_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host detection keeps a deterministic fallback when CPU probes fail."""

    def raise_cpuinfo_oserror(
        _path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        assert encoding == "utf-8"
        assert errors is None
        raise OSError("cpuinfo unavailable")

    monkeypatch.setattr(Path, "read_text", raise_cpuinfo_oserror)
    monkeypatch.setattr(platform, "processor", lambda: "")

    assert bs.current_cpu() == "unknown"
