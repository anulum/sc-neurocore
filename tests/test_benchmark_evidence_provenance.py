# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Committed benchmark evidence names what produced it

"""Committed benchmark evidence must say what produced it, truthfully.

Two ways a published cross-language comparison can be wrong without any test
noticing, both found in the committed corpus and both fixed by the change these
cases guard.

The first is an unoptimised binary. Six records bound a 525 MB Rust extension
where every other record bound the ~35 MB release build. A debug build is
several times slower, so those records rated the Rust lane *last* of four
native lanes when a release build puts it first or second — the numbers were
real measurements of the wrong artefact.

The second is lost provenance. The shared runner asked for compiler versions
only through ``.venv/bin``, where three of the four shims do not exist, so every
rerun silently recorded ``unavailable`` for compilers that were installed and in
use. Nothing failed; the record simply stopped saying what built it.

These cases speak only about records that carry the field in question, so a
simpler older record is not retrofitted with a claim it never made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"

#: Compilers a record names when it carries a toolchain block.
TOOLCHAIN_KEYS = ("rustc", "go", "julia", "mojo")

#: Values the version probe writes when it could not run the tool at all.
UNRECORDED = ("", "unavailable")

#: A release build of the engine extension is ~35 MB and a debug build ~525 MB.
#: The bound sits far from both, so it survives ordinary release-size drift and
#: still refuses any record measured against an unoptimised binary.
RELEASE_EXTENSION_CEILING_BYTES = 100 * 1024 * 1024


def _records() -> list[tuple[str, dict[str, Any]]]:
    """Return every committed benchmark record as ``(name, payload)``."""
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(RESULTS.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            found.append((path.name, payload))
    return found


def _toolchain_block(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the record's toolchain block, whichever schema names it."""
    for key in ("tool_versions", "environment"):
        block = payload.get(key)
        if isinstance(block, dict) and any(tool in block for tool in TOOLCHAIN_KEYS):
            return block
    return {}


def _rust_extension_bytes(payload: dict[str, Any]) -> int | None:
    """Return the size of the Rust extension a record bound, if it bound one."""
    binaries = payload.get("binary_hashes")
    if not isinstance(binaries, dict):
        return None
    extension = binaries.get("rust_extension")
    if not isinstance(extension, dict):
        return None
    size = extension.get("size_bytes")
    return size if isinstance(size, int) else None


ALL_RECORDS = _records()


def test_the_corpus_has_records_to_check() -> None:
    """A gate that found nothing to check would pass for the wrong reason."""
    assert len(ALL_RECORDS) > 100


class TestNoRecordQuotesAnUnoptimisedBinary:
    def test_every_bound_rust_extension_is_a_release_build(self) -> None:
        """The defect: a lane comparison measured against a debug build."""
        oversized = {
            name: size
            for name, payload in ALL_RECORDS
            if (size := _rust_extension_bytes(payload)) is not None
            and size > RELEASE_EXTENSION_CEILING_BYTES
        }
        assert oversized == {}

    def test_some_record_actually_binds_an_extension(self) -> None:
        """Otherwise the case above is vacuous and would never fail."""
        bound = [name for name, payload in ALL_RECORDS if _rust_extension_bytes(payload)]
        assert bound


class TestEveryRecordNamesWhatProducedIt:
    def test_no_toolchain_block_leaves_a_compiler_unrecorded(self) -> None:
        """A rerun that cannot find a compiler must fail loudly, not quietly."""
        unrecorded: dict[str, list[str]] = {}
        for name, payload in ALL_RECORDS:
            block = _toolchain_block(payload)
            missing = sorted(
                tool
                for tool in TOOLCHAIN_KEYS
                if tool in block
                and (
                    not isinstance(block[tool], str)
                    or block[tool] in UNRECORDED
                    or str(block[tool]).startswith("exit ")
                )
            )
            if missing:
                unrecorded[name] = missing
        assert unrecorded == {}

    def test_some_record_actually_carries_a_toolchain_block(self) -> None:
        """Otherwise the case above is vacuous and would never fail."""
        carriers = [name for name, payload in ALL_RECORDS if _toolchain_block(payload)]
        assert len(carriers) >= 25


class TestTheRunnerResolvesAPinnedToolchain:
    @pytest.mark.parametrize(
        "module_name",
        ["benchmarks._non_resetting_lif_benchmark", "benchmarks.bench_compte_wm"],
    )
    def test_a_present_shim_is_preferred_over_the_path(
        self, module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A project that pins a toolchain pins it in the venv; use that one."""
        import importlib

        module = importlib.import_module(module_name)
        shim = tmp_path / ".venv" / "bin" / "rustc"
        shim.parent.mkdir(parents=True)
        shim.write_text("", encoding="utf-8")
        monkeypatch.setattr(module, "REPOSITORY", tmp_path)
        assert module._toolchain_command("rustc", "--version") == [str(shim), "--version"]

    @pytest.mark.parametrize(
        "module_name",
        ["benchmarks._non_resetting_lif_benchmark", "benchmarks.bench_compte_wm"],
    )
    def test_an_absent_shim_falls_back_to_the_path(
        self, module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is the half that was missing: no shim must not mean no answer."""
        import importlib

        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "REPOSITORY", tmp_path)
        assert module._toolchain_command("rustc", "--version") == ["rustc", "--version"]
