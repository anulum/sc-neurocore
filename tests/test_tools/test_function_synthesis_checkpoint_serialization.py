# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete large native checkpoint serialization

"""Qualify large real checkpoints without discarding source data or raising parser limits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tools.function_synthesis_json import MAX_NETLIST_BYTES, native_netlist_artifacts
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_proof import _FLOWS
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_residual_exports import export_pre_lut, verify_pre_lut_exports
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    check_mapped_equivalence,
    read_function_synthesis_measurement,
    run_function_synthesis,
)


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Read actual file identities without changing retained proof evidence."""
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def normalized_arctan(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[SynthesisCase, Path, SynthesisToolchain]:
    """Acquire and read the complete public arctangent under one unchanged caller deadline."""
    root = tmp_path_factory.mktemp("actual-large-arctan-checkpoint")
    deadline = time.monotonic() + 900
    tools = SynthesisToolchain.inspect(root / "inspection")
    bounded = replace(tools, cohort_deadline=deadline)
    case = SynthesisCase("arctan", 32, 16)
    directory = root / "measurement"
    measurement = run_function_synthesis(case, directory, bounded)
    before = _snapshot(directory)
    assert read_function_synthesis_measurement(case, directory, bounded) == measurement
    assert _snapshot(directory) == before
    assert time.monotonic() <= deadline
    return case, directory, tools


def test_complete_arctan_retains_both_mapping_components_within_parser_budget(
    normalized_arctan: tuple[SynthesisCase, Path, SynthesisToolchain],
) -> None:
    """Exercise the public measurement and both complete native mapping components."""
    case, root, tools = normalized_arctan
    directory = root / "composition-proof"
    assert (directory / "pre-lut.json").stat().st_size <= MAX_NETLIST_BYTES
    for stage in ("word-mapping-proof", "primitive-proof"):
        native = json.loads((directory / stage / "gia-proof.json").read_text())
        assert native["status"] == "equivalent"
        for name in ("aligned-word.json", "word-cut.json", "physical-cut.json"):
            assert (directory / stage / name).stat().st_size <= MAX_NETLIST_BYTES
    before = _snapshot(root)
    assert read_function_synthesis_measurement(case, root, tools).proof_method == "compositional"
    assert _snapshot(root) == before


@pytest.fixture(scope="module")
def oversized_native_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Netlist, SynthesisToolchain]:
    """Acquire an actually oversized native checkpoint for the inspected SDK."""
    root = tmp_path_factory.mktemp("actual-oversized-pre-lut")
    tools = SynthesisToolchain.inspect(root / "inspection")
    bounded = replace(tools, cohort_deadline=time.monotonic() + 900)
    legacy = dict(tools.versions)["yosys"].startswith("Yosys 0.33 ")
    if legacy:
        filename = "arcsin_q16_16_before_bounded_division.v"
        original_sha256 = "f2f5b4e139b199f6921276add6d5ca2cbb80e95dae2a813f27ed9ab339d37868"
    else:
        filename = "arctan_q16_16_before_bounded_reciprocal.v"
        original_sha256 = "ac32e0e1d9ebacddefe86eeacf7693e245beaa87596732ce89c606c6c2342d4f"
    # Each genuine prior compiler input exceeded the native parser budget on this SDK.
    fixture = Path(__file__).resolve().parents[1] / "fixtures/function_synthesis" / filename
    rtl = fixture.read_bytes()
    assert hashlib.sha256(rtl).hexdigest() == original_sha256
    (root / "unit.v").write_bytes(rtl)
    flow = "post-dsp.ys"
    (root / flow).write_text(_FLOWS[flow])
    bounded.run(
        [str(tools.yosys), "-Q", "-T", "-s", flow],
        root,
        root / "post-dsp-preparation.log",
        timeout=90,
    )
    bounded.check_library_log(root / "post-dsp-preparation.log")
    post = Netlist.read(root / "post-dsp.json")
    export_pre_lut(post, root, bounded, 90)
    normalized = root / "pre-lut-native"
    assert native_netlist_artifacts(normalized)
    record = json.loads((normalized / "mapped-json-normalization.json").read_text())
    assert record["original_bytes"] > MAX_NETLIST_BYTES
    assert record["compact_bytes"] <= record["parser_budget_bytes"] == MAX_NETLIST_BYTES
    assert (root / "pre-lut.json").read_bytes() == (normalized / "mapped.json").read_bytes()
    before = _snapshot(root)
    verify_pre_lut_exports(post, root, bounded)
    assert _snapshot(root) == before
    return root, post, tools


@pytest.mark.parametrize("damage", ["record", "raw", "checkpoint", "directory-link"])
def test_changed_checkpoint_serialization_is_refused_without_restamping(
    oversized_native_checkpoint: tuple[Path, Netlist, SynthesisToolchain],
    tmp_path: Path,
    damage: str,
) -> None:
    """Bind retained normalization to the actual large checkpoint and its original native bytes."""
    original, post, tools = oversized_native_checkpoint
    root = tmp_path / "changed-checkpoint"
    shutil.copytree(original, root, copy_function=os.link)
    normalized = root / "pre-lut-native"
    if damage == "directory-link":
        target = tmp_path / "normalization-target"
        normalized.rename(target)
        normalized.symlink_to(target, target_is_directory=True)
    else:
        paths = {
            "record": normalized / "mapped-json-normalization.json",
            "raw": normalized / "mapped-native.json",
            "checkpoint": root / "pre-lut.json",
        }
        path = paths[damage]
        payload = path.read_bytes()
        if damage == "record":
            value = json.loads(payload)
            value["original_sha256"] = "0" * 64
            payload = json.dumps(value).encode()
        else:
            payload += b"\n"
        path.unlink()
        path.write_bytes(payload)
    before, source = _snapshot(root), _snapshot(original)
    with pytest.raises(ValueError):
        verify_pre_lut_exports(post, root, tools)
    assert _snapshot(root) == before
    assert _snapshot(original) == source


def test_expired_composition_never_starts_a_new_full_budget_fallback(
    normalized_arctan: tuple[SynthesisCase, Path, SynthesisToolchain], tmp_path: Path
) -> None:
    """Exhaust real admission time and retain the inherited deadline at the public fallback boundary."""
    case, original, tools = normalized_arctan
    root = tmp_path / "expired-admission"
    root.mkdir()
    for path in original.iterdir():
        if path.is_file():
            os.link(path, root / path.name)
    delayed = False

    def expire_actual_admission(event: str, args: tuple[object, ...]) -> None:
        """Wait only at this actual proof directory's first native-admission boundary."""
        nonlocal delayed
        if (
            not delayed
            and event == "os.mkdir"
            and args[0] in (root / "composition-proof", str(root / "composition-proof"))
        ):
            delayed = True
            time.sleep(10.1)

    expire_actual_admission.__dict__["__cantrace__"] = True
    sys.addaudithook(expire_actual_admission)
    before = _snapshot(original)
    with pytest.raises(
        ValueError, match="Mapped proof validation exhausted the shared cohort budget"
    ):
        check_mapped_equivalence(case, root, tools, timeout=1.0, cohort_seconds=10.0)
    assert delayed and not (root / "complete-bdd-proof").exists()
    assert not list((root / "composition-proof").glob("*.command.json"))
    assert _snapshot(original) == before
