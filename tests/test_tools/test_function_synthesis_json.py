# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native JSON acquisition and retained identity contracts

"""Exercise lossless acquisition and adverse retained artifacts using actual Yosys output."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_exports import SYNTHESIS_FLOW
from tools.function_synthesis_json import (
    MAX_NETLIST_BYTES,
    decode_native_json,
    native_netlist_artifacts,
    normalize_native_netlist,
)
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_toolchain import SynthesisToolchain


@pytest.mark.parametrize("binary", [False, True])
def test_public_native_decoder_normalizes_real_stack_exhaustion(binary: bool) -> None:
    """Normalize actual parser stack exhaustion without relying on JSON nesting heuristics."""
    assert decode_native_json('{"complete": [0, 1]}') == {"complete": [0, 1]}
    script = """import json
import sys
from tools.function_synthesis_json import decode_native_json

sys.setrecursionlimit(128)
frame, frames = sys._getframe(), 0
while frame is not None:
    frames += 1
    frame = frame.f_back
text = '{"complete": [0, 1]}'
payload = text.encode() if sys.argv[1] == "1" else text
assert decode_native_json(payload) == {"complete": [0, 1]}

def recursive_caller(depth):
    if depth:
        return recursive_caller(depth - 1)
    return decode_native_json(payload)

try:
    recursive_caller(sys.getrecursionlimit() - frames - 3)
except ValueError as error:
    assert str(error) == "Native JSON exceeds the decoder nesting budget"
    assert isinstance(error.__cause__, RecursionError)
else:
    raise AssertionError("Real exhausted caller stack was accepted")
print(json.dumps({"normalized_recursion_error": True}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, "1" if binary else "0"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"normalized_recursion_error": True}


def _sha(path: Path) -> str:
    """Read the identity of the actual retained artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(scope="module")
def native_wire(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Acquire genuine native library JSON with quoted source metadata and a complete wire."""
    root = tmp_path_factory.mktemp("normalization-native-wire")
    tools = SynthesisToolchain.inspect(root / "inspection")
    rtl = emit_function_unit_verilog("floor", data_width=8, fraction=0)
    label = json.dumps('actual "quoted" label with a \\ path and spaces ')
    rtl = rtl.replace(
        "module function_unit", f"(* audit_label = {label} *)\nmodule function_unit", 1
    )
    (root / "unit.v").write_text(rtl)
    (root / "flow.ys").write_text(SYNTHESIS_FLOW)
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-s", "flow.ys"],
        root,
        root / "yosys.log",
        timeout=90,
    )
    tools.check_library_log(root / "yosys.log")
    return root


@pytest.fixture(scope="module")
def normalized_wire(native_wire: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Normalize a genuine copied acquisition once for retained-reader controls."""
    root = tmp_path_factory.mktemp("normalized-native-wire")
    shutil.copyfile(native_wire / "mapped.json", root / "mapped.json")
    normalize_native_netlist(root)
    return root


def test_complete_native_values_and_original_bytes_are_preserved(
    native_wire: Path, normalized_wire: Path
) -> None:
    """Preserve every library field, source string, cell, vector and original byte."""
    raw = native_wire / "mapped.json"
    before = raw.read_bytes(), raw.stat().st_mtime_ns
    assert native_netlist_artifacts(native_wire) == ()
    assert native_netlist_artifacts(normalized_wire) == (
        "mapped-native.json",
        "mapped-json-normalization.json",
    )
    compact = normalized_wire / "mapped.json"
    original = normalized_wire / "mapped-native.json"
    assert original.read_bytes() == before[0]
    assert json.loads(compact.read_bytes()) == json.loads(before[0])
    assert compact.stat().st_size < original.stat().st_size <= MAX_NETLIST_BYTES
    assert Netlist.read(compact).clone_module() == Netlist.read(raw).clone_module()
    assert before == (raw.read_bytes(), raw.stat().st_mtime_ns)
    snapshot = {p.name: (_sha(p), p.stat().st_mtime_ns) for p in normalized_wire.iterdir()}
    assert native_netlist_artifacts(normalized_wire)
    assert snapshot == {p.name: (_sha(p), p.stat().st_mtime_ns) for p in normalized_wire.iterdir()}


@pytest.mark.parametrize("operation", [normalize_native_netlist, native_netlist_artifacts])
def test_linked_acquisition_directory_is_refused(
    native_wire: Path, tmp_path: Path, operation: object
) -> None:
    """Refuse real directory links through both public acquisition and retained entry points."""
    link = tmp_path / "linked"
    link.symlink_to(native_wire, target_is_directory=True)
    if operation is normalize_native_netlist:
        with pytest.raises(ValueError, match="directory link"):
            normalize_native_netlist(link)
    else:
        with pytest.raises(ValueError, match="directory link"):
            native_netlist_artifacts(link)


@pytest.mark.parametrize("kind", ["missing", "linked"])
def test_missing_or_linked_original_is_refused(
    native_wire: Path, tmp_path: Path, kind: str
) -> None:
    """Refuse unavailable original data without opening a linked native artifact."""
    if kind == "linked":
        (tmp_path / "mapped.json").symlink_to(native_wire / "mapped.json")
    with pytest.raises(ValueError, match="regular unlinked"):
        normalize_native_netlist(tmp_path)


@pytest.mark.parametrize(
    "name", ["mapped-native.json", "mapped-json-normalization.json", "mapped-compact.json"]
)
@pytest.mark.parametrize("kind", ["existing", "dangling"])
def test_existing_normalization_evidence_cannot_be_overwritten(
    native_wire: Path, tmp_path: Path, name: str, kind: str
) -> None:
    """Preserve pre-existing regular records and dangling links on actual acquisition."""
    shutil.copyfile(native_wire / "mapped.json", tmp_path / "mapped.json")
    target = tmp_path / name
    if kind == "existing":
        target.write_text("retained owner data")
    else:
        target.symlink_to(tmp_path / "absent-target")
    before = _sha(tmp_path / "mapped.json")
    with pytest.raises(ValueError, match="must not overwrite"):
        normalize_native_netlist(tmp_path)
    assert _sha(tmp_path / "mapped.json") == before
    assert (
        target.is_symlink() if kind == "dangling" else target.read_text() == "retained owner data"
    )


@pytest.mark.parametrize("damage", ["raw", "compact", "record"])
def test_changed_retained_representation_is_refused(
    normalized_wire: Path, tmp_path: Path, damage: str
) -> None:
    """Refuse changed original bytes, normalized bytes and the retained identity record."""
    shutil.copytree(normalized_wire, tmp_path / "retained")
    root = tmp_path / "retained"
    names = {
        "raw": "mapped-native.json",
        "compact": "mapped.json",
        "record": "mapped-json-normalization.json",
    }
    path = root / names[damage]
    if damage == "record":
        path.write_text("[]")
    else:
        with path.open("ab") as stream:
            stream.write(b"\n")
    before = {p.name: _sha(p) for p in root.iterdir()}
    with pytest.raises(ValueError, match="differs from its retained record"):
        native_netlist_artifacts(root)
    assert before == {p.name: _sha(p) for p in root.iterdir()}


@pytest.mark.parametrize("damage", ["record", "raw", "compact", "record-link"])
def test_incomplete_or_linked_retained_evidence_is_refused(
    normalized_wire: Path, tmp_path: Path, damage: str
) -> None:
    """Refuse partial or linked actual normalization artifacts."""
    root = tmp_path / "retained"
    shutil.copytree(normalized_wire, root)
    name = {
        "record": "mapped-json-normalization.json",
        "raw": "mapped-native.json",
        "compact": "mapped.json",
        "record-link": "mapped-json-normalization.json",
    }[damage]
    (root / name).unlink()
    if damage == "record-link":
        (root / name).symlink_to(tmp_path / "absent-record")
    with pytest.raises(ValueError):
        native_netlist_artifacts(root)


@pytest.mark.parametrize("kind", ["regular", "dangling"])
def test_orphan_native_original_is_not_legacy_evidence(tmp_path: Path, kind: str) -> None:
    """Keep a partial normalization distinct from an original unnormalized acquisition."""
    raw = tmp_path / "mapped-native.json"
    if kind == "regular":
        raw.write_text("{}")
    else:
        raw.symlink_to(tmp_path / "missing-native-data")
    with pytest.raises(ValueError, match="incomplete"):
        native_netlist_artifacts(tmp_path)


@pytest.mark.parametrize("damage", ["record", "compact"])
def test_retained_artifact_budgets_stay_unchanged(
    normalized_wire: Path, tmp_path: Path, damage: str
) -> None:
    """Refuse oversized retained metadata and compact JSON without increasing either budget."""
    root = tmp_path / "retained"
    shutil.copytree(normalized_wire, root)
    path, limit = (
        (root / "mapped-json-normalization.json", 65536)
        if damage == "record"
        else (root / "mapped.json", MAX_NETLIST_BYTES)
    )
    with path.open("r+b") as stream:
        stream.truncate(limit + 1)
    with pytest.raises(ValueError, match="oversized|unchanged parser budget"):
        native_netlist_artifacts(root)


@pytest.mark.parametrize("tail", [b'"unfinished', b'"unfinished\\'])
def test_unterminated_native_string_preserves_original(
    native_wire: Path, tmp_path: Path, tail: bytes
) -> None:
    """Retain a genuinely damaged native acquisition instead of emitting a success record."""
    source = tmp_path / "mapped.json"
    shutil.copyfile(native_wire / "mapped.json", source)
    with source.open("ab") as stream:
        stream.write(tail)
    before = _sha(source)
    with pytest.raises(ValueError, match="unterminated"):
        normalize_native_netlist(tmp_path)
    assert _sha(source) == before
    assert not (tmp_path / "mapped-json-normalization.json").exists()


def test_malformed_native_json_preserves_original(native_wire: Path, tmp_path: Path) -> None:
    """Reject a truncated native JSON object after token-preserving serialization."""
    source = tmp_path / "mapped.json"
    source.write_bytes((native_wire / "mapped.json").read_bytes().rstrip()[:-1])
    before = _sha(source)
    with pytest.raises(ValueError):
        normalize_native_netlist(tmp_path)
    assert _sha(source) == before


@pytest.mark.parametrize("value", [b"1 2", b"t rue", b"nul l"])
@pytest.mark.parametrize("cross_chunk", [False, True])
def test_separated_native_value_tokens_cannot_be_merged(
    native_wire: Path, tmp_path: Path, value: bytes, cross_chunk: bool
) -> None:
    """Refuse corrupted real native values, including separators across stream chunks."""
    raw = (native_wire / "mapped.json").read_bytes()
    marker = b'"hide_name": '
    offset = raw.index(marker) + len(marker)
    assert raw[offset : offset + 1] in {b"0", b"1"}
    damaged = raw[:offset] + value + raw[offset + 1 :]
    if cross_chunk:
        padding = (65535 - offset) % 65536
        damaged = b" " * padding + damaged
        assert (padding + offset) % 65536 == 65535
    with pytest.raises(json.JSONDecodeError):
        json.loads(damaged)
    source = tmp_path / "mapped.json"
    source.write_bytes(damaged)
    before = _sha(source), source.stat().st_mtime_ns
    with pytest.raises(ValueError, match="separates distinct value tokens"):
        normalize_native_netlist(tmp_path)
    assert (_sha(source), source.stat().st_mtime_ns) == before
    assert not (tmp_path / "mapped-json-normalization.json").exists()
    assert not (tmp_path / "mapped-native.json").exists()


def test_compaction_does_not_raise_the_parser_budget(native_wire: Path, tmp_path: Path) -> None:
    """Bound a corrupt oversized native file before loading its compact representation."""
    source = tmp_path / "mapped.json"
    shutil.copyfile(native_wire / "mapped.json", source)
    with source.open("r+b") as stream:
        stream.truncate(MAX_NETLIST_BYTES + source.stat().st_size)
    before = _sha(source)
    with pytest.raises(ValueError, match="unchanged parser budget"):
        normalize_native_netlist(tmp_path)
    assert _sha(source) == before
    assert not (tmp_path / "mapped-json-normalization.json").exists()


def test_source_change_during_real_serialization_is_refused(
    native_wire: Path, tmp_path: Path
) -> None:
    """Reject an actual source edit at the complete native serialization output boundary."""
    source = tmp_path / "mapped.json"
    shutil.copyfile(native_wire / "mapped.json", source)
    changed = False
    compact = tmp_path / "mapped-compact.json"

    def change_source(event: str, args: tuple[object, ...]) -> None:
        """Append actual whitespace after source binding, when the real compact output opens."""
        nonlocal changed
        if (
            event == "open"
            and args[0] in (compact, str(compact), str(compact).encode())
            and not changed
        ):
            changed = True
            with source.open("ab") as stream:
                stream.write(b"\n")

    change_source.__dict__["__cantrace__"] = True
    sys.addaudithook(change_source)
    with pytest.raises(ValueError, match="changed during normalization"):
        normalize_native_netlist(tmp_path)
    assert changed
    assert source.is_file() and not (tmp_path / "mapped-native.json").exists()
