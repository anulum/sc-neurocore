# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence reference grammar, resolution and digests

"""Evidence reference tokens are typed, resolved on disk and never dropped."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sc_neurocore.neurons.evidence_references import (
    classify_reference,
    parse_evidence_field,
    resolve_reference,
    sha256_canonical_json,
    sha256_file,
    sha256_tree,
    split_evidence_field,
    node_is_defined,
)

_TEST_MODULE = """
def test_top_level() -> None:
    pass


class TestGroup:
    def test_member(self) -> None:
        pass

    def helper(self) -> None:
        pass
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_probe.py").write_text(_TEST_MODULE, encoding="utf-8")
    (tmp_path / "hdl").mkdir()
    (tmp_path / "hdl" / "report.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_split_keeps_every_non_empty_token_in_order() -> None:
    """Semicolon-separated fields split into stripped tokens; empties vanish."""
    assert split_evidence_field(" a.py ; b.py;; c ") == ("a.py", "b.py", "c")
    assert split_evidence_field("") == ()
    assert split_evidence_field("   ") == ()


def test_classification_covers_every_kind() -> None:
    """Node ids, test files, artefacts, inline JSON and prose are told apart."""
    assert classify_reference("tests/test_x.py::TestA::test_b[param-1]") == (
        "test-node",
        "tests/test_x.py",
        "TestA::test_b[param-1]",
    )
    assert classify_reference("tests/test_x.py") == ("test-file", "tests/test_x.py", "")
    assert classify_reference("hdl/reports/yosys.json") == (
        "artifact-file",
        "hdl/reports/yosys.json",
        "",
    )
    assert classify_reference('{"kind":"sampled_batch_v1","n":3}') == ("inline-config", "", "")
    assert classify_reference("{not json") == ("free-text", "", "")
    assert classify_reference("compile-only or map-suite separate gates") == ("free-text", "", "")
    assert classify_reference("three-way exact per enrolment commit") == ("free-text", "", "")


def test_resolution_reports_missing_files_and_nodes(tmp_path: Path) -> None:
    """A named file or node that does not exist stays visible as missing."""
    repo = _repo(tmp_path)
    assert resolve_reference("tests/test_probe.py::test_top_level", repo).resolution == "resolved"
    assert resolve_reference("tests/test_probe.py::TestGroup::test_member", repo).is_resolved
    assert resolve_reference("tests/test_probe.py::TestGroup::test_member[x]", repo).is_resolved
    assert resolve_reference("tests/test_probe.py::TestGroup", repo).is_resolved
    assert resolve_reference("tests/test_probe.py::test_gone", repo).resolution == "missing-node"
    assert (
        resolve_reference("tests/test_probe.py::TestGroup::test_gone", repo).resolution
        == "missing-node"
    )
    assert resolve_reference("tests/test_absent.py", repo).resolution == "missing-file"
    assert resolve_reference("hdl/report.json", repo).resolution == "resolved"
    assert resolve_reference("hdl/absent.json", repo).resolution == "missing-file"
    prose = resolve_reference("exact three-way spike parity", repo)
    assert prose.resolution == "unresolvable"
    assert not prose.is_locatable


def test_parse_field_returns_one_reference_per_token(tmp_path: Path) -> None:
    """Mixed fields keep every token with its own resolution."""
    repo = _repo(tmp_path)
    references = parse_evidence_field(
        "tests/test_probe.py::test_top_level; tests/test_absent.py; prose here", repo
    )
    assert [reference.kind for reference in references] == ["test-node", "test-file", "free-text"]
    assert [reference.resolution for reference in references] == [
        "resolved",
        "missing-file",
        "unresolvable",
    ]
    assert references[0].to_public_dict()["node"] == "test_top_level"


def test_node_lookup_survives_a_rewritten_module(tmp_path: Path) -> None:
    """The definition cache keys on modification time, so edits are seen."""
    repo = _repo(tmp_path)
    module = repo / "tests" / "test_probe.py"
    assert node_is_defined(module, "test_top_level")
    module.write_text("def test_other() -> None:\n    pass\n", encoding="utf-8")
    import os

    stat = module.stat()
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert not node_is_defined(module, "test_top_level")
    assert node_is_defined(module, "test_other")
    module.write_text("def broken(:\n", encoding="utf-8")
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000))
    assert not node_is_defined(module, "broken")


def test_digests_are_content_bound_and_order_free(tmp_path: Path) -> None:
    """File, tree and canonical-JSON digests depend on content only."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    assert sha256_file(first) == hashlib.sha256(b"alpha").hexdigest()
    forward = sha256_tree([first, second], tmp_path)
    assert forward == sha256_tree([second, first], tmp_path)
    second.write_bytes(b"gamma")
    assert sha256_tree([first, second], tmp_path) != forward
    assert sha256_canonical_json({"b": 1, "a": [1, 2]}) == sha256_canonical_json(
        {"a": [1, 2], "b": 1}
    )
    assert sha256_canonical_json({"a": 1}) != sha256_canonical_json({"a": 2})
