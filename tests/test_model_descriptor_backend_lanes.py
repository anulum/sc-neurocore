# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Backend lanes of a model descriptor: their names, their evidence and their sources

"""Bind what a model descriptor may say about a backend lane.

A lane table names one of six lanes, or the one legacy name that is still
accepted. Beside its status and parity word it may name the tests that are its
evidence and the tracked files that implement the model in that lane. A name
outside the list, evidence that is not text and a source that is not a path
inside the repository are refused, so a lane that no facet can see and a source
that no receipt could bind cannot be written.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from sc_neurocore.neurons.descriptor_generator import merge_descriptor_payloads
from sc_neurocore.neurons.model_descriptor import (
    BACKEND_LANES,
    BACKEND_NAMES,
    LEGACY_BACKEND_LANES,
    BackendSupport,
    ModelDescriptorError,
    parse_model_descriptor,
)

_DESCRIPTORS = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_descriptors"
# Eleven shipped descriptors still use the legacy name. The number may fall, never rise.
_LEGACY_TABLE_CEILING = 11


def _payload(backends: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal descriptor payload with the given backend section."""
    return {
        "metadata": {
            "schema_version": 2,
            "name": "Leaky demonstration",
            "class_name": "LapicqueNeuron",
            "module": "lapicque",
        },
        "state": {"v": {"init": 0.0}},
        "parameters": {"tau": {"default": 20.0}},
        "integration": {"dt": 0.1},
        "backends": backends,
    }


def _lanes(backends: dict[str, Any]) -> tuple[BackendSupport, ...]:
    """Parse a payload with the given backend section and return its lanes."""
    return parse_model_descriptor(_payload(backends)).backends


def test_the_lanes_are_six_and_one_legacy_name() -> None:
    """The list of lanes is closed; the five kernel lanes are part of it."""
    assert BACKEND_LANES == ("python", "rust", "rust_safety", "julia", "go", "mojo")
    assert LEGACY_BACKEND_LANES == ("rust_engine",)
    assert set(BACKEND_NAMES) < set(BACKEND_LANES)
    assert not set(LEGACY_BACKEND_LANES) & set(BACKEND_LANES)


def test_a_lane_states_its_evidence_and_its_sources() -> None:
    """Evidence is kept as written; sources are kept in their written order."""
    lanes = _lanes(
        {
            "python": {"status": "implemented"},
            "rust": {
                "status": "implemented",
                "parity": "ulp-bounded",
                "evidence": "tests/test_lapicque_backend_parity.py::test_rust; tests/test_x.py",
                "sources": ["engine/src/neuron/lapicque.rs", "engine/src/bindings/lapicque.rs"],
            },
        }
    )
    assert lanes == (
        BackendSupport("python", "implemented", "n/a", "", ()),
        BackendSupport(
            "rust",
            "implemented",
            "ulp-bounded",
            "tests/test_lapicque_backend_parity.py::test_rust; tests/test_x.py",
            ("engine/src/neuron/lapicque.rs", "engine/src/bindings/lapicque.rs"),
        ),
    )


def test_a_lane_without_evidence_or_sources_has_none() -> None:
    """Both fields are optional, in the table form and in the one-word form."""
    lanes = _lanes({"go": {"status": "planned"}, "julia": "unsupported"})
    assert [(lane.name, lane.status, lane.evidence, lane.sources) for lane in lanes] == [
        ("go", "planned", "", ()),
        ("julia", "unsupported", "", ()),
    ]


@pytest.mark.parametrize("key", ["source", "evidnce", "Status", "notes"])
def test_a_key_outside_the_four_is_refused(key: str) -> None:
    """A misspelt key would drop what it states without a word, so every other key is refused.

    Parameters
    ----------
    key : str
        A key that is not one of the four a lane table takes.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"rust": {"status": "implemented", key: "tests/test_x.py"}})
    assert str(caught.value) == (
        f"descriptor backend 'rust' has the key {key!r}; a lane table takes status, parity, "
        "evidence and sources"
    )


@pytest.mark.parametrize(
    "table", [{}, {"parity": "exact"}, {"evidence": "tests/test_x.py"}, {"sources": ["a.rs"]}]
)
def test_a_lane_table_states_its_status(table: dict[str, Any]) -> None:
    """A table without a status is not read as implemented, whatever else it states.

    Parameters
    ----------
    table : dict
        A lane table that lacks its status.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"rust": table})
    assert str(caught.value) == "descriptor backend 'rust' table must state its status"


def test_evidence_is_kept_as_written() -> None:
    """The evidence text is not trimmed or rewritten; resolving it is another module's work."""
    lanes = _lanes(
        {"go": {"status": "implemented", "evidence": " tests/test_x.py ;tests/test_y.py "}}
    )
    assert lanes[0].evidence == " tests/test_x.py ;tests/test_y.py "


def test_a_status_outside_the_vocabulary_is_refused_in_both_forms() -> None:
    """The one-word form and the table form take the same three status words."""
    for backends in ({"rust": "yes"}, {"rust": {"status": "yes"}}):
        with pytest.raises(ModelDescriptorError) as caught:
            _lanes(backends)
        assert str(caught.value) == (
            "descriptor 'backend status' must be one of "
            "['implemented', 'planned', 'unsupported'], got 'yes'"
        )


@pytest.mark.parametrize("name", [*BACKEND_LANES, *LEGACY_BACKEND_LANES])
def test_every_listed_lane_name_is_accepted(name: str) -> None:
    """Each of the six lanes and the legacy name parses under its own name.

    Parameters
    ----------
    name : str
        A lane name of the closed list, or the legacy name.
    """
    assert _lanes({name: {"status": "implemented"}}) == (
        BackendSupport(name, "implemented", "n/a", "", ()),
    )


@pytest.mark.parametrize("name", ["cuda", "Rust", "rust-safety", "python3", ""])
def test_a_lane_name_outside_the_list_is_refused(name: str) -> None:
    """A table under any other name is refused with the list in the message.

    Parameters
    ----------
    name : str
        A table name that is not a lane.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"python": {"status": "implemented"}, name: {"status": "implemented"}})
    assert str(caught.value) == (
        f"descriptor backend lane {name!r} is not a lane; the lanes are python, rust, "
        "rust_safety, julia, go, mojo, and the legacy name rust_engine"
    )


@pytest.mark.parametrize("value", [5, 2.5, True, None, ["implemented"]])
def test_a_lane_that_is_neither_a_table_nor_a_status_word_is_refused(value: object) -> None:
    """A number, a truth value, nothing or a list in place of a lane is refused.

    Parameters
    ----------
    value : object
        What stands where a table or a status word is expected.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"rust": value})
    assert str(caught.value) == (
        f"descriptor backend 'rust' must be a table or a status word, got {value!r}"
    )


@pytest.mark.parametrize("value", [7, ["tests/test_x.py"], {"path": "tests/test_x.py"}, True])
def test_evidence_that_is_not_text_is_refused(value: object) -> None:
    """Evidence is one string in the grammar of evidence fields; nothing else is read as one.

    Parameters
    ----------
    value : object
        What stands where the evidence string is expected.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"rust": {"status": "implemented", "evidence": value}})
    assert (
        str(caught.value) == f"descriptor backend 'rust' evidence must be a string, got {value!r}"
    )


@pytest.mark.parametrize("value", ["engine/src/neuron/lapicque.rs", 3, {"a": "b"}])
def test_sources_that_are_not_a_list_are_refused(value: object) -> None:
    """A single path written as text is not a list of one path.

    Parameters
    ----------
    value : object
        What stands where the list of sources is expected.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"rust": {"status": "implemented", "sources": value}})
    assert str(caught.value) == (
        f"descriptor backend 'rust' sources must be a list of repository paths, got {value!r}"
    )


@pytest.mark.parametrize(
    "path",
    ["/engine/src/neuron/lapicque.rs", "engine/../secrets.rs", "..", "", 5, None, ["a.rs"]],
)
def test_a_source_that_is_not_a_path_inside_the_repository_is_refused(path: object) -> None:
    """An absolute path, a path through a parent folder, an empty entry and a non-string are refused.

    Parameters
    ----------
    path : object
        One entry of the list of sources.
    """
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"go": {"status": "implemented", "sources": ["accel/go/lapicque.go", path]}})
    assert str(caught.value) == (
        f"descriptor backend 'go' source must be a repository-relative path, got {path!r}"
    )


def test_a_source_listed_twice_is_refused() -> None:
    """A lane names each of its files once."""
    with pytest.raises(ModelDescriptorError) as caught:
        _lanes({"mojo": {"status": "implemented", "sources": ["a.mojo", "b.mojo", "a.mojo"]}})
    assert str(caught.value) == "descriptor backend 'mojo' lists the source 'a.mojo' twice"


def test_a_name_with_dots_inside_a_folder_name_is_a_path() -> None:
    """Only a whole component of two dots leaves the repository; dots inside a name do not."""
    lanes = _lanes({"julia": {"status": "implemented", "sources": ["accel/julia/..hidden/x.jl"]}})
    assert lanes[0].sources == ("accel/julia/..hidden/x.jl",)


def test_evidence_and_sources_survive_the_generator_merge_and_the_written_file() -> None:
    """A structural refresh of a descriptor keeps both fields, through the TOML that is written."""
    curated = _payload(
        {
            "python": {"status": "implemented"},
            "rust_safety": {
                "status": "implemented",
                "parity": "exact",
                "evidence": "tests/test_lapicque_backend_parity.py",
                "sources": ["src/sc_neurocore/accel/rust/safety/lapicque.rs"],
            },
        }
    )
    regenerated = _payload({"python": {"status": "implemented"}})
    merged = merge_descriptor_payloads(curated, regenerated)
    written = tomllib.loads(tomli_w.dumps(merged))
    assert parse_model_descriptor(written).backends == parse_model_descriptor(curated).backends
    assert parse_model_descriptor(written).backends[1] == BackendSupport(
        "rust_safety",
        "implemented",
        "exact",
        "tests/test_lapicque_backend_parity.py",
        ("src/sc_neurocore/accel/rust/safety/lapicque.rs",),
    )


def test_the_shipped_descriptors_use_listed_lanes_and_the_legacy_name_does_not_spread() -> None:
    """Every shipped lane table has a listed name; the legacy name stays under its ceiling."""
    names: dict[str, int] = {}
    files = sorted(_DESCRIPTORS.glob("*.toml"))
    assert len(files) >= 185
    for path in files:
        descriptor = parse_model_descriptor(tomllib.loads(path.read_text(encoding="utf-8")))
        for lane in descriptor.backends:
            names[lane.name] = names.get(lane.name, 0) + 1
    assert set(names) <= {*BACKEND_LANES, *LEGACY_BACKEND_LANES}
    assert names.get("rust_engine", 0) <= _LEGACY_TABLE_CEILING
    assert names["python"] == len(files)
