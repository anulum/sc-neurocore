# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue contracts match actual public interfaces

"""Validate discoverability, immutable metadata and real public constructor signatures."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from importlib import import_module
import inspect
import json
import subprocess
import sys

import pytest

from sc_neurocore.spike_codec import (
    CodecDescriptor,
    describe_codec,
    list_codec_descriptors,
    list_codecs,
)


def test_cold_static_discovery_does_not_import_optional_backends() -> None:
    """Observe real imports in a fresh interpreter before any catalogue access."""
    program = '''
import sys

attempts: list[str] = []

def observe(event: str, arguments: tuple[object, ...]) -> None:
    """Record native import attempts without changing import behaviour."""
    if event == "import" and str(arguments[0]).startswith("sc_neurocore_engine"):
        attempts.append(str(arguments[0]))

sys.addaudithook(observe)
from sc_neurocore.spike_codec import describe_codec, list_codec_descriptors

descriptors = list_codec_descriptors()
assert len(descriptors) == 8
assert all(describe_codec(row.name) is row for row in descriptors)
assert not attempts, attempts
assert not any(name.startswith("sc_neurocore_engine") for name in sys.modules)
assert "sc_neurocore.spike_codec.registry" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_catalogue_counts_and_legacy_factory_remain_distinct() -> None:
    """Six families and two stream interfaces must not change the five-name factory."""
    rows = list_codec_descriptors()
    assert [row.name for row in rows] == [
        "aer",
        "delta",
        "isi",
        "lossless_stream",
        "predictive",
        "spike_events",
        "streaming",
        "waveform",
    ]
    assert sum(row.category == "family" for row in rows) == 6
    assert sum(row.category == "stream_api" for row in rows) == 2
    assert list_codecs() == ["aer", "delta", "isi", "predictive", "streaming"]


def test_descriptions_are_static_immutable_and_json_serializable() -> None:
    """Lookup neither probes optional engines nor mutates constructor configuration."""
    before = set(sys.modules)
    rows = list_codec_descriptors()
    encoded = json.dumps([asdict(row) for row in rows])
    for row in rows:
        assert describe_codec(row.name) is row
        assert row.input_layout == "time_channel"
        assert row.time_unit == "sample_index"
        assert row.input_dtype
        for variant in row.variants:
            assert row.variant(variant.name) is variant
            assert set(variant.formats) <= {wire.name for wire in row.wire_formats}
    assert set(sys.modules) == before
    assert len(json.loads(encoded)) == 8
    for target, attribute in ((rows[0], "title"), (rows[0].variants[0], "name")):
        with pytest.raises(FrozenInstanceError):
            setattr(target, attribute, "changed")
    with pytest.raises(ValueError, match="Unknown codec"):
        describe_codec("Waveform")
    with pytest.raises(ValueError, match="Unknown variant"):
        describe_codec("waveform").variant("gpu")
    with pytest.raises(ValueError, match="Unknown variant"):
        replace(rows[0], variants=()).variant()


@pytest.mark.parametrize("descriptor", list_codec_descriptors(), ids=lambda row: row.name)
def test_descriptions_match_public_signatures(descriptor: CodecDescriptor) -> None:
    """Resolve actual public classes, required arguments, methods and default variants."""
    classes = []
    for path, method, required in (
        (
            descriptor.encoder_entrypoint,
            descriptor.encode_method,
            descriptor.encoder_required_parameters,
        ),
        (
            descriptor.decoder_entrypoint,
            descriptor.decode_method,
            descriptor.decoder_required_parameters,
        ),
    ):
        module, _, symbol = path.rpartition(".")
        cls = getattr(import_module(module), symbol)
        classes.append(cls)
        assert callable(getattr(cls, method))
        parameters = inspect.signature(cls).parameters
        actual_required = tuple(
            name for name, p in parameters.items() if p.default is inspect.Parameter.empty
        )
        assert actual_required == required
    parameters = inspect.signature(classes[0]).parameters
    assert tuple(parameters) == descriptor.configuration_parameters
    assert ("backend" in parameters) is descriptor.backend_parameter
    if descriptor.variant_parameter:
        assert parameters[descriptor.variant_parameter].default == descriptor.variant().name
    decode_parameters = inspect.signature(getattr(classes[1], descriptor.decode_method)).parameters
    actual_required = tuple(
        name
        for name, p in decode_parameters.items()
        if name not in ("self", "data", "packet") and p.default is inspect.Parameter.empty
    )
    assert actual_required == descriptor.decode_required_parameters


def test_native_scope_is_component_specific_and_never_counts_generated_mirrors() -> None:
    """Only real engine modules can contribute native capability."""
    for descriptor in list_codec_descriptors():
        for variant in descriptor.variants:
            if variant.native_scope == "none":
                assert variant.native_module is None and not variant.native_symbols
            else:
                assert variant.native_module is not None
                assert variant.native_module.startswith("sc_neurocore_engine")
                assert variant.native_symbols
    waveform = describe_codec("waveform")
    assert waveform.variant("lossless").native_scope == "packet_coder"
    assert waveform.variant("lossless_rice").native_scope == "channel_coder"
    assert describe_codec("predictive").variant("ema").native_scope == "predictor_kernels"
    assert describe_codec("predictive").variant("context").native_scope == "none"
    assert not describe_codec("predictive").backend_parameter
    legacy = next(wire for wire in waveform.wire_formats if wire.name == "wfcx")
    assert legacy.readable and not legacy.writable
