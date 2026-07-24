# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (hierarchy) from former test_scnir_schema.py

from __future__ import annotations

from tests.scnir_schema_support import *  # noqa: F403


def test_scnir_records_hierarchy_instance_port_metadata() -> None:
    base = _valid_document()
    doc = SCNIRDocument(
        producer=base.producer,
        streams=base.streams,
        hierarchy=(
            SCNIRHierarchyInstance(
                instance_id="top.subgraph0",
                module_name="scnir_subgraph0",
                ports=(
                    SCNIRHierarchyPort(
                        port_name="spike_in",
                        direction="input",
                        stream_id="layer0_input",
                        signal_kind="spike",
                        bit_width=16,
                    ),
                    SCNIRHierarchyPort(
                        port_name="weight_out",
                        direction="output",
                        stream_id="layer0_weight",
                        signal_kind="weight",
                        bit_width=12,
                    ),
                ),
            ),
        ),
    )

    payload = scnir_to_dict(doc)

    assert payload["schema_version"] == SCNIR_SCHEMA_VERSION
    assert payload["hierarchy"] == [
        {
            "instance_id": "top.subgraph0",
            "module_name": "scnir_subgraph0",
            "ports": [
                {
                    "port_name": "spike_in",
                    "direction": "input",
                    "stream_id": "layer0_input",
                    "signal_kind": "spike",
                    "bit_width": 16,
                },
                {
                    "port_name": "weight_out",
                    "direction": "output",
                    "stream_id": "layer0_weight",
                    "signal_kind": "weight",
                    "bit_width": 12,
                },
            ],
        }
    ]
    assert scnir_to_dict(scnir_from_dict(payload)) == payload
    validate_scnir_dict(payload)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["hierarchy"][0]["ports"].append(
                dict(payload["hierarchy"][0]["ports"][0])
            ),
            "duplicate",
        ),
        (
            lambda payload: payload["hierarchy"].append(dict(payload["hierarchy"][0])),
            "duplicate",
        ),
        (
            lambda payload: payload["hierarchy"][0]["ports"][0].update(
                {"stream_id": "missing.stream"}
            ),
            "stream_id",
        ),
        (
            lambda payload: payload["hierarchy"][0]["ports"][0].update({"signal_kind": "weight"}),
            "signal_kind",
        ),
        (
            lambda payload: payload["hierarchy"][0].update({"module_name": "bad-module"}),
            "module_name",
        ),
        (
            lambda payload: payload["hierarchy"][0]["ports"][0].update({"bit_width": 0}),
            "bit_width",
        ),
        (
            lambda payload: payload["hierarchy"][0].update({"ports": []}),
            "ports",
        ),
    ],
)
def test_scnir_rejects_invalid_hierarchy_metadata(mutator: object, message: str) -> None:
    base = _valid_document()
    payload = scnir_to_dict(
        SCNIRDocument(
            producer=base.producer,
            streams=base.streams,
            hierarchy=(
                SCNIRHierarchyInstance(
                    instance_id="top.subgraph0",
                    module_name="scnir_subgraph0",
                    ports=(
                        SCNIRHierarchyPort(
                            port_name="spike_in",
                            direction="input",
                            stream_id="layer0_input",
                            signal_kind="spike",
                            bit_width=16,
                        ),
                    ),
                ),
            ),
        )
    )
    mutator(payload)

    with pytest.raises(SCNIRValidationError, match=message):
        validate_scnir_dict(payload)
