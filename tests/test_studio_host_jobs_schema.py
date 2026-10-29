# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — hosted scientific request admission tests

"""Exercise the public closed request parser with exact scientific inputs."""

from __future__ import annotations

import json

import pytest

from sc_neurocore.studio.api.host_jobs_schema import (
    HOST_REQUEST_MAX_BYTES,
    HostAnalysisRequest,
    parse_host_job,
)


def test_analysis_retains_parameter_order_before_worker_admission() -> None:
    """Source-order coefficients survive validation without sorted-map rewriting."""
    source = '{"operation":"sensitivity","request":{"equations":["dv/dt=I-z*v+a"],"params":{"z":1,"a":2},"init":{"v":0},"duration":1,"dt":0.1}}'
    request = parse_host_job(source)
    assert isinstance(request, HostAnalysisRequest)
    assert isinstance(request.request["params"], dict)
    assert list(request.request["params"]) == ["z", "a"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "null",
        "{}",
        "\ufeff{}",
        '{"a":NaN}',
        '{"a":1e1000}',
        '{"operation":"simulate","operation":"simulate","request":{}}',
        '{"operation":"simulate","request":{"model_name":"LIFNeuron","params":{"a":1,"a":2}}}',
        '{"operation":"simulate","request":{},"owner":"caller"}',
        '{"operation":"training","request":{}}',
        '{"operation":"simulate","request":{}}',
        " " * (HOST_REQUEST_MAX_BYTES + 1),
    ],
)
def test_invalid_or_ambiguous_request_never_becomes_a_job(text: str) -> None:
    """Malformed, ambiguous, unsupported and oversized requests refuse normally."""
    with pytest.raises(ValueError):
        parse_host_job(text)


def test_compiler_preserves_exact_large_key_and_ordered_source() -> None:
    """No JavaScript-sized integer or sorted source map replaces the native input."""
    request = parse_host_job(
        json.dumps(
            {
                "operation": "ir.build",
                "request": {
                    "schema_version": "studio.equation-ir-request.v1",
                    "model": {
                        "equations": [{"name": "v", "expression": "I-v"}],
                        "rng_seed": "18446744073709551615",
                        "random_stream": 4294967295,
                    },
                },
            }
        )
    )
    assert request.operation == "ir.build"
    assert request.request.model.rng_seed == "18446744073709551615"


def test_excessive_json_nesting_returns_a_controlled_parser_refusal() -> None:
    """A bounded but overnested document cannot escape as an interpreter recursion error."""
    with pytest.raises(ValueError):
        parse_host_job("[" * 10_000 + "0" + "]" * 10_000)
