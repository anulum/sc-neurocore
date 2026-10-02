# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real catalogue query refusal contracts

"""Exercise malformed and valid readiness filters through the actual Studio API."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.catalogue_query import CatalogueQuery, CatalogueQueryRejected
from sc_neurocore.studio.platform import StudioRuntimeSettings


@pytest.fixture
def query_client(tmp_path: Path) -> Iterator[TestClient]:
    """Mount the real application with a job root owned by this test."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path)))
    with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize("field", ["min_verified_science", "min_verified_silicon"])
@pytest.mark.parametrize(
    "raw",
    ["", "-1", "+1", "1.0", " 1", "6", "²", "①", "1" * 5000],
    ids=[
        "empty",
        "negative",
        "signed",
        "fraction",
        "space",
        "out-of-range",
        "superscript",
        "circled",
        "long-decimal",
    ],
)
def test_invalid_readiness_is_an_authored_http_refusal(
    query_client: TestClient, field: str, raw: str
) -> None:
    """Invalid numbers produce the declared reason rather than an HTTP server error."""
    response = query_client.get("/api/models/query", params={field: raw})
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": {"reason": f"{field} must be an integer from 0 to 5"}}


@pytest.mark.parametrize("field", ["min_verified_science", "min_verified_silicon"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", "0"), ("5", "5"), ("0003", "3"), ("٣", "3"), ("００５", "5"), ("٠" * 5000 + "٣", "3")],
    ids=["zero", "ceiling", "zero-padded", "arabic", "full-width", "long-zero-padded"],
)
def test_decimal_readiness_preserves_the_actual_catalogue_result(
    query_client: TestClient, field: str, raw: str, expected: str
) -> None:
    """Unicode and zero-padded tiers select the same real corpus as ordinary decimal tiers."""
    reference = query_client.get("/api/models/query", params={field: expected})
    response = query_client.get("/api/models/query", params={field: raw})
    assert reference.status_code == response.status_code == 200, response.text
    assert response.json() == reference.json()


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"unknown": "3"}, "unknown catalogue query parameter: unknown"),
        ({"verified_perfect_only": "yes"}, "verified_perfect_only must be true or false"),
        ({"min_verified_science": "6"}, "min_verified_science must be an integer from 0 to 5"),
        ({"min_verified_silicon": "²"}, "min_verified_silicon must be an integer from 0 to 5"),
    ],
)
def test_public_query_parser_raises_only_its_authored_domain_refusal(
    params: dict[str, str], reason: str
) -> None:
    """The public parser keeps deliberate messages compatible with existing ValueError callers."""
    with pytest.raises(CatalogueQueryRejected) as caught:
        CatalogueQuery.from_params(params)
    assert isinstance(caught.value, AuthoredRefusal)
    assert str(caught.value) == reason


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"unknown": "3"}, "unknown catalogue query parameter: unknown"),
        ({"verified_perfect_only": "yes"}, "verified_perfect_only must be true or false"),
    ],
)
def test_other_query_refusals_preserve_their_http_contract(
    query_client: TestClient, params: dict[str, str], reason: str
) -> None:
    """Unknown filters and malformed booleans keep their existing route-level explanation."""
    response = query_client.get("/api/models/query", params=params)
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": {"reason": reason}}
