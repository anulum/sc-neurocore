# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.studio import model_catalogue


def test_a_checkout_serves_the_page_it_generated() -> None:
    """The working tree resolves the pages it builds under `docs/api/models`."""
    page = model_catalogue.model_documentation("LapicqueNeuron")

    assert page is not None
    assert page["slug"] == "models/lapicque"
    assert page["markdown"]


def test_a_packaged_directory_is_preferred_over_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A distribution that carries its own pages serves those, not a stray tree.

    Preferring the package means packaging the pages later needs no code change.
    """
    packaged = tmp_path / "model_docs"
    packaged.mkdir()
    (packaged / "lapicque.md").write_text("# packaged page\n", encoding="utf-8")
    monkeypatch.setattr(model_catalogue, "_PACKAGED_DOCS_DIR", packaged)

    page = model_catalogue.model_documentation("LapicqueNeuron")

    assert page is not None
    assert page["markdown"] == "# packaged page\n"


def test_no_pages_installed_is_reported_as_a_distribution_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every model undocumented for one reason is not a fact about any model.

    An installed wheel packages no pages and resolves `parents[3]` to a
    directory above `site-packages`, so the checkout fallback finds nothing
    either. Returning `None` there said "this model has no documentation",
    which blamed the model for the distribution.
    """
    monkeypatch.setattr(model_catalogue, "_PACKAGED_DOCS_DIR", tmp_path / "absent")
    monkeypatch.setattr(model_catalogue, "_CHECKOUT_DOCS_DIR", tmp_path / "also-absent")

    with pytest.raises(model_catalogue.ModelDocumentationUnavailable, match="no model reference"):
        model_catalogue.model_documentation("LapicqueNeuron")


def test_a_model_without_a_page_is_still_just_that(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With pages installed, a missing one is a fact about that model alone."""
    packaged = tmp_path / "model_docs"
    packaged.mkdir()
    (packaged / "something_else.md").write_text("# other\n", encoding="utf-8")
    monkeypatch.setattr(model_catalogue, "_PACKAGED_DOCS_DIR", packaged)

    assert model_catalogue.model_documentation("LapicqueNeuron") is None


def test_an_unknown_model_is_not_a_documentation_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name the catalogue does not hold is answered before the pages are read."""
    monkeypatch.setattr(model_catalogue, "_PACKAGED_DOCS_DIR", tmp_path / "absent")
    monkeypatch.setattr(model_catalogue, "_CHECKOUT_DOCS_DIR", tmp_path / "also-absent")

    assert model_catalogue.model_documentation("NoSuchNeuron") is None


def test_documentation_root_names_which_directory_answered() -> None:
    """A caller can ask where the pages came from without reading one."""
    root = model_catalogue.documentation_root()

    assert root is not None
    assert root.is_dir()


def test_the_route_reports_a_missing_distribution_as_503_not_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over HTTP the two answers must not arrive as the same status either.

    404 tells an operator to look for another model. 503 tells them the pages
    were never installed, which is what they can act on.
    """
    from starlette.testclient import TestClient

    from sc_neurocore.studio.app import create_app

    monkeypatch.setattr(model_catalogue, "_PACKAGED_DOCS_DIR", tmp_path / "absent")
    monkeypatch.setattr(model_catalogue, "_CHECKOUT_DOCS_DIR", tmp_path / "also-absent")

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.get("/api/models/LapicqueNeuron/doc")

    assert response.status_code == 503
    assert "no model reference pages are installed" in response.text


def test_the_route_still_says_404_for_a_model_with_no_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With pages installed, the old answer is still the right one."""
    from starlette.testclient import TestClient

    from sc_neurocore.studio.app import create_app

    packaged = tmp_path / "model_docs"
    packaged.mkdir()
    (packaged / "something_else.md").write_text("# other\n", encoding="utf-8")
    monkeypatch.setattr(model_catalogue, "_PACKAGED_DOCS_DIR", packaged)

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.get("/api/models/LapicqueNeuron/doc")

    assert response.status_code == 404
