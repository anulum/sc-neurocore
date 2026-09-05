# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF primary-source and paired-schema contracts

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from sc_neurocore.neurons.models.expif import ExpIFNeuron


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_DIRECTORY = REPOSITORY / "src/sc_neurocore/neurons/model_schemas"


def test_paired_schemas_separate_source_and_sc_profiles_exactly() -> None:
    with (SCHEMA_DIRECTORY / "exp_if.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((SCHEMA_DIRECTORY / "exp_if.json").read_text(encoding="utf-8"))

    assert toml_schema == json_schema
    assert toml_schema["extensions"]["profile"] == "sc_rk4"
    source = toml_schema["profiles"]["fourcaud_trocme_2003"]
    assert source == {
        "v_threshold": -30.0,
        "dt": 0.01,
        "dt_source_constraint": (
            "less than 0.02 ms; 0.01 ms is a maintained converged specialization"
        ),
        "method": (
            "deterministic-zero-noise-rk2-below-boundary-with-derived-analytical-tail-duration"
        ),
        "refractory_period": 1.7,
        "analytical_tail_ms": 0.001855930799631619,
        "observation_threshold": 20.0,
    }


def test_source_factory_matches_the_paired_schema_profile() -> None:
    source = ExpIFNeuron.fourcaud_trocme_2003()
    with (SCHEMA_DIRECTORY / "exp_if.toml").open("rb") as handle:
        profile = tomllib.load(handle)["profiles"]["fourcaud_trocme_2003"]

    assert source.v_threshold == profile["v_threshold"]
    assert source.dt == profile["dt"]
    assert source.refractory_period == profile["refractory_period"]
    assert source.analytical_tail_ms() == profile["analytical_tail_ms"]
