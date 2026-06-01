# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path

import yaml


def _workflow() -> dict:
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_text(job: dict) -> str:
    return "\n".join(
        step["run"] for step in job["steps"] if isinstance(step, dict) and "run" in step
    )


def test_python_and_engine_pypi_publish_use_oidc_trusted_publishing() -> None:
    jobs = _workflow()["jobs"]

    python_publish = jobs["publish-python-pypi"]
    assert python_publish["environment"] == "pypi"
    assert python_publish["permissions"] == {"id-token": "write"}

    engine_publish = jobs["publish-engine-pypi"]
    assert engine_publish["environment"] == "pypi"
    assert engine_publish["permissions"] == {"id-token": "write"}

    engine_publish_text = str(engine_publish)
    assert "PYPI_ENGINE_TOKEN" in engine_publish_text
    assert "password" in engine_publish_text
    assert "attestations: false" not in engine_publish_text


def test_publish_workflow_builds_and_smoke_tests_engine_wheels_before_upload() -> None:
    jobs = _workflow()["jobs"]

    build_text = _run_text(jobs["build-engine-wheels"])
    assert "maturin build --release --out ../dist-engine/" in build_text

    smoke = jobs["smoke-engine-wheels"]
    assert smoke["needs"] == ["build-engine-wheels"]
    smoke_text = _run_text(smoke)
    assert "pip install dist-engine/*.whl" in smoke_text
    assert "ZipFile(wheels[0])" not in smoke_text
    assert "PYTHONPATH=dist-engine/smoke" not in smoke_text
    assert "pip install --upgrade pip" not in smoke_text
    assert "import sc_neurocore_engine" in smoke_text
    assert "simd_tier()" in smoke_text

    assert jobs["publish-engine-pypi"]["needs"] == ["smoke-engine-wheels"]
