# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio jobs architecture contracts

"""Guard the focused Studio jobs module graph and historical public facade."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

from coverage import Coverage

import sc_neurocore.studio.platform.jobs as jobs
from tests.studio_jobs_support import (
    EXPECTED_JOBS_EXPORTS,
    JOBS_FACADE,
    JOBS_IMPLEMENTATION_MODULES,
    JOBS_SOURCE_PATHS,
    JOBS_TEST_PATHS,
    assert_acyclic,
    implementation_import_graph,
)

PUBLIC_MODEL_NAMES = (
    "StudioJobRejected",
    "StudioJobCancelled",
    "StudioJobArtifactUnavailable",
    "StudioJobArtifact",
    "StudioJobRecord",
    "StudioJobResourceProfile",
    "StudioJobStatusSnapshot",
    "StudioJobListSnapshot",
    "StudioJobArtifactPayload",
)


def test_studio_jobs_facade_preserves_exact_public_contract() -> None:
    """Keep exported names, object identity, and historical modules stable."""

    assert tuple(jobs.__all__) == EXPECTED_JOBS_EXPORTS
    models = importlib.import_module(f"{jobs.__package__}.jobs_models")
    for name in PUBLIC_MODEL_NAMES:
        facade_object = getattr(jobs, name)
        assert facade_object is getattr(models, name)
        assert facade_object.__module__ == JOBS_FACADE
    context = importlib.import_module(f"{jobs.__package__}.jobs_context")
    manager = importlib.import_module(f"{jobs.__package__}.jobs_manager")
    assert jobs.StudioJobContext is context.StudioJobContext
    assert jobs.StudioJobManager is manager.StudioJobManager
    assert jobs.StudioJobContext.__module__ == JOBS_FACADE
    assert jobs.StudioJobManager.__module__ == JOBS_FACADE


def test_studio_jobs_public_records_pickle_through_historical_facade() -> None:
    """Preserve serialized references created before implementation extraction."""

    artifact = jobs.StudioJobArtifact(relative_path="result.bin", size_bytes=2, sha256="ab")
    record = jobs.StudioJobRecord(
        job_id="sj_0000000000000000",
        kind="synthesis",
        owner="operator",
        request_id="request",
        status="completed",
        execution_model="thread",
        created_at_utc="2026-07-13T00:00:00Z",
        artifacts=(artifact,),
    )
    assert pickle.loads(pickle.dumps(artifact)) == artifact
    assert pickle.loads(pickle.dumps(record)) == record
    rejection = pickle.loads(pickle.dumps(jobs.StudioJobRejected("rejected")))
    assert type(rejection) is jobs.StudioJobRejected
    assert str(rejection) == "rejected"


def test_studio_jobs_implementation_graph_is_acyclic_and_facade_independent() -> None:
    """Keep implementation imports one-way and independent of the facade."""

    graph = implementation_import_graph()
    assert set(graph) == set(JOBS_IMPLEMENTATION_MODULES)
    assert all(JOBS_FACADE not in dependencies for dependencies in graph.values())
    assert_acyclic(graph)


def test_studio_jobs_files_remain_below_godfile_threshold() -> None:
    """Prevent source and focused tests from regrowing beyond 300 lines."""

    for path in (*JOBS_SOURCE_PATHS, *JOBS_TEST_PATHS):
        assert path.is_file(), path
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= 300, f"{path.name}: {line_count} lines"


def test_jobs_coverage_gate_matches_the_measured_contract() -> None:
    """Keep zero-hit discovery, branch and subprocess measurement, and the full gate.

    Spawned worker and guard interpreters are traced so real process tests
    count; the gate stays at 100% and every uncovered line stays visible
    until a real test reaches it.
    """
    root = Path(__file__).resolve().parents[1]
    measurement = Coverage(config_file=str(root / "tests/studio_jobs.coveragerc"))
    assert measurement.get_option("run:source_dirs") == ["src/sc_neurocore/studio/platform"]
    assert measurement.get_option("run:branch") is True
    assert measurement.get_option("run:patch") == ["subprocess"]
    assert measurement.get_option("run:parallel") is True
    assert measurement.get_option("run:sigterm") is True
    assert measurement.get_option("report:include") == ["*/sc_neurocore/studio/platform/jobs*.py"]
    assert measurement.get_option("report:fail_under") == 100
    assert measurement.get_option("report:show_missing") is True
