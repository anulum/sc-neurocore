# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training modular-architecture contracts

"""Lock the Studio training facade, dependency graph, and HTTP wiring."""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path
from typing import Any

from sc_neurocore.studio import training
from sc_neurocore.studio.app import create_app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATHS = {
    "training": _REPO_ROOT / "src/sc_neurocore/studio/training.py",
    "_training_attach": _REPO_ROOT / "src/sc_neurocore/studio/_training_attach.py",
    "_training_control": _REPO_ROOT / "src/sc_neurocore/studio/_training_control.py",
    "_training_events": _REPO_ROOT / "src/sc_neurocore/studio/_training_events.py",
    "_training_job": _REPO_ROOT / "src/sc_neurocore/studio/_training_job.py",
}
_MODULE_LINE_CEILINGS = {
    "training": 375,
    "_training_attach": 225,
    "_training_control": 350,
    "_training_events": 175,
    "_training_job": 675,
}
_EXPECTED_DEPENDENCIES = {
    "training": {
        "_training_attach",
        "_training_control",
        "_training_events",
        "_training_job",
    },
    "_training_attach": {"_training_control", "_training_job"},
    "_training_control": {"_training_events", "_training_job"},
    "_training_events": set(),
    "_training_job": {"_training_events"},
}
_EXPECTED_EXPORTS = {
    "HAS_TORCH",
    "TRAINING_EVENT_LOG_ARTIFACT_PATH",
    "TrainingJob",
    "export_training_checkpoint",
    "get_training_status",
    "import_training_checkpoint",
    "list_cell_types",
    "list_jobs",
    "list_surrogates",
    "request_live_training_weight_attach",
    "start_training",
    "start_training_attach",
    "stop_training",
    "stream_metrics",
}
_EXPECTED_SIGNATURES = {
    "TrainingJob": "(config: 'dict[str, Any]', *, job_id: 'str | None' = None, "
    "cancelled: 'Callable[[], bool] | None' = None, "
    "event_sink: 'Callable[[dict[str, object]], None] | None' = None, "
    "initial_state_dict: 'Mapping[str, object] | None' = None) -> 'None'",
    "export_training_checkpoint": "(job_id: 'str', job_manager: 'StudioJobManager | None' = None) -> 'dict[str, Any]'",
    "get_training_status": "(job_id: 'str', job_manager: 'StudioJobManager | None' = None) -> 'dict[str, Any]'",
    "import_training_checkpoint": "(data: 'dict[str, Any]') -> 'dict[str, Any]'",
    "list_cell_types": "() -> 'list[dict[str, Any]]'",
    "list_jobs": "() -> 'list[dict[str, Any]]'",
    "list_surrogates": "() -> 'list[dict[str, Any]]'",
    "request_live_training_weight_attach": "(target_job_id: 'str', source_job_id: 'str', "
    "job_manager: 'StudioJobManager', *, expected_config_sha256: 'str | None' = None) -> 'dict[str, Any]'",
    "start_training": "(config: 'dict[str, Any]', job_manager: 'StudioJobManager | None' = None) -> 'dict[str, Any]'",
    "start_training_attach": "(source_job_id: 'str', config: 'dict[str, Any]', "
    "job_manager: 'StudioJobManager', *, expected_config_sha256: 'str | None' = None) -> 'dict[str, Any]'",
    "stop_training": "(job_id: 'str', job_manager: 'StudioJobManager | None' = None) -> 'dict[str, Any]'",
    "stream_metrics": "(job_id: 'str', job_manager: 'StudioJobManager | None' = None) -> 'Any'",
}
_EXPECTED_HTTP_ROUTES = {
    ("POST", "/api/studio/training/weight-restore"),
    ("POST", "/api/studio/training/weight-restore/attach"),
    ("POST", "/api/studio/training/weight-restore/attach/live"),
    ("GET", "/api/training/surrogates"),
    ("GET", "/api/training/cell-types"),
    ("POST", "/api/training/start"),
    ("POST", "/api/training/stop"),
    ("GET", "/api/training/jobs"),
    ("GET", "/api/training/status/{job_id}"),
    ("GET", "/api/training/checkpoint/{job_id}"),
    ("POST", "/api/training/checkpoint/import"),
    ("GET", "/api/training/stream/{job_id}"),
}


def _training_dependencies(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            prefix = "sc_neurocore.studio."
            if node.module.startswith(prefix):
                candidate = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
                if candidate in _MODULE_PATHS:
                    dependencies.add(candidate)
    return dependencies


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(module: str) -> None:
        if module in active:
            raise AssertionError(f"training import cycle reaches {module}")
        if module in visited:
            return
        active.add(module)
        for dependency in graph[module]:
            visit(dependency)
        active.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def test_training_modules_have_bounded_single_direction_dependencies() -> None:
    """Training responsibilities remain bounded and form the intended DAG."""
    graph = {name: _training_dependencies(path) for name, path in _MODULE_PATHS.items()}

    assert graph == _EXPECTED_DEPENDENCIES
    _assert_acyclic(graph)
    for name, path in _MODULE_PATHS.items():
        assert len(path.read_text(encoding="utf-8").splitlines()) <= _MODULE_LINE_CEILINGS[name]


def test_training_facade_preserves_exports_signatures_and_pickle_identity() -> None:
    """The historical facade retains its exact callable and class contract."""
    assert set(training.__all__) == _EXPECTED_EXPORTS
    for name, expected_signature in _EXPECTED_SIGNATURES.items():
        value: Any = getattr(training, name)
        assert str(inspect.signature(value)) == expected_signature
        assert value.__module__ == "sc_neurocore.studio.training"

    assert inspect.isgeneratorfunction(training.stream_metrics)
    assert pickle.loads(pickle.dumps(training.TrainingJob)) is training.TrainingJob


def test_training_http_routes_remain_wired_into_the_composed_application() -> None:
    """The composed Studio application exposes every established training route."""
    openapi = create_app().openapi()
    actual = {
        (method.upper(), path)
        for path, operations in openapi["paths"].items()
        if "training" in path
        for method in operations
    }

    assert actual == _EXPECTED_HTTP_ROUTES
