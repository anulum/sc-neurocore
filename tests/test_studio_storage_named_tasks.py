# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — named storage task tests

"""Check reviewed task selection against the actual Studio route policies."""

import ast
import json
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.policy_models import RouteVisibility
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.platform.storage_named_tasks import (
    named_studio_task_for_path,
    resolve_named_studio_task,
)
from tests.studio_seccomp_support import run_child


_NAMED_ROUTES = {
    "laboratory.run": "/api/fits/jobs",
    "analysis.run": "/api/analysis/jobs",
    "model.scan": "/api/models/scan/jobs",
    "audit.quarantine_archive": "/api/studio/audit/quarantine/archive",
    "audit.quarantine_restore": "/api/studio/audit/quarantine/archive/restore",
    "evidence.bundle": "/api/studio/evidence/bundle",
    "training.weight_restore": "/api/studio/training/weight-restore",
    "compiler.compile": "/api/compile",
    "compiler.model_compile": "/api/models/compile",
    "compiler.model_cosim": "/api/models/cosim",
    "compiler.pipeline": "/api/pipeline/run",
    "synthesis.run": "/api/synth/run",
    "synthesis.multi_target": "/api/synth/multi-target",
    "synthesis.terminal": "/api/synth/terminal",
    "synthesis.pnr": "/api/synth/pnr",
    "training.start": "/api/training/start",
    "training.attach": "/api/studio/training/weight-restore/attach",
}


def test_named_process_tasks_require_actual_nonpublic_post_policies() -> None:
    """All observed process submissions have an exact protected policy route."""
    policies = build_default_studio_route_policy_registry()
    source_root = Path(__file__).resolve().parents[1] / "src"
    assert len(_NAMED_ROUTES) == 17
    for name, route in _NAMED_ROUTES.items():
        task = resolve_named_studio_task(name, authorized_route=route)
        assert task.name == name
        assert task.kind and task.owner
        assert task.task_path.startswith("sc_neurocore.studio.")
        assert policies.policy_for("POST", route).visibility != RouteVisibility.PUBLIC
        module, function = task.task_path.split(":", 1)
        source = source_root.joinpath(*module.split(".")).with_suffix(".py")
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        assert any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function
            for node in tree.body
        )


def test_training_attach_routes_share_only_the_reviewed_task() -> None:
    """Both weight-restore attach routes resolve to one preserved task owner."""
    base = "/api/studio/training/weight-restore/attach"
    ordinary = resolve_named_studio_task("training.attach", authorized_route=base)
    live = resolve_named_studio_task("training.attach", authorized_route=base + "/live")
    assert ordinary == live
    assert ordinary.kind == "training"
    assert ordinary.owner == "studio-training-attach"


@pytest.mark.parametrize(
    "name,route",
    [
        ("unknown", "/api/analysis/jobs"),
        (
            "sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
            "/api/analysis/jobs",
        ),
        ("analysis.run", "/api/models/scan/jobs"),
        ("training.attach", "/api/training/start"),
        ("analysis.run", "POST /api/analysis/jobs"),
    ],
)
def test_unknown_names_and_route_confusion_refuse(name: str, route: str) -> None:
    """A task name cannot select another endpoint or a raw import path."""
    with pytest.raises(ValueError, match="not available"):
        resolve_named_studio_task(name, authorized_route=route)


@pytest.mark.parametrize("field", ["name", "route"])
def test_nontext_task_selection_refuses(field: str) -> None:
    """Actual untyped callers cannot resolve a nontext name or route."""
    arguments = [None, "/api/analysis/jobs"] if field == "name" else ["analysis.run", None]
    result = run_child(
        "import json, sys\n"
        "from sc_neurocore.refusals import AuthoredRefusal\n"
        "from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task\n"
        "name, route = json.loads(sys.argv[1])\n"
        "try:\n"
        "    resolve_named_studio_task(name, authorized_route=route)\n"
        "except ValueError as error:\n"
        "    print(json.dumps({'authored': isinstance(error, AuthoredRefusal),\n"
        "        'message': str(error)}))\n",
        arguments=(json.dumps(arguments),),
    )
    assert result == {"authored": True, "message": "invalid named Studio task selection"}


@pytest.mark.parametrize("principal", [None, ""])
def test_laboratory_owner_requires_actual_requester_identity(principal: str | None) -> None:
    """Requester-owned tasks refuse missing identity and preserve a real actor."""
    task = resolve_named_studio_task("laboratory.run", authorized_route="/api/fits/jobs")
    with pytest.raises(ValueError) as caught:
        task.owner_for(principal)
    assert isinstance(caught.value, AuthoredRefusal)
    assert str(caught.value) == "requester-owned task requires authenticated identity"
    assert task.owner_for("laboratory-operator") == "laboratory-operator"


@pytest.mark.parametrize(
    ("route", "operation"),
    [
        ("/api/fits/jobs", "fit"),
        ("/api/fits/replay/jobs", "fit_replay"),
        ("/api/cohorts/jobs", "cohort"),
    ],
)
def test_laboratory_selection_binds_payload_and_admission(route: str, operation: str) -> None:
    """Reviewed operations require matching route, payload and admission metadata."""
    task = resolve_named_studio_task("laboratory.run", authorized_route=route)
    task.validate_admission(
        authorized_route=route,
        payload={"operation": operation},
        admission={"laboratory_task": operation},
    )
    for payload, admission in [
        ({"operation": "foreign"}, {"laboratory_task": operation}),
        ({"operation": operation}, None),
        ({"operation": operation}, {"laboratory_task": "foreign"}),
    ]:
        with pytest.raises(ValueError) as caught:
            task.validate_admission(authorized_route=route, payload=payload, admission=admission)
        assert isinstance(caught.value, AuthoredRefusal)
        assert (
            str(caught.value)
            == "laboratory operation and admission must match the authorized route"
        )


def test_import_path_selection_refuses_a_different_reviewed_route() -> None:
    """An exact real task path cannot select an endpoint outside its registered routes."""
    task = resolve_named_studio_task("model.scan", authorized_route="/api/models/scan/jobs")
    assert named_studio_task_for_path(task.task_path, authorized_route=task.routes[0]) == task
    with pytest.raises(ValueError) as caught:
        named_studio_task_for_path(task.task_path, authorized_route="/api/analysis/jobs")
    assert isinstance(caught.value, AuthoredRefusal)
    assert str(caught.value) == "named Studio task is not available on this route"
