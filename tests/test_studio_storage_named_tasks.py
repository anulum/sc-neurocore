# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — named storage task tests

"""Check reviewed task selection against the actual Studio route policies."""

import ast
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.policy_models import RouteVisibility
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task


_NAMED_ROUTES = {
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
    assert len(_NAMED_ROUTES) == 16
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


def test_nontext_task_selection_refuses() -> None:
    """Non-string wire values cannot reach catalogue lookup."""
    with pytest.raises(ValueError, match="invalid named"):
        resolve_named_studio_task(cast(str, None), authorized_route="/api/analysis/jobs")
