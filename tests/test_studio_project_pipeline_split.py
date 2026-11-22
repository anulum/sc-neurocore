# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio project storage and pipeline split contract

"""The pipeline left the project storage module; every historical import still reaches it.

``sc_neurocore.studio.project`` held two responsibilities: project storage and the
graph-to-target pipeline. The pipeline now lives in
``sc_neurocore.studio.project_pipeline``. The storage module re-exports the
pipeline's public names as the same objects, so code and pickles that name
``sc_neurocore.studio.project.run_pipeline`` keep working, and the process task
imports the pipeline from its own module.
"""

from __future__ import annotations

import pickle

import sc_neurocore.studio.project as project
import sc_neurocore.studio.project_pipeline as project_pipeline
from sc_neurocore.studio.platform import pipeline_process

PUBLIC_PIPELINE_NAMES = (
    "PIPELINE_COSIM_STEP_LIMIT",
    "PIPELINE_Q_FORMATS",
    "PIPELINE_ROUTE",
    "run_pipeline",
)


def test_historical_project_imports_are_the_pipeline_module_objects() -> None:
    """Each public pipeline name of the storage module is the pipeline module's own object."""
    for name in PUBLIC_PIPELINE_NAMES:
        assert getattr(project, name) is getattr(project_pipeline, name), name


def test_the_pipeline_is_defined_in_its_own_module() -> None:
    """``run_pipeline`` is owned by the pipeline module, by qualified name and by pickle."""
    assert project_pipeline.run_pipeline.__module__ == "sc_neurocore.studio.project_pipeline"
    assert project_pipeline.run_pipeline.__qualname__ == "run_pipeline"
    assert pickle.loads(pickle.dumps(project.run_pipeline)) is project_pipeline.run_pipeline


def test_the_process_task_runs_the_pipeline_module_function() -> None:
    """The named process task imports the pipeline and its formats from the pipeline module."""
    names = vars(pipeline_process)
    assert names["run_pipeline"] is project_pipeline.run_pipeline
    assert names["PIPELINE_Q_FORMATS"] is project_pipeline.PIPELINE_Q_FORMATS


def test_the_storage_module_no_longer_carries_pipeline_machinery() -> None:
    """Project storage imports neither the synthesis process limits nor the pipeline's tools."""
    for name in ("EdaProcessLimits", "hashlib", "tempfile"):
        assert not hasattr(project, name), name
