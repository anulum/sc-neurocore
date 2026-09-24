# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training exact resume

"""Continuing a run must reach the same place as never stopping it.

The acceptance case is a real one: train two epochs; separately train one,
stop, and resume for the second. If the resume is exact the two agree on every
metric, and if it is only a warm start they do not — because a warm start
throws away the optimiser's moment estimates and restarts the shuffle order.
These cases hold both halves of that, and hold the refusals that stop a saved
position being restored into a run it does not belong to.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio._training_job import TrainingJob
from sc_neurocore.studio.platform.training_process import run_training_process_task
from sc_neurocore.studio.platform.training_weight_loader import load_training_resume_block
from sc_neurocore.studio.training import TRAINING_EVENT_LOG_ARTIFACT_PATH
from sc_neurocore.studio.platform.training_weights import TRAINING_WEIGHT_ARTIFACT_PATH
from sc_neurocore.studio.training_resume import (
    TRAINING_RESUME_SCHEMA_VERSION,
    apply_resume_state,
    TrainingResumeMismatch,
    TrainingResumeState,
    dataset_fingerprint,
    resume_state_from_payload,
)
from tests.test_studio_training_job import _context

pytest.importorskip("torch")

#: Small enough to run twice in a test, long enough for the optimiser to hold
#: state that a warm start would discard.
_CONFIG: dict[str, Any] = {
    "dataset": "synthetic",
    "epochs": 2,
    "batch_size": 64,
    "hidden": [8],
    "timesteps": 2,
    "seed": 11,
}


def _saved_checkpoint(tmp_path: Path, job_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run a job to completion and return its loaded weight checkpoint."""
    import torch

    context = _context(tmp_path, job_id)
    result = run_training_process_task(context, dict(config))
    blob = (tmp_path / job_id / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes()
    loaded = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    return {"result": result, "checkpoint": loaded, "blob": blob}


def _resumed(
    tmp_path: Path,
    job_id: str,
    config: dict[str, Any],
    saved: dict[str, Any],
    resume: TrainingResumeState | None,
) -> dict[str, Any]:
    """Continue (or warm-start) from a saved checkpoint and run to completion."""
    context = _context(tmp_path, job_id)
    job = TrainingJob(
        dict(config),
        job_id=context.job_id,
        cancelled=lambda: context.cancelled,
        event_sink=lambda event: context.append_artifact_event(
            TRAINING_EVENT_LOG_ARTIFACT_PATH, event
        ),
        initial_state_dict=saved["checkpoint"]["model_state_dict"],
        resume_state=resume,
    )
    return job.run_blocking(context)


class TestExactResume:
    def test_resuming_reaches_the_same_place_as_never_stopping(self, tmp_path: Path) -> None:
        """The acceptance case, and the reason this contract exists."""
        uninterrupted = _saved_checkpoint(tmp_path, "sj_whole", _CONFIG)
        first = _saved_checkpoint(tmp_path, "sj_first", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(first["checkpoint"]["resume_state"])

        resumed = _resumed(tmp_path, "sj_resumed", _CONFIG, first, resume)

        assert resume.epochs_completed == 1
        assert resumed["final_metrics"] == uninterrupted["result"]["final_metrics"]

    def test_a_concurrent_in_process_run_does_not_disturb_an_exact_resume(
        self, tmp_path: Path
    ) -> None:
        """The generators are process-global, so a second run must wait its turn.

        A legacy in-process run left training in this process drew from the same
        generators while the runs below trained; the resumed run then no longer
        matched the uninterrupted one. The background run is mid-training (its
        first epoch has been reported) when the runs below start.
        """
        background = TrainingJob({**_CONFIG, "epochs": 50, "seed": 3})
        background.start()
        while background.metrics.get(timeout=120)["event"] != "epoch":
            pass
        try:
            uninterrupted = _saved_checkpoint(tmp_path, "sj_whole_busy", _CONFIG)
            first = _saved_checkpoint(tmp_path, "sj_first_busy", {**_CONFIG, "epochs": 1})
            resume = resume_state_from_payload(first["checkpoint"]["resume_state"])
            resumed = _resumed(tmp_path, "sj_resumed_busy", _CONFIG, first, resume)
        finally:
            assert background._thread is not None
            background._thread.join(timeout=300)

        assert background.status == "completed"
        assert resumed["final_metrics"] == uninterrupted["result"]["final_metrics"]

    def test_a_warm_start_is_a_different_run_and_does_not_pretend_otherwise(
        self, tmp_path: Path
    ) -> None:
        """Weights alone restart the optimiser and the shuffle order."""
        uninterrupted = _saved_checkpoint(tmp_path, "sj_whole_warm", _CONFIG)
        first = _saved_checkpoint(tmp_path, "sj_first_warm", {**_CONFIG, "epochs": 1})

        warm = _resumed(tmp_path, "sj_warm", {**_CONFIG, "epochs": 1}, first, None)

        assert warm["final_metrics"] != uninterrupted["result"]["final_metrics"]

    def test_a_resume_starts_after_the_epochs_already_done(self, tmp_path: Path) -> None:
        """Neither repeating nor skipping the interrupted run's work."""
        first = _saved_checkpoint(tmp_path, "sj_first_epochs", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(first["checkpoint"]["resume_state"])

        context = _context(tmp_path, "sj_epoch_events")
        events: list[dict[str, object]] = []

        def sink(event: dict[str, object]) -> None:
            events.append(event)
            context.append_artifact_event(TRAINING_EVENT_LOG_ARTIFACT_PATH, event)

        job = TrainingJob(
            dict(_CONFIG),
            job_id=context.job_id,
            cancelled=lambda: context.cancelled,
            event_sink=sink,
            initial_state_dict=first["checkpoint"]["model_state_dict"],
            resume_state=resume,
        )
        job.run_blocking(context)

        config_event = next(event for event in events if event["event"] == "config")
        assert config_event["data"]["start_epoch"] == 1  # type: ignore[index]
        epochs = [
            event["data"]["epoch"]  # type: ignore[index]
            for event in events
            if event["event"] == "epoch"
        ]
        assert epochs == [1]


class TestResumeRefusals:
    def test_a_resume_into_a_different_network_is_refused(self, tmp_path: Path) -> None:
        """The saved optimiser belongs to a network this run does not have."""
        import torch

        first = _saved_checkpoint(tmp_path, "sj_arch_source", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(first["checkpoint"]["resume_state"])
        other = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)

        with pytest.raises(TrainingResumeMismatch) as raised:
            apply_resume_state(
                resume,
                optimiser=other,
                architecture="64->16->10",
                config=resume.config,
            )

        assert raised.value.field == "architecture"
        assert raised.value.to_public_detail()["error"] == "training_resume_mismatch"

    def test_a_run_whose_weights_do_not_fit_is_refused_before_the_resume(
        self, tmp_path: Path
    ) -> None:
        """The strict weight load fires first, and says so in its own words."""
        first = _saved_checkpoint(tmp_path, "sj_arch_weights", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(first["checkpoint"]["resume_state"])

        with pytest.raises(ValueError, match="incompatible with the target architecture"):
            _resumed(tmp_path, "sj_arch_target", {**_CONFIG, "hidden": [16]}, first, resume)

    def test_a_resume_into_a_different_configuration_is_refused(self, tmp_path: Path) -> None:
        """A different learning rate is a different experiment."""
        first = _saved_checkpoint(tmp_path, "sj_cfg_source", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(first["checkpoint"]["resume_state"])

        with pytest.raises(TrainingResumeMismatch) as raised:
            _resumed(tmp_path, "sj_cfg_target", {**_CONFIG, "lr": 0.5}, first, resume)

        assert raised.value.field == "configuration"

    def test_continuing_for_more_epochs_is_not_a_mismatch(self, tmp_path: Path) -> None:
        """Asking for more epochs is the ordinary reason to resume."""
        first = _saved_checkpoint(tmp_path, "sj_more_source", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(first["checkpoint"]["resume_state"])

        result = _resumed(tmp_path, "sj_more_target", {**_CONFIG, "epochs": 3}, first, resume)

        assert result["training_status"] == "completed"

    def test_a_checkpoint_without_a_position_supports_only_a_warm_start(self) -> None:
        with pytest.raises(TrainingResumeMismatch, match="resume state"):
            resume_state_from_payload({})

    def test_a_position_from_another_resume_schema_is_refused(self) -> None:
        with pytest.raises(TrainingResumeMismatch, match="resume schema"):
            resume_state_from_payload({"schema_version": "studio.training-resume.v99"})


class TestSavedPosition:
    def test_the_checkpoint_carries_what_a_resume_needs(self, tmp_path: Path) -> None:
        saved = _saved_checkpoint(tmp_path, "sj_position", {**_CONFIG, "epochs": 1})

        block = saved["checkpoint"]["resume_state"]
        assert set(block) == {
            "architecture",
            "config",
            "dataset_fingerprint",
            "epochs_completed",
            "optimiser_state",
            "rng_state",
            "schema_version",
        }
        assert block["schema_version"] == TRAINING_RESUME_SCHEMA_VERSION
        assert set(block["rng_state"]) == {"numpy", "python", "torch"}
        assert block["dataset_fingerprint"].startswith("sha256:")

    def test_the_position_loads_under_the_weights_only_boundary(self, tmp_path: Path) -> None:
        """A checkpoint arrives from a user; unpickling one is not an option.

        The generator states are integers and a hex string, and the optimiser
        state is tensors, precisely so this loader keeps working.
        """
        saved = _saved_checkpoint(tmp_path, "sj_weights_only", {**_CONFIG, "epochs": 1})

        block = load_training_resume_block(saved["blob"])

        assert block["schema_version"] == TRAINING_RESUME_SCHEMA_VERSION

    def test_a_checkpoint_with_no_position_yields_an_empty_block(self) -> None:
        import torch

        buffer = io.BytesIO()
        torch.save(
            {
                "schema_version": "studio.training.torch-state-dict.v1",
                "model_state_dict": {"w": torch.zeros(1)},
            },
            buffer,
        )

        assert load_training_resume_block(buffer.getvalue()) == {}


class TestSavedPositionSurface:
    def test_a_saved_position_re_resolves_its_configuration(self, tmp_path: Path) -> None:
        saved = _saved_checkpoint(tmp_path, "sj_position_config", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(saved["checkpoint"]["resume_state"])

        assert resume.resolved_config().hidden_widths == (8,)

    def test_the_public_summary_omits_the_tensors(self, tmp_path: Path) -> None:
        """A status document carries what a reader can act on, not buffers."""
        saved = _saved_checkpoint(tmp_path, "sj_position_public", {**_CONFIG, "epochs": 1})
        resume = resume_state_from_payload(saved["checkpoint"]["resume_state"])

        summary = resume.to_public_dict()

        assert set(summary) == {
            "architecture",
            "dataset_fingerprint",
            "epochs_completed",
            "schema_version",
        }

    def test_a_schema_mismatch_is_caught_when_the_state_is_applied(self) -> None:
        """Not only when it is loaded: a state built in memory is checked too."""
        import torch

        state = TrainingResumeState(
            schema_version="studio.training-resume.v0",
            epochs_completed=1,
            architecture="64->8->10",
            config={},
            optimiser_state={},
            rng_state={},
            dataset_fingerprint="sha256:none",
        )
        optimiser = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)

        with pytest.raises(TrainingResumeMismatch, match="resume schema"):
            apply_resume_state(state, optimiser=optimiser, architecture="64->8->10", config={})


class TestPartialState:
    def test_capturing_without_a_position_writes_weights_alone(self) -> None:
        """A caller that wants only a warm-startable artefact gets one."""
        import io

        import torch

        from sc_neurocore.studio._training_weight_capture import capture_weight_checkpoint
        from sc_neurocore.training import SpikingNet

        model = SpikingNet(n_input=4, n_hidden=[2], n_output=3)

        captured = capture_weight_checkpoint(
            model=model,
            architecture="4->2->3",
            model_info={},
            config={},
            final_metrics=None,
        )

        loaded = torch.load(io.BytesIO(captured.payload), map_location="cpu", weights_only=True)
        assert "resume_state" not in loaded
        assert captured.architecture == "4->2->3"

    def test_restoring_a_partial_generator_state_leaves_the_rest_alone(self) -> None:
        """A state naming only one generator moves only that one."""
        import random

        import numpy as np
        import torch

        from sc_neurocore.studio.training_resume import (
            _generator_states,
            _restore_generator_states,
        )

        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        saved = _generator_states()
        expected_python = random.random()
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        expected_numpy = float(np.random.rand())
        expected_torch = float(torch.rand(1))
        np.random.seed(99)
        torch.manual_seed(99)

        _restore_generator_states({"python": saved["python"]})
        # And a state naming none of them moves nothing.
        _restore_generator_states({})

        assert random.random() == expected_python
        assert float(np.random.rand()) == expected_numpy
        assert float(torch.rand(1)) == expected_torch


class TestDatasetFingerprint:
    def test_a_different_dataset_fingerprints_differently(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        first = DataLoader(TensorDataset(torch.zeros(8, 3), torch.zeros(8, dtype=torch.long)))
        second = DataLoader(TensorDataset(torch.ones(8, 3), torch.zeros(8, dtype=torch.long)))

        assert dataset_fingerprint(first) != dataset_fingerprint(second)

    def test_the_same_dataset_fingerprints_identically(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        features = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        labels = torch.zeros(8, dtype=torch.long)
        first = DataLoader(TensorDataset(features, labels))
        second = DataLoader(TensorDataset(features.clone(), labels.clone()))

        assert dataset_fingerprint(first) == dataset_fingerprint(second)

    def test_a_loader_without_a_dataset_says_so(self) -> None:
        assert dataset_fingerprint(object()) == "sha256:unavailable"

    def test_an_empty_dataset_still_fingerprints(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        empty = DataLoader(TensorDataset(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)))

        assert dataset_fingerprint(empty).startswith("sha256:")

    def test_a_non_tensor_sample_is_fingerprinted_by_its_value(self) -> None:
        class _Plain:
            dataset = ["first", "second", "third"]
            batch_size = 1

        first = _Plain()
        second = _Plain()
        second.dataset = ["first", "second", "different"]

        assert dataset_fingerprint(first) != dataset_fingerprint(second)
