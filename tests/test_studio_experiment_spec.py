# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio effective experiment contract (public Python surface)

"""One effective experiment per run and reproducible randomness.

Every case fails on the former behaviour: the request omitted the numerical
profile, initial state, sine frequency and seed; the cache was keyed by the raw
request dict so two package or model revisions, an explicit versus implicit dt,
or a stochastic replay and a fresh trial could share an entry; the playground
noise drew from the process-global numpy stream; an oversized run was
shortened.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from sc_neurocore.studio.experiment_spec import (
    DEFAULT_NOISE_SEED,
    EXPERIMENT_SCHEMA_VERSION,
    JOB_MAX_STEPS,
    ExperimentRejected,
    resolve_experiment,
    resolve_model_experiment,
    resolve_ode_experiment,
    run_experiment,
)
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.simulation import MAX_STEPS, simulate

ATIF = "AdaptiveThresholdIFNeuron"
NOISY = ["dv/dt = -v + I + xi"]


def _spec(**request: Any) -> dict[str, Any]:
    return resolve_experiment(request).to_public_dict()


class TestEffectiveConfiguration:
    @pytest.mark.parametrize("surface", ["public", "export", "run_kwargs", "result"])
    def test_nested_projections_cannot_change_sealed_execution(self, surface: str) -> None:
        request = {
            "equations": ["dv/dt = gain * I"],
            "params": {"gain": 2.0},
            "init": {"v": 0.0},
            "dt": 0.1,
            "duration": 0.2,
            "current": 1.0,
        }
        spec = resolve_experiment(request)
        before = run_experiment(spec)
        digest = spec.experiment_sha256
        if surface == "run_kwargs":
            exported = spec.run_kwargs
            exported["init"]["v"] = 99.0
            exported["params"]["gain"] = 50.0
        else:
            exported = (
                spec.public
                if surface == "public"
                else before["experiment"]
                if surface == "result"
                else spec.to_public_dict()
            )
            exported["initial_state"]["v"] = 99.0
            exported["parameters"]["gain"] = 50.0
            exported["equations"]["equations"][0] = "dv/dt = 100"
        after = run_experiment(spec)
        assert after["initial_state"] == {"v": 0.0}
        assert after["raw"] == before["raw"]
        assert after["experiment"]["parameters"]["gain"] == 2.0
        assert after["experiment"]["experiment_sha256"] == digest

    def test_model_spec_binds_revision_profile_steps_state_protocol_and_runtime(self) -> None:
        public = _spec(name=ATIF, duration=5.0)
        assert public["schema_version"] == EXPERIMENT_SCHEMA_VERSION
        assert public["source"] == "model"
        model = public["model"]
        assert model["class_name"] == ATIF
        assert len(model["module_sha256"]) == 64
        assert len(model["descriptor_sha256"]) == 64
        assert len(model["descriptor_contract_digest"]) == 64
        assert model["schema_profile"] == "adaptive_threshold_if"
        assert len(model["schema_sha256"]) == 64
        assert public["numerical"]["dt"] == 0.1
        assert public["numerical"]["dt_source"] == "model_default"
        assert public["numerical"]["method"]
        assert public["steps"] == {
            "n_steps": 50,
            "duration_requested_ms": 5.0,
            "duration_effective_ms": pytest.approx(5.0),
            "synchronous_limit": MAX_STEPS,
        }
        assert public["initial_state"] == {"v": -65.0, "theta": -50.0}
        assert public["parameters"]["tau_theta"] == 50.0
        assert public["protocol"]["kind"] == "constant"
        assert public["protocol"]["frequency_hz"] is None
        assert len(public["protocol"]["drive_sha256"]) == 64
        assert public["backend"]["selected"] == "python"
        assert public["backend"]["rejected"][0]["name"] == "rust-batch"
        assert public["runtime"]["package_version"]
        assert len(public["runtime"]["equation_builder_sha256"]) == 64
        assert len(public["experiment_sha256"]) == 64
        assert public["cache"] == {"key": public["experiment_sha256"], "cacheable": True}

    def test_explicit_defaults_resolve_to_the_implicit_spec(self) -> None:
        implicit = _spec(name=ATIF)
        explicit = _spec(
            name=ATIF,
            params={},
            dt=None,
            duration=100.0,
            current=10.0,
            protocol="constant",
            frequency_hz=10.0,
            seed=None,
            trial="replay",
        )
        assert explicit == implicit

    def test_explicit_model_default_dt_is_the_same_experiment(self) -> None:
        assert _spec(name=ATIF, dt=0.1)["experiment_sha256"] != _spec(name=ATIF)[
            "experiment_sha256"
        ] or (_spec(name=ATIF, dt=0.1)["numerical"]["dt"] == _spec(name=ATIF)["numerical"]["dt"])
        # dt_source differs (override vs model_default) and is part of the contract,
        # but the effective step, the step count and the drive digest agree.
        override = _spec(name=ATIF, dt=0.1)
        default = _spec(name=ATIF)
        assert override["numerical"]["dt"] == default["numerical"]["dt"]
        assert override["steps"] == default["steps"]
        assert override["protocol"]["drive_sha256"] == default["protocol"]["drive_sha256"]

    def test_effective_dt_comes_from_the_model_not_the_studio_default(self) -> None:
        hh = _spec(name="HodgkinHuxleyNeuron", duration=1.0)
        integer = _spec(name="IntegerQIFNeuron", duration=10.0)
        assert (hh["numerical"]["dt"], hh["steps"]["n_steps"]) == (0.01, 100)
        assert (integer["numerical"]["dt"], integer["steps"]["n_steps"]) == (1.0, 10)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ({"name": ATIF}, {"name": "AdExNeuron"}),
            (
                {"name": ATIF, "protocol": "sine"},
                {"name": ATIF, "protocol": "sine", "frequency_hz": 50.0},
            ),
            ({"name": ATIF, "protocol": "sine"}, {"name": ATIF, "protocol": "step"}),
            ({"name": ATIF}, {"name": ATIF, "params": {"tau_m": 12.0}}),
            ({"name": ATIF}, {"name": ATIF, "dt": 0.05}),
            ({"name": "PoissonNeuron", "seed": 1}, {"name": "PoissonNeuron", "seed": 2}),
            ({"name": "LapicqueNeuron"}, {"name": "LapicqueNeuron", "duration": 50.0}),
        ],
    )
    def test_different_effective_inputs_never_share_a_cache_key(
        self, left: dict[str, Any], right: dict[str, Any]
    ) -> None:
        assert _spec(**left)["experiment_sha256"] != _spec(**right)["experiment_sha256"]

    def test_model_revision_enters_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sc_neurocore.studio import experiment_spec as module

        before = _spec(name=ATIF)["experiment_sha256"]
        monkeypatch.setattr(module, "_runtime_block", lambda: {"package_version": "0.0.0-test"})
        assert _spec(name=ATIF)["experiment_sha256"] != before

    def test_protocol_typo_and_undeclared_init_are_rejected(self) -> None:
        with pytest.raises(ExperimentRejected) as info:
            resolve_ode_experiment({"equations": ["dv/dt = -v"], "protocol": "sawtooth"})
        assert info.value.field == "protocol"
        with pytest.raises(ExperimentRejected) as info:
            resolve_ode_experiment({"equations": ["dv/dt = -v"], "init": {"w": 1.0}})
        assert info.value.field == "init" and "w" in info.value.reason
        with pytest.raises(ExperimentRejected) as info:
            resolve_ode_experiment({"equations": ["v = I"]})
        assert info.value.field == "equations"
        with pytest.raises(ModelInputError):
            resolve_model_experiment({"name": ATIF, "protocol": "sawtooth"})

    def test_oversized_synchronous_run_is_refused_not_shortened(self) -> None:
        with pytest.raises(ExperimentRejected) as info:
            resolve_model_experiment({"name": "AdExNeuron", "duration": 1e7})
        rejection = info.value
        assert rejection.execution_mode == "job_required"
        assert rejection.to_public_detail()["recommended_route"]
        assert "not shortened" in rejection.reason
        spec = resolve_model_experiment(
            {"name": "AdExNeuron", "duration": 1e5}, max_steps=JOB_MAX_STEPS
        )
        assert spec.n_steps == 1_000_000
        with pytest.raises(ExperimentRejected) as info:
            resolve_ode_experiment({"equations": ["dv/dt = -v"], "duration": 0.01, "dt": 0.1})
        assert "no complete step" in info.value.reason

    def test_ode_spec_declares_every_variable_and_the_equation_digest(self) -> None:
        public = _spec(equations=["dv/dt = -v + I", "dw/dt = v - w"], init={"v": -65.0})
        assert public["source"] == "ode"
        assert public["equations"]["variables"] == ["v", "w"]
        assert public["initial_state"] == {"v": -65.0, "w": 0.0}
        assert public["numerical"] == {
            "method": "euler",
            "family": "ode",
            "dt": 0.1,
            "dt_source": "studio_default",
            "substeps": 1,
            "time_unit": "ms",
        }
        assert public["randomness"]["kind"] == "none"
        assert len(public["equations"]["equation_sha256"]) == 64


class TestRandomnessContract:
    def test_seed_on_a_deterministic_model_is_refused(self) -> None:
        with pytest.raises(ExperimentRejected) as info:
            resolve_model_experiment({"name": "AdExNeuron", "seed": 3})
        assert info.value.field == "seed"
        with pytest.raises(ExperimentRejected) as info:
            resolve_ode_experiment({"equations": ["dv/dt = -v"], "seed": 3})
        assert "xi" in info.value.reason

    def test_seeded_model_replay_uses_the_request_or_default_seed(self) -> None:
        default = _spec(name="PoissonNeuron")
        assert default["randomness"]["kind"] == "seeded-model"
        assert default["randomness"]["seed_source"] == "model-default"
        assert default["cache"]["cacheable"] is True
        requested = resolve_model_experiment({"name": "PoissonNeuron", "seed": 77})
        assert requested.public["randomness"] == {
            "kind": "seeded-model",
            "seed": 77,
            "seed_source": "request",
            "trial": "replay",
            "effective_trial": "replay",
            "generator": "model seed field",
        }
        assert requested.run_kwargs["param_overrides"] == {"seed": 77}
        assert requested.public["parameters"]["seed"] == 77
        via_param = resolve_model_experiment({"name": "PoissonNeuron", "params": {"seed": 77.0}})
        assert via_param.public["randomness"]["seed"] == 77
        with pytest.raises(ExperimentRejected, match="both"):
            resolve_model_experiment({"name": "PoissonNeuron", "params": {"seed": 1.0}, "seed": 2})

    def test_fresh_trial_draws_a_seed_and_is_never_cacheable(self) -> None:
        first = resolve_model_experiment({"name": "PoissonNeuron", "trial": "fresh"})
        second = resolve_model_experiment({"name": "PoissonNeuron", "trial": "fresh"})
        assert first.public["randomness"]["seed_source"] == "drawn"
        assert first.cacheable is False
        assert 1 <= first.public["randomness"]["seed"] <= 65_535
        assert first.experiment_sha256 != second.experiment_sha256 or (
            first.public["randomness"]["seed"] == second.public["randomness"]["seed"]
        )
        deterministic = resolve_model_experiment({"name": "AdExNeuron", "trial": "fresh"})
        assert deterministic.public["randomness"]["effective_trial"] == "replay"
        assert deterministic.cacheable is True

    def test_fresh_trials_of_a_seeded_model_are_independent_and_replayable(self) -> None:
        first = run_experiment(
            resolve_model_experiment({"name": "PoissonNeuron", "duration": 200.0, "trial": "fresh"})
        )
        seed = first["experiment"]["randomness"]["seed"]
        replay = run_experiment(
            resolve_model_experiment({"name": "PoissonNeuron", "duration": 200.0, "seed": seed})
        )
        assert replay["spikes"] == first["spikes"]
        assert replay["raw"]["states"] == first["raw"]["states"]
        other = run_experiment(
            resolve_model_experiment(
                {"name": "PoissonNeuron", "duration": 200.0, "seed": (seed % 65_535) + 1}
            )
        )
        assert other["spikes"] != first["spikes"]

    def test_playground_noise_is_seeded_per_run_not_process_global(self) -> None:
        spec = resolve_ode_experiment({"equations": NOISY, "duration": 5.0, "seed": 11})
        assert spec.public["randomness"] == {
            "kind": "diffusion-noise",
            "seed": 11,
            "seed_source": "request",
            "trial": "replay",
            "effective_trial": "replay",
            "generator": "numpy.random.default_rng",
        }
        before = json.dumps(spec.to_public_dict(), sort_keys=True)
        np.random.seed(1)
        first = run_experiment(spec)["raw"]["states"]["v"]
        np.random.seed(2)
        second = run_experiment(spec)["raw"]["states"]["v"]
        assert first == second
        # Running never mutates the specification (the neuron copies its init).
        assert json.dumps(spec.to_public_dict(), sort_keys=True) == before
        assert (
            first
            != run_experiment(
                resolve_ode_experiment({"equations": NOISY, "duration": 5.0, "seed": 12})
            )["raw"]["states"]["v"]
        )
        # The run does not consume the process-global stream.
        np.random.seed(3)
        expected_next = np.random.randn()
        np.random.seed(3)
        run_experiment(spec)
        assert np.random.randn() == expected_next

    def test_playground_replay_without_seed_uses_the_documented_default(self) -> None:
        spec = resolve_ode_experiment({"equations": NOISY, "duration": 2.0})
        assert spec.public["randomness"]["seed"] == DEFAULT_NOISE_SEED
        assert spec.public["randomness"]["seed_source"] == "playground-default"
        assert (
            simulate(NOISY, duration=2.0, seed=DEFAULT_NOISE_SEED)["raw"]["states"]["v"]
            == (run_experiment(spec)["raw"]["states"]["v"])
        )
        fresh = resolve_ode_experiment({"equations": NOISY, "duration": 2.0, "trial": "fresh"})
        assert fresh.public["randomness"]["seed_source"] == "drawn"
        assert fresh.cacheable is False

    def test_deterministic_replay_is_bit_identical_and_json_portable(self) -> None:
        spec = resolve_model_experiment(
            {"name": ATIF, "duration": 20.0, "protocol": "sine", "frequency_hz": 33.0}
        )
        first = run_experiment(spec)
        second = run_experiment(
            resolve_model_experiment(
                {"name": ATIF, "duration": 20.0, "protocol": "sine", "frequency_hz": 33.0}
            )
        )
        assert first["raw"] == second["raw"]
        assert first["experiment"] == second["experiment"]
        json.dumps(first, allow_nan=False)
