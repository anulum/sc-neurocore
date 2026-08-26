# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aggregate coverage edge contracts

"""Close genuine public and foreign-boundary branches in the aggregate gate."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


SPIKE_STATS_MODULE_NAMES = ("correlation", "distance", "information", "variability")


@pytest.mark.parametrize("name", SPIKE_STATS_MODULE_NAMES)
def test_spike_stats_import_detects_real_extension(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Import-time acceleration flags reflect an importable compiled module."""

    module = importlib.import_module(f"sc_neurocore.analysis.spike_stats.{name}")
    extension_name = "sc_neurocore.analysis.spike_stats.spike_stats_core"
    fake = ModuleType(extension_name)
    monkeypatch.setitem(sys.modules, extension_name, fake)
    reloaded = importlib.reload(module)
    assert reloaded._HAS_RUST is True
    monkeypatch.undo()
    importlib.reload(reloaded)


def test_spike_stats_native_dispatch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Correlation, distances, and variability pass canonical arrays to Rust."""

    correlation = importlib.import_module("sc_neurocore.analysis.spike_stats.correlation")
    distance = importlib.import_module("sc_neurocore.analysis.spike_stats.distance")
    variability = importlib.import_module("sc_neurocore.analysis.spike_stats.variability")
    fake = SimpleNamespace(
        py_event_synchronization=lambda *_args: 0.25,
        py_victor_purpura_distance=lambda *_args: 1.0,
        py_spike_sync=lambda *_args: 0.5,
        py_lempel_ziv_complexity=lambda *_args: 0.75,
        py_approximate_entropy=lambda *_args: 0.1,
        py_sample_entropy=lambda *_args: 0.2,
        py_permutation_entropy=lambda *_args: 0.3,
    )
    for module in (correlation, distance, variability):
        monkeypatch.setattr(module, "_HAS_RUST", True)
        monkeypatch.setattr(module, "_ssc", fake)
    train = np.asarray([1, 0, 1, 0, 1], dtype=np.int64)
    assert correlation.event_synchronization(train, train) == 0.25
    assert distance.victor_purpura_distance(np.asarray([0.1]), np.asarray([0.2])) == 1.0
    assert distance.spike_sync(np.asarray([0.1]), np.asarray([0.2])) == 0.5
    assert variability.lempel_ziv_complexity(train) == 0.75
    assert variability.approximate_entropy(train) == 0.1
    assert variability.sample_entropy(train) == 0.2
    assert variability.permutation_entropy(train) == 0.3


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_mixed_dense_language_dispatch_converts_native_mapping(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Go and Mojo dense results enter the same typed public receipt."""

    module = importlib.import_module("sc_neurocore.compiler.mixed_dense_kernel")
    native_name = f"sc_neurocore.accel.{backend}.mixed_dense"
    fake = ModuleType(native_name)
    fake.mixed_dense_forward_batch = lambda *_args: {
        "outputs_q1616": [1],
        "overflow": [False],
        "underflow": [False],
    }
    monkeypatch.setitem(sys.modules, native_name, fake)
    result = getattr(module, f"_backend_{backend}")([1], [256], 1, 1)
    assert result.outputs_q1616.tolist() == [[1]]


def test_circt_success_writes_exported_verilog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Successful verify/export writes the exact CIRCT stdout artifact."""

    module = importlib.import_module("sc_neurocore.compiler.mlir_emitter")
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="module ok;", stderr=""),
        )
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    mlir = tmp_path / "model.mlir"
    verilog = tmp_path / "model.sv"
    mlir.write_text("module", encoding="utf-8")
    module._lower_with_circt("circt-opt", mlir, verilog)
    assert verilog.read_text(encoding="utf-8") == "module ok;"


def test_scc_native_odd_word_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packed SCC pads odd u32 input and passes the original bit count."""

    module = importlib.import_module("sc_neurocore.debug.sc_scope")
    observed: dict[str, object] = {}

    def packed(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any], bits: int) -> float:
        observed.update(a=a, b=b, bits=bits)
        return 0.5

    monkeypatch.setattr(module, "_HAS_RUST_SCC", True)
    monkeypatch.setattr(module, "_sdc", SimpleNamespace(py_scc_packed=packed))
    words = np.asarray([1, 2, 3], dtype=np.uint32)
    assert module.compute_scc(words, words) == 0.5
    assert np.asarray(observed["a"]).shape == (2,)
    assert observed["bits"] == 96


def test_evolution_native_import_and_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speciation detects and executes the compiled genomic-distance path."""

    module = importlib.import_module("sc_neurocore.evo_substrate.speciation")
    extension_name = "sc_neurocore.evo_substrate.evo_substrate_core"
    fake = ModuleType(extension_name)
    fake.py_genomic_distance = lambda *_args: 0.125
    monkeypatch.setitem(sys.modules, extension_name, fake)
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake)
    module = importlib.reload(module)
    genome = SimpleNamespace(to_vector=lambda: np.asarray([1.0]))
    assert module._HAS_RUST_EVO is True
    assert module.genomic_distance(genome, genome) == 0.125


def test_reference_trace_loader_skips_v1_payload_without_model(tmp_path: Any) -> None:
    """A v1-adjacent identity payload without a model remains outside the DSL corpus."""

    module = importlib.import_module("sc_neurocore.neurons.reference_trace_io")
    payload = {"schema_version": module.REFERENCE_TRACE_SCHEMA_VERSION}
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module._load_spec_file(path) is None


def test_gotm_brain_import_detects_local_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional GOTM adapter advertises availability when both symbols exist."""

    module = importlib.import_module("sc_neurocore.quantum_cognition.gotm_brain")
    fake = ModuleType("llm")
    fake.Endpoint = object
    fake.chat = lambda *_args, **_kwargs: "ok"
    monkeypatch.setitem(sys.modules, "llm", fake)
    reloaded = importlib.reload(module)
    assert reloaded.HAS_LLM is True
    monkeypatch.undo()
    importlib.reload(reloaded)


def _behavior_run() -> Any:
    """Construct a ten-window receipt with meaningful circular observables."""

    backends = importlib.import_module("sc_neurocore.network.sc_compte_wm_backends")
    network = importlib.import_module("sc_neurocore.network.sc_compte_wm_network")
    sc = importlib.import_module("sc_neurocore.network.sc_compte_wm")
    stats = [sc.SCCompteWMActivityStatistics(0.0, 0.0, 180.0, 0.0, None) for _ in range(10)]
    stats[1] = sc.SCCompteWMActivityStatistics(10.0, 0.0, 180.0, 0.9, 10.0)
    stats[2] = sc.SCCompteWMActivityStatistics(10.0, 0.0, 178.0, 0.9, 10.0)
    stats[3] = sc.SCCompteWMActivityStatistics(10.0, 0.0, 182.0, 0.9, 10.0)
    stats[6] = sc.SCCompteWMActivityStatistics(10.0, 0.0, 180.0, 0.9, 10.0)
    stats[7] = sc.SCCompteWMActivityStatistics(100.0, 0.0, 0.0, 0.0, None)
    windows = tuple(
        network.SCCompteWMWindowReceipt(i * 250.0, (i + 1) * 250.0, 1, 0, value)
        for i, value in enumerate(stats)
    )
    receipt = network.SCCompteWMRunReceipt(
        "v1", 42, 2500.0, 125000, 10, 0, windows, "input", "spike", "state"
    )
    return backends.SCCompteWMBackendRun("rust", 1, receipt)


def test_behavior_assessment_and_trial_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behavior classification consumes all ten windows and preserves execution custody."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_behavior")
    run = _behavior_run()
    wrong = replace(run, receipt=replace(run.receipt, duration_ms=1.0))
    with pytest.raises(ValueError, match="ten-window"):
        module.assess_sc_compte_wm_behavior(wrong)
    missing_stats = replace(
        run,
        receipt=replace(
            run.receipt,
            windows=(replace(run.receipt.windows[0], statistics=None), *run.receipt.windows[1:]),
        ),
    )
    with pytest.raises(ValueError, match="every behavior window"):
        module.assess_sc_compte_wm_behavior(missing_stats)
    trial = module.assess_sc_compte_wm_behavior(run)
    assert trial.seed == 42
    assert trial.metrics.cue_error_deg == 0.0
    monkeypatch.setattr(module, "run_sc_compte_wm_network", lambda *_args, **_kwargs: run)
    dispatched = module.run_sc_compte_wm_behavior_trial(backend="rust", seed=42)
    assert dispatched.input_sha256 == "input"


def test_behavior_ensemble_requires_references_and_exact_runtime_anchors() -> None:
    """Ensemble acceptance requires all seeds, signed drift, and exact five-lane custody."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_behavior")
    with pytest.raises(ValueError, match="at least one"):
        module.summarize_sc_compte_wm_behavior_ensemble(())
    base = module.assess_sc_compte_wm_behavior(_behavior_run())
    with pytest.raises(ValueError, match="reference seed"):
        module.summarize_sc_compte_wm_behavior_ensemble((base,))
    trials = []
    for seed, drift in zip((41, 42, 43), (-1.0, 0.5, 1.0), strict=True):
        metrics = replace(base.metrics, signed_delay_drift_deg=drift)
        trials.append(replace(base, backend="rust", seed=seed, metrics=metrics, passed=True))
    for backend in ("python", "julia", "go", "mojo"):
        trials.append(replace(base, backend=backend, seed=42, passed=True))
    ensemble = module.summarize_sc_compte_wm_behavior_ensemble(tuple(trials))
    assert ensemble.all_runtime_input_spike_count_exact is True
    assert ensemble.bidirectional_seed_drift is True
    assert ensemble.passed is True


def test_sc_compte_spec_and_population_validation() -> None:
    """The frozen network specification rejects every unsafe structural variant."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm")
    with pytest.raises(ValueError, match="positive"):
        module._positive_finite("value", 0.0)
    with pytest.raises(ValueError, match="center_deg"):
        module.circular_distance_deg([0.0], np.nan)
    with pytest.raises(ValueError, match="angles_deg"):
        module.circular_distance_deg([[0.0]], 0.0)
    with pytest.raises(ValueError, match="circular angles"):
        module.circular_displacement_deg(np.nan, 0.0)
    with pytest.raises(ValueError, match="potentials"):
        module.SCCompteCellSpec(0.5, 25.0, leak_reversal_mv=np.nan)
    with pytest.raises(ValueError, match="below"):
        module.SCCompteCellSpec(0.5, 25.0, reset_mv=-40.0, threshold_mv=-50.0)
    with pytest.raises(ValueError, match="identity"):
        module.SCCompteWMNetworkSpec(identity="other")
    with pytest.raises(ValueError, match="fastest"):
        module.SCCompteWMNetworkSpec(dt_ms=3.0)
    with pytest.raises(ValueError, match="peaks"):
        module.SCCompteWMNetworkSpec(ee_j_plus=1.0)
    with pytest.raises(ValueError, match="180"):
        module.SCCompteWMNetworkSpec(ee_sigma_deg=181.0)
    spec = module.SCCompteWMNetworkSpec()
    with pytest.raises(ValueError, match="population"):
        spec.preferred_angles_deg("other")
    with pytest.raises(ValueError, match="target_angles"):
        spec.connectivity_footprint("ee", 0.0, [])
    assert np.all(spec.connectivity_footprint("ei", 0.0, [0.0, 1.0]) == 1.0)
    with pytest.raises(ValueError, match="non-degenerate"):
        spec.connectivity_footprint("ee", 0.0, [0.0])
    sharp = module.SCCompteWMNetworkSpec(ee_j_plus=100.0)
    with pytest.raises(ValueError, match="non-positive distal"):
        sharp.connectivity_footprint("ee", 0.0, [0.0, 90.0, 180.0, 270.0])
    with pytest.raises(ValueError, match="finite"):
        module.summarize_activity(spec, [np.nan] * 2048, [0] * 512, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        module.summarize_activity(spec, [-1] * 2048, [0] * 512, 1.0)


def test_sc_compte_counter_drive_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter-Poisson generation rejects excessive means and bounded-CDF overflow."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_drive")
    with pytest.raises(ValueError, match="mean"):
        module.CounterPoissonDrive(1, 33_000.0, 1.0, 0, 0)
    drive = module.CounterPoissonDrive(1, 1.0, 1.0, 0, 0)
    monkeypatch.setattr(module.math, "exp", lambda _value: 0.0)
    with pytest.raises(ValueError, match="bounded event range"):
        drive._cdf()


def test_sc_compte_behavior_protocol_rejects_nonfinite_and_negative_start() -> None:
    """Behavior protocol values and epoch starts remain finite and non-negative."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_behavior")
    with pytest.raises(ValueError, match="finite"):
        module.SCCompteWMBehaviorProtocol(cue_center_deg=np.nan)
    with pytest.raises(ValueError, match="non-negative"):
        module.SCCompteWMBehaviorProtocol(cue_start_ms=-1.0)


def test_sc_compte_stimulus_state_and_run_validation() -> None:
    """Python network validation mirrors the Mojo state and timing envelopes."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_network")
    with pytest.raises(ValueError, match="positive duration"):
        module.SCCompteWMStimulus(0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="kind"):
        module.SCCompteWMStimulus(0.0, 1.0, 1.0, kind="other")
    with pytest.raises(ValueError, match="center_deg"):
        module.SCCompteWMStimulus(0.0, 1.0, 1.0, center_deg=None)
    with pytest.raises(ValueError, match="center_deg=None"):
        module.SCCompteWMStimulus(0.0, 1.0, 1.0, kind="global_current", center_deg=0.0)
    network = module.SCCompteWMNetwork()
    network._state.step_index = 7
    network.reset()
    assert network._state.step_index == 0
    state = network.state()
    state.step_index = True
    with pytest.raises(ValueError, match="step_index"):
        network._validate_state(state)
    state = network.state()
    state.v_exc_mv = state.v_exc_mv[:-1]
    with pytest.raises(ValueError, match="shape"):
        network._validate_state(state)
    state = network.state()
    state.v_exc_mv[0] = 101.0
    with pytest.raises(ValueError, match="excitatory voltage"):
        network._validate_state(state)
    state = network.state()
    state.v_inh_mv[0] = 101.0
    with pytest.raises(ValueError, match="inhibitory voltage"):
        network._validate_state(state)
    state = network.state()
    state.external_ampa_exc[0] = -1.0
    with pytest.raises(ValueError, match="channel state"):
        network._validate_state(state)
    state = network.state()
    state.recurrent_nmda[0] = 2.0
    with pytest.raises(ValueError, match="bounded by one"):
        network._validate_state(state)
    with pytest.raises(ValueError, match="integer array"):
        network._events("events", [], 1)
    with pytest.raises(ValueError, match="non-negative"):
        network._events("events", [-1], 1)
    localized = module.SCCompteWMStimulus(0.0, 0.02, 1.0, center_deg=0.0)
    assert np.max(network._stimulus_current(0.0, (localized,))) > 0.0
    with pytest.raises(ValueError, match="duration_ms"):
        network.run(np.nan)
    with pytest.raises(ValueError, match="integral number"):
        network.run(0.03)
    with pytest.raises(ValueError, match="statistics_window_ms"):
        network.run(0.02, statistics_window_ms=np.nan)
    with pytest.raises(ValueError, match="integral number"):
        network.run(0.02, statistics_window_ms=0.03)


def test_sc_compte_structured_ei_aggregate_path() -> None:
    """Structured E-to-I recurrence uses its explicit circular convolution."""

    sc = importlib.import_module("sc_neurocore.network.sc_compte_wm")
    network_module = importlib.import_module("sc_neurocore.network.sc_compte_wm_network")
    network = network_module.SCCompteWMNetwork(sc.SCCompteWMNetworkSpec(structured_ei=True))
    aggregates = network._recurrent_aggregates(np.zeros(2048), np.zeros(512))
    assert tuple(values.shape for values in aggregates) == ((2048,), (512,), (2048,), (512,))


def test_sc_compte_backend_argument_and_command_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend commands preserve flags and fail explicitly without a runtime."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_backends")
    monkeypatch.setattr(module, "_runtime_executable", lambda _name: None)
    with pytest.raises(module.SCCompteWMBackendUnavailable, match="PATH"):
        module._require_runtime_executable("missing")
    spec_module = importlib.import_module("sc_neurocore.network.sc_compte_wm")
    spec = spec_module.SCCompteWMNetworkSpec(
        structured_ei=True, modulated=True, allow_recurrent_autapses=True
    )
    arguments = module._native_args(spec, 1.0, 1.0, ())
    assert "--structured-ei" in arguments
    assert "--modulated" in arguments
    assert "--allow-recurrent-autapses" in arguments
    with pytest.raises(ValueError, match="does not use"):
        module._command("python", [])


def test_sc_compte_native_command_failure_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process, identity, and timing failures become one governed unavailable error."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_backends")
    monkeypatch.setattr(module, "_command", lambda *_args: (["runner"], module._REPOSITORY))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    with pytest.raises(module.SCCompteWMBackendUnavailable, match="execution failed"):
        module._run_native_command("go", [], None)

    def completed(payload: dict[str, object]) -> Any:
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        module.subprocess, "run", lambda *_args, **_kwargs: completed({"runtime": "rust"})
    )
    with pytest.raises(module.SCCompteWMBackendUnavailable, match="invalid run receipt"):
        module._run_native_command("go", [], None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: completed({"runtime": "go", "execution_ns": -1}),
    )
    with pytest.raises(module.SCCompteWMBackendUnavailable, match="invalid run receipt"):
        module._run_native_command("go", [], None)


def test_sc_compte_public_backend_rejects_timeout_and_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public dispatch validates timeout and selected runtime before launch."""

    module = importlib.import_module("sc_neurocore.network.sc_compte_wm_backends")
    with pytest.raises(ValueError, match="timeout_s"):
        module.run_sc_compte_wm_network(0.02, backend="python", timeout_s=0.0)
    unavailable = tuple(
        module.SCCompteWMBackendStatus(name, False, "none", f"{name} absent")
        for name in module._BACKENDS
    )
    monkeypatch.setattr(module, "sc_compte_wm_backend_status", lambda: unavailable)
    with pytest.raises(module.SCCompteWMBackendUnavailable, match="go absent"):
        module.run_sc_compte_wm_network(0.02, backend="go")
