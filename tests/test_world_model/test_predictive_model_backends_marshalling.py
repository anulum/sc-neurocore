# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (marshalling) from former test_predictive_model_backends.py

from __future__ import annotations

from tests.test_world_model.predictive_model_backends_support import *  # noqa: F403


def test_rust_marshalling_returns_validated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def rust_filter(**kwargs: object) -> object:
        captured.update(kwargs)
        return _native_mapping()

    monkeypatch.setattr(backends, "_rust_kalman_filter", rust_filter)
    observations, controls = _inputs()
    result = backends.filter_native("rust", _model(), observations, controls)

    assert captured["t_len"] == 3
    assert captured["p_dim"] == 2
    assert captured["m_dim"] == 0
    assert len(cast(list[float], captured["a_flat"])) == 4
    assert result.means.shape == (3, 2)
    assert result.log_likelihood == -4.0


def test_rust_marshalling_rejects_non_numeric_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _native_mapping()
    mapping["log_likelihood"] = object()
    monkeypatch.setattr(backends, "_rust_kalman_filter", lambda **_kwargs: mapping)
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match="non-numeric log_likelihood"):
        backends.filter_native("rust", _model(), observations, controls)


def test_rust_marshalling_rejects_out_of_range_numeric_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _native_mapping()
    mapping["log_likelihood"] = 10**10_000
    monkeypatch.setattr(backends, "_rust_kalman_filter", lambda **_kwargs: mapping)
    observations, controls = _inputs()

    with pytest.raises(RuntimeError, match="out-of-range log_likelihood"):
        backends.filter_native("rust", _model(), observations, controls)


@pytest.mark.parametrize(
    "malformed_result",
    [
        object(),
        {"means": np.zeros((3, 2))},
        {
            **_native_mapping(),
            "means": object(),
        },
        {
            **_native_mapping(),
            "means": [[10**10_000]],
        },
        {
            **_native_mapping(),
            "means": np.zeros((2, 2)),
        },
        {
            "means": np.zeros((2, 2)),
            "covariances": np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            "pred_means": np.zeros((2, 2)),
            "pred_covariances": np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            "log_likelihood": -1.0,
        },
    ],
)
def test_rust_marshalling_rejects_malformed_payloads(
    monkeypatch: pytest.MonkeyPatch,
    malformed_result: object,
) -> None:
    monkeypatch.setattr(
        backends,
        "_rust_kalman_filter",
        lambda **_kwargs: malformed_result,
    )
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match="returned|missing"):
        backends.filter_native("rust", _model(), observations, controls)


def test_julia_marshalling_reads_named_result_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = _native_mapping()
    result_object = SimpleNamespace(
        means=mapping["means"],
        covariances=mapping["covariances"],
        pred_means=mapping["pred_means"],
        pred_covs=mapping["pred_covariances"],
        log_lik=mapping["log_likelihood"],
    )

    class JuliaModule:
        @staticmethod
        def kalman_filter(*_args: object) -> object:
            return result_object

    monkeypatch.setattr(backends, "_julia_module", JuliaModule())
    observations, controls = _inputs()
    result = backends.filter_native("julia", _model(), observations, controls)
    assert result.log_likelihood == -4.0


def test_julia_marshalling_rejects_incomplete_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class JuliaModule:
        @staticmethod
        def kalman_filter(*_args: object) -> object:
            return SimpleNamespace(means=np.zeros((3, 2)))

    monkeypatch.setattr(backends, "_julia_module", JuliaModule())
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match="incomplete filter result"):
        backends.filter_native("julia", _model(), observations, controls)


def test_mojo_raw_address_marshalling(monkeypatch: pytest.MonkeyPatch) -> None:
    class MojoLibrary:
        @staticmethod
        def kalman_filter_c(*args: object) -> None:
            values = [cast(int, value) for value in args]
            _write_c_outputs(values[10], values[13], *values[14:19])

    monkeypatch.setattr(backends, "_mojo_lib", MojoLibrary())
    observations, controls = _inputs()
    result = backends.filter_native("mojo", _model(), observations, controls)
    assert result.covariances.shape == (3, 2, 2)
    assert result.log_likelihood == -4.0


def test_go_pointer_marshalling(monkeypatch: pytest.MonkeyPatch) -> None:
    class GoLibrary:
        @staticmethod
        def kalman_filter_c(*args: object) -> None:
            time_steps = cast(ctypes.c_int, args[10]).value
            state_dim = cast(ctypes.c_int, args[13]).value
            addresses = [
                ctypes.addressof(
                    cast(_DoublePointer, pointer).contents,
                )
                for pointer in args[14:19]
            ]
            _write_c_outputs(time_steps, state_dim, *addresses)

    monkeypatch.setattr(backends, "_go_lib", GoLibrary())
    observations, controls = _inputs()
    result = backends.filter_native("go", _model(), observations, controls)
    assert result.pred_covariances.shape == (3, 2, 2)
    assert result.log_likelihood == -4.0
