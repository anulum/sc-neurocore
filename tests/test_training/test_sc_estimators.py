# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for differentiable stochastic-computing estimators

"""Tests for differentiable stochastic-computing training estimators."""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from sc_neurocore.training.sc_estimators import (
    DifferentiableSCConfig,
    estimate_bitstream_statistics,
    finite_difference_gradients,
    relaxed_sc_multiply,
    sample_sc_bitstreams,
    sampled_sc_multiply,
)


def _base_config(**overrides):
    values = {
        "bitstream_length": 256,
        "encoding": "bipolar",
        "generator": "sobol",
        "estimator": "pathwise_relaxation",
        "input_seed": 17,
        "weight_seed": 29,
        "correlation": 0.0,
    }
    values.update(overrides)
    return DifferentiableSCConfig(**values)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("bitstream_length", 0, "bitstream_length"),
        ("encoding", "ternary", "encoding"),
        ("generator", "xoroshiro", "generator"),
        ("estimator", "unknown", "estimator"),
        ("correlation", 1.5, "correlation"),
    ],
)
def test_config_rejects_invalid_contract_values(field, value, match):
    with pytest.raises(ValueError, match=match):
        _base_config(**{field: value})


def test_config_rejects_seed_reuse():
    with pytest.raises(ValueError, match="seed"):
        _base_config(input_seed=19, weight_seed=19)


def test_relaxed_bipolar_multiply_matches_independent_product_and_autograd():
    cfg = _base_config(encoding="bipolar", correlation=0.0)
    x = torch.tensor([0.25, -0.5, 0.75], requires_grad=True)
    w = torch.tensor([-0.4, 0.2, 0.5], requires_grad=True)

    result = relaxed_sc_multiply(x, w, cfg)
    assert torch.allclose(result.value, x * w)

    result.value.sum().backward()
    assert torch.allclose(x.grad, w)
    assert torch.allclose(w.grad, x)


def test_relaxed_unipolar_multiply_models_positive_correlation_bias():
    independent = _base_config(encoding="unipolar", correlation=0.0)
    correlated = _base_config(encoding="unipolar", correlation=0.5)
    x = torch.tensor([0.25, 0.5, 0.75])
    w = torch.tensor([0.2, 0.5, 0.8])

    y_independent = relaxed_sc_multiply(x, w, independent).value
    y_correlated = relaxed_sc_multiply(x, w, correlated).value

    assert torch.allclose(y_independent, x * w)
    assert torch.all(y_correlated > y_independent)


def test_relaxed_multiply_rejects_values_outside_encoding_domain():
    cfg = _base_config(encoding="unipolar")
    with pytest.raises(ValueError, match="unipolar"):
        relaxed_sc_multiply(torch.tensor([-0.1]), torch.tensor([0.5]), cfg)


def test_finite_difference_gradient_matches_autograd_for_relaxed_path():
    cfg = _base_config(encoding="bipolar", correlation=0.0)
    x0 = 0.31
    w0 = -0.42
    x = torch.tensor([x0], requires_grad=True)
    w = torch.tensor([w0], requires_grad=True)

    relaxed_sc_multiply(x, w, cfg).value.sum().backward()
    fd_x, fd_w = finite_difference_gradients(x0, w0, cfg)

    assert x.grad.item() == pytest.approx(fd_x, abs=1e-5)
    assert w.grad.item() == pytest.approx(fd_w, abs=1e-5)


def test_sobol_sampler_is_reproducible_and_tracks_unipolar_probability():
    cfg = _base_config(
        bitstream_length=256,
        encoding="unipolar",
        generator="sobol",
        input_seed=101,
        weight_seed=211,
    )
    values = torch.tensor([0.25, 0.5, 0.75])

    first = sample_sc_bitstreams(values, cfg, role="input")
    second = sample_sc_bitstreams(values, cfg, role="input")

    assert torch.equal(first.bits, second.bits)
    assert first.bits.shape == (3, 256)
    assert set(torch.unique(first.bits).tolist()).issubset({0.0, 1.0})
    assert torch.allclose(first.decoded, values, atol=1.0 / cfg.bitstream_length)


def test_bernoulli_sampler_uses_distinct_input_and_weight_seed_lanes():
    cfg = _base_config(
        bitstream_length=512,
        encoding="bipolar",
        generator="bernoulli",
        input_seed=31,
        weight_seed=47,
    )
    values = torch.tensor([0.2, -0.6])

    input_streams = sample_sc_bitstreams(values, cfg, role="input")
    weight_streams = sample_sc_bitstreams(values, cfg, role="weight")

    assert not torch.equal(input_streams.bits, weight_streams.bits)
    assert torch.allclose(input_streams.decoded, values, atol=0.12)
    assert torch.allclose(weight_streams.decoded, values, atol=0.12)


def test_lfsr_sampler_requires_supported_polynomial_and_matches_domain():
    cfg = _base_config(
        bitstream_length=1024,
        encoding="unipolar",
        generator="lfsr",
        lfsr_polynomial="x^16+x^14+x^13+x^11+1",
    )

    sampled = sample_sc_bitstreams(torch.tensor([0.6]), cfg, role="input")

    assert sampled.decoded.item() == pytest.approx(0.6, abs=0.04)


def test_lfsr_sampler_matches_native_core_engine_when_available():
    from sc_neurocore._native import core_engine_bridge

    if not core_engine_bridge.is_available():
        pytest.skip("native core_engine bridge is not available")

    cfg = _base_config(
        bitstream_length=192,
        encoding="unipolar",
        generator="lfsr",
        input_seed=0xACE1,
        weight_seed=0xBEEF,
        lfsr_polynomial="x^16+x^14+x^13+x^11+1",
    )
    values = torch.tensor([0.25, 0.75])

    sampled = sample_sc_bitstreams(values, cfg, role="input")
    expected = torch.tensor(
        np.stack(
            [
                core_engine_bridge.lfsr_encode_bits(
                    seed=0xACE1,
                    threshold=round(0.25 * 0xFFFF),
                    bit_length=cfg.bitstream_length,
                ),
                core_engine_bridge.lfsr_encode_bits(
                    seed=(0xACE1 + 7_919) & 0xFFFF,
                    threshold=round(0.75 * 0xFFFF),
                    bit_length=cfg.bitstream_length,
                ),
            ]
        ),
        dtype=torch.float32,
    )

    assert torch.equal(sampled.bits, expected)


def test_lfsr_sampler_uses_native_core_engine_when_available(monkeypatch):
    from sc_neurocore._native import core_engine_bridge

    calls = []

    def fake_encode_bits(*, seed, threshold, bit_length):
        calls.append((seed, threshold, bit_length))
        return np.ones(bit_length, dtype=np.uint8)

    monkeypatch.setattr(core_engine_bridge, "is_available", lambda: True)
    monkeypatch.setattr(core_engine_bridge, "lfsr_encode_bits", fake_encode_bits)

    cfg = _base_config(
        bitstream_length=8,
        encoding="unipolar",
        generator="lfsr",
        input_seed=0xACE1,
        weight_seed=0xBEEF,
        lfsr_polynomial="x^16+x^14+x^13+x^11+1",
    )

    sampled = sample_sc_bitstreams(torch.tensor([0.5]), cfg, role="input")

    assert calls == [(0xACE1, round(0.5 * 0xFFFF), 8)]
    assert torch.equal(sampled.bits, torch.ones(1, 8))


def test_bitstream_statistics_expose_rate_variance_and_correlation():
    streams = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    stats = estimate_bitstream_statistics(streams)

    assert torch.allclose(stats.rate, torch.tensor([0.5, 0.5, 0.5]))
    assert stats.variance.shape == (3,)
    assert stats.correlation.shape == (3, 3)
    assert stats.max_abs_off_diagonal_correlation.item() == pytest.approx(1.0)


def test_sampled_bipolar_multiply_matches_relaxed_path_with_low_discrepancy_streams():
    cfg = _base_config(
        bitstream_length=1024,
        encoding="bipolar",
        generator="sobol",
        input_seed=13,
        weight_seed=97,
        correlation=0.0,
    )
    x = torch.tensor([0.25, -0.5, 0.75])
    w = torch.tensor([-0.4, 0.2, 0.5])

    sampled = sampled_sc_multiply(x, w, cfg)
    relaxed = relaxed_sc_multiply(x, w, cfg).value

    assert torch.allclose(sampled.value, relaxed, atol=0.08)
    assert sampled.input_statistics.rate.shape == x.shape
    assert sampled.weight_statistics.rate.shape == w.shape


def test_training_package_exports_sampler_contracts():
    import sc_neurocore.training as training

    assert training.SCBitstreamSample is not None
    assert training.SCBitstreamStatistics is not None
    assert training.SampledSCProduct is not None
    assert training.sample_sc_bitstreams is sample_sc_bitstreams
    assert training.sampled_sc_multiply is sampled_sc_multiply
    assert training.estimate_bitstream_statistics is estimate_bitstream_statistics
