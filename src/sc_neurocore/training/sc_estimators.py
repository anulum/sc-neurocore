# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Differentiable stochastic-computing estimators

"""Differentiable stochastic-computing estimators for SC-aware training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SCEncoding = Literal["unipolar", "bipolar"]
SCGenerator = Literal["bernoulli", "sobol", "halton", "lfsr", "low_discrepancy"]
SCEstimator = Literal["pathwise_relaxation", "straight_through", "score_function"]

SC_ENCODINGS: tuple[SCEncoding, ...] = ("unipolar", "bipolar")
SC_GENERATORS: tuple[SCGenerator, ...] = (
    "bernoulli",
    "sobol",
    "halton",
    "lfsr",
    "low_discrepancy",
)
SC_ESTIMATORS: tuple[SCEstimator, ...] = (
    "pathwise_relaxation",
    "straight_through",
    "score_function",
)
SUPPORTED_LFSR_POLYNOMIALS: tuple[str, ...] = (
    "x^16+x^14+x^13+x^11+1",
    "x^32+x^22+x^2+x+1",
)


@dataclass(frozen=True, slots=True)
class DifferentiableSCConfig:
    """Validated contract for differentiable SC training operators."""

    bitstream_length: int
    encoding: SCEncoding = "bipolar"
    generator: SCGenerator = "sobol"
    estimator: SCEstimator = "pathwise_relaxation"
    input_seed: int = 1
    weight_seed: int = 2
    correlation: float = 0.0
    lfsr_polynomial: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bitstream_length, int) or self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be a positive integer")
        if self.encoding not in SC_ENCODINGS:
            raise ValueError(f"encoding must be one of {SC_ENCODINGS}")
        if self.generator not in SC_GENERATORS:
            raise ValueError(f"generator must be one of {SC_GENERATORS}")
        if self.estimator not in SC_ESTIMATORS:
            raise ValueError(f"estimator must be one of {SC_ESTIMATORS}")
        if self.input_seed == self.weight_seed:
            raise ValueError("input_seed and weight_seed must not reuse the same seed")
        if not -1.0 <= float(self.correlation) <= 1.0:
            raise ValueError("correlation must be in the closed interval [-1, 1]")
        if self.generator == "lfsr":
            if self.lfsr_polynomial not in SUPPORTED_LFSR_POLYNOMIALS:
                raise ValueError(
                    "lfsr_polynomial must be one of "
                    f"{SUPPORTED_LFSR_POLYNOMIALS} for lfsr generator"
                )
        elif self.lfsr_polynomial is not None:
            raise ValueError("lfsr_polynomial is only supported with generator='lfsr'")


@dataclass(frozen=True, slots=True)
class RelaxedSCProduct:
    """Result bundle for a differentiable relaxed SC product."""

    value: torch.Tensor
    bitstream_length: int
    encoding: SCEncoding
    generator: SCGenerator
    estimator: SCEstimator

    @property
    def length_cost(self) -> float:
        return 1.0 / float(self.bitstream_length)


@dataclass(frozen=True, slots=True)
class SCBitstreamSample:
    """Sampled SC bitstreams with decoded values."""

    bits: torch.Tensor
    decoded: torch.Tensor
    probabilities: torch.Tensor
    bitstream_length: int
    encoding: SCEncoding
    generator: SCGenerator
    role: str


@dataclass(frozen=True, slots=True)
class SCBitstreamStatistics:
    """Rate, variance, and correlation evidence for sampled bitstreams."""

    rate: torch.Tensor
    variance: torch.Tensor
    correlation: torch.Tensor
    max_abs_off_diagonal_correlation: torch.Tensor


@dataclass(frozen=True, slots=True)
class SampledSCProduct:
    """Decoded sampled SC multiply result and stream statistics."""

    value: torch.Tensor
    input_sample: SCBitstreamSample
    weight_sample: SCBitstreamSample
    product_bits: torch.Tensor
    input_statistics: SCBitstreamStatistics
    weight_statistics: SCBitstreamStatistics


def _require_domain(name: str, value: torch.Tensor, lower: float, upper: float) -> None:
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((detached < lower).any() or (detached > upper).any()):
        raise ValueError(f"{name} must be in the {lower:g}..{upper:g} {name} domain")


def _bernoulli_product_expectation(
    p_input: torch.Tensor, p_weight: torch.Tensor, correlation: float
) -> torch.Tensor:
    independent = p_input * p_weight
    if correlation == 0.0:
        return independent

    variance_scale = torch.sqrt(
        torch.clamp(p_input * (1.0 - p_input) * p_weight * (1.0 - p_weight), min=0.0)
    )
    return torch.clamp(independent + correlation * variance_scale, min=0.0, max=1.0)


def _role_seed(config: DifferentiableSCConfig, role: str) -> int:
    if role == "input":
        return config.input_seed
    if role == "weight":
        return config.weight_seed
    raise ValueError("role must be 'input' or 'weight'")


def _probabilities_from_values(
    values: torch.Tensor,
    config: DifferentiableSCConfig,
) -> torch.Tensor:
    if config.encoding == "unipolar":
        _require_domain("unipolar values", values, 0.0, 1.0)
        return values
    _require_domain("bipolar values", values, -1.0, 1.0)
    return (values + 1.0) * 0.5


def _decode_probabilities(
    probabilities: torch.Tensor, config: DifferentiableSCConfig
) -> torch.Tensor:
    if config.encoding == "unipolar":
        return probabilities
    return probabilities * 2.0 - 1.0


def _sample_bernoulli_row(
    probability: float,
    length: int,
    seed: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed & 0x7FFF_FFFF)
    random_values = torch.rand(length, generator=generator, dtype=torch.float32)
    return (random_values < probability).to(dtype=torch.float32, device=device)


def _sample_sobol_row(
    probability: float,
    length: int,
    seed: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed & 0x7FFF_FFFF)
    samples = engine.draw(length).squeeze(-1)
    return (samples < probability).to(dtype=torch.float32, device=device)


def _radical_inverse_base2(index: int) -> float:
    inverse = 0.0
    fraction = 0.5
    while index > 0:
        inverse += fraction * (index & 1)
        index >>= 1
        fraction *= 0.5
    return inverse


def _sample_halton_row(
    probability: float,
    length: int,
    seed: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    offset = (seed & 0xFFFF) + 1
    samples = [_radical_inverse_base2(offset + index) for index in range(length)]
    sample_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
    return (sample_tensor < probability).to(dtype=torch.float32)


def _lfsr16_step(state: int) -> int:
    feedback = ((state >> 15) ^ (state >> 13) ^ (state >> 12) ^ (state >> 10)) & 0x1
    return ((state << 1) & 0xFFFF) | feedback


def _sample_lfsr_row(
    probability: float,
    length: int,
    seed: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    state = seed & 0xFFFF
    if state == 0:
        state = 1
    threshold = max(0, min(0xFFFF, int(round(probability * 0xFFFF))))

    from sc_neurocore._native import core_engine_bridge

    if core_engine_bridge.is_available():
        native_bits = core_engine_bridge.lfsr_encode_bits(
            seed=state,
            threshold=threshold,
            bit_length=length,
        )
        return torch.tensor(native_bits, dtype=torch.float32, device=device)

    bits: list[float] = []
    for _ in range(length):
        bits.append(1.0 if state < threshold else 0.0)
        state = _lfsr16_step(state)
        if state == 0:
            state = 1
    return torch.tensor(bits, dtype=torch.float32, device=device)


def _sample_row(
    probability: float,
    length: int,
    seed: int,
    generator: SCGenerator,
    *,
    device: torch.device,
) -> torch.Tensor:
    if generator == "bernoulli":
        return _sample_bernoulli_row(probability, length, seed, device=device)
    if generator in {"sobol", "low_discrepancy"}:
        return _sample_sobol_row(probability, length, seed, device=device)
    if generator == "halton":
        return _sample_halton_row(probability, length, seed, device=device)
    if generator == "lfsr":
        return _sample_lfsr_row(probability, length, seed, device=device)
    raise ValueError(f"unsupported generator: {generator}")


def _sample_paired_sobol_rows(
    input_probability: float,
    weight_probability: float,
    length: int,
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed & 0x7FFF_FFFF)
    samples = engine.draw(length)
    input_bits = (samples[:, 0] < input_probability).to(dtype=torch.float32, device=device)
    weight_bits = (samples[:, 1] < weight_probability).to(dtype=torch.float32, device=device)
    return input_bits, weight_bits


def sample_sc_bitstreams(
    values: torch.Tensor,
    config: DifferentiableSCConfig,
    *,
    role: str,
) -> SCBitstreamSample:
    """Sample deterministic SC bitstreams for training-time statistics."""

    probabilities = _probabilities_from_values(values, config)
    flat_probabilities = probabilities.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
    base_seed = _role_seed(config, role)
    rows = [
        _sample_row(
            float(probability.item()),
            config.bitstream_length,
            base_seed + row_index * 7_919,
            config.generator,
            device=values.device,
        )
        for row_index, probability in enumerate(flat_probabilities)
    ]
    bits = torch.stack(rows, dim=0).reshape(*values.shape, config.bitstream_length)
    sampled_probabilities = bits.mean(dim=-1)
    decoded = _decode_probabilities(sampled_probabilities, config)
    return SCBitstreamSample(
        bits=bits,
        decoded=decoded,
        probabilities=probabilities,
        bitstream_length=config.bitstream_length,
        encoding=config.encoding,
        generator=config.generator,
        role=role,
    )


def estimate_bitstream_statistics(
    streams: torch.Tensor, *, eps: float = 1e-8
) -> SCBitstreamStatistics:
    """Return rate, variance, and Pearson correlation evidence for bitstreams."""

    if streams.ndim < 2:
        raise ValueError("streams must have shape (..., bitstream_length)")
    if streams.shape[-1] < 2:
        raise ValueError("bitstream_length must be at least two")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not bool(torch.isfinite(streams.detach()).all()):
        raise ValueError("streams must contain only finite values")

    flat = streams.reshape(-1, streams.shape[-1]).to(dtype=torch.float32)
    rate = flat.mean(dim=-1).reshape(streams.shape[:-1])
    variance = flat.var(dim=-1, unbiased=False).reshape(streams.shape[:-1])

    centred = flat - flat.mean(dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(centred, ord=2, dim=1, keepdim=True).clamp_min(eps)
    normalised = centred / norm
    correlation = (normalised @ normalised.T).clamp(min=-1.0, max=1.0)
    indices = torch.arange(correlation.shape[0], device=correlation.device)
    correlation[indices, indices] = 1.0
    if correlation.shape[0] <= 1:
        max_off_diagonal = torch.zeros((), dtype=correlation.dtype, device=correlation.device)
    else:
        mask = ~torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)
        max_off_diagonal = correlation[mask].abs().max()

    return SCBitstreamStatistics(
        rate=rate,
        variance=variance,
        correlation=correlation,
        max_abs_off_diagonal_correlation=max_off_diagonal,
    )


def relaxed_sc_multiply(
    input_value: torch.Tensor,
    weight_value: torch.Tensor,
    config: DifferentiableSCConfig,
) -> RelaxedSCProduct:
    """Return differentiable expected SC multiplication under the config contract."""

    if input_value.shape != weight_value.shape:
        try:
            torch.broadcast_shapes(input_value.shape, weight_value.shape)
        except RuntimeError as exc:
            raise ValueError("input_value and weight_value must be broadcast-compatible") from exc

    correlation = float(config.correlation)
    if config.encoding == "unipolar":
        _require_domain("unipolar input_value", input_value, 0.0, 1.0)
        _require_domain("unipolar weight_value", weight_value, 0.0, 1.0)
        value = _bernoulli_product_expectation(input_value, weight_value, correlation)
    else:
        _require_domain("bipolar input_value", input_value, -1.0, 1.0)
        _require_domain("bipolar weight_value", weight_value, -1.0, 1.0)
        p_input = (input_value + 1.0) * 0.5
        p_weight = (weight_value + 1.0) * 0.5
        unipolar_product = _bernoulli_product_expectation(p_input, p_weight, correlation)
        value = 4.0 * unipolar_product - 2.0 * p_input - 2.0 * p_weight + 1.0

    return RelaxedSCProduct(
        value=value,
        bitstream_length=config.bitstream_length,
        encoding=config.encoding,
        generator=config.generator,
        estimator=config.estimator,
    )


def sampled_sc_multiply(
    input_value: torch.Tensor,
    weight_value: torch.Tensor,
    config: DifferentiableSCConfig,
) -> SampledSCProduct:
    """Sample SC bitstreams, multiply them, and decode the empirical product."""

    if input_value.shape != weight_value.shape:
        try:
            torch.broadcast_shapes(input_value.shape, weight_value.shape)
        except RuntimeError as exc:
            raise ValueError("input_value and weight_value must be broadcast-compatible") from exc

    if config.generator in {"sobol", "low_discrepancy"}:
        input_probabilities = _probabilities_from_values(input_value, config)
        weight_probabilities = _probabilities_from_values(weight_value, config)
        flat_input = input_probabilities.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        flat_weight = (
            weight_probabilities.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        )
        input_rows: list[torch.Tensor] = []
        weight_rows: list[torch.Tensor] = []
        base_seed = config.input_seed ^ (config.weight_seed << 8)
        for row_index, (input_probability, weight_probability) in enumerate(
            zip(flat_input, flat_weight, strict=True)
        ):
            input_bits, weight_bits = _sample_paired_sobol_rows(
                float(input_probability.item()),
                float(weight_probability.item()),
                config.bitstream_length,
                base_seed + row_index * 7_919,
                device=input_value.device,
            )
            input_rows.append(input_bits)
            weight_rows.append(weight_bits)
        input_bits = torch.stack(input_rows, dim=0).reshape(
            *input_value.shape, config.bitstream_length
        )
        weight_bits = torch.stack(weight_rows, dim=0).reshape(
            *weight_value.shape,
            config.bitstream_length,
        )
        input_sample = SCBitstreamSample(
            bits=input_bits,
            decoded=_decode_probabilities(input_bits.mean(dim=-1), config),
            probabilities=input_probabilities,
            bitstream_length=config.bitstream_length,
            encoding=config.encoding,
            generator=config.generator,
            role="input",
        )
        weight_sample = SCBitstreamSample(
            bits=weight_bits,
            decoded=_decode_probabilities(weight_bits.mean(dim=-1), config),
            probabilities=weight_probabilities,
            bitstream_length=config.bitstream_length,
            encoding=config.encoding,
            generator=config.generator,
            role="weight",
        )
    else:
        input_sample = sample_sc_bitstreams(input_value, config, role="input")
        weight_sample = sample_sc_bitstreams(weight_value, config, role="weight")
    if config.encoding == "unipolar":
        product_bits = input_sample.bits * weight_sample.bits
        product_probability = product_bits.mean(dim=-1)
        value = product_probability
    else:
        product_bits = (input_sample.bits == weight_sample.bits).to(dtype=torch.float32)
        product_probability = product_bits.mean(dim=-1)
        value = product_probability * 2.0 - 1.0

    return SampledSCProduct(
        value=value,
        input_sample=input_sample,
        weight_sample=weight_sample,
        product_bits=product_bits,
        input_statistics=estimate_bitstream_statistics(input_sample.bits),
        weight_statistics=estimate_bitstream_statistics(weight_sample.bits),
    )


def finite_difference_gradients(
    input_value: float,
    weight_value: float,
    config: DifferentiableSCConfig,
    *,
    epsilon: float = 1e-4,
) -> tuple[float, float]:
    """Central finite-difference gradients for deterministic relaxed SC operators."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    def evaluate(x_value: float, w_value: float) -> float:
        x = torch.tensor([x_value], dtype=torch.float64)
        w = torch.tensor([w_value], dtype=torch.float64)
        return float(relaxed_sc_multiply(x, w, config).value.sum().item())

    grad_input = (
        evaluate(input_value + epsilon, weight_value)
        - evaluate(input_value - epsilon, weight_value)
    ) / (2.0 * epsilon)
    grad_weight = (
        evaluate(input_value, weight_value + epsilon)
        - evaluate(input_value, weight_value - epsilon)
    ) / (2.0 * epsilon)
    return grad_input, grad_weight
