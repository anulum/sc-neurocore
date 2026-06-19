# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Differentiable SNN layers with surrogate gradient support

"""Differentiable SNN layers with surrogate gradient support.

Train in float with PyTorch autograd, deploy to SC bitstreams via to_sc_weights().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Tuple, cast

import torch
import torch.nn as nn

from .surrogate import atan_surrogate

SurrogateFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SCWeightNoiseModel:
    """Deterministic export-time noise model for SC weight probabilities."""

    mode: str = "binomial"
    bitstream_length: int = 256
    seed: int = 0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"none", "binomial", "gaussian"}:
            raise ValueError(f"unsupported SC weight noise mode: {self.mode}")
        if self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bitstream_length": self.bitstream_length,
            "seed": self.seed,
            "sigma": self.sigma,
        }


SCWeightNoiseSpec = SCWeightNoiseModel | str | Mapping[str, Any] | None


def _coerce_sc_weight_noise_model(noise_model: SCWeightNoiseSpec) -> SCWeightNoiseModel | None:
    if noise_model is None:
        return None
    if isinstance(noise_model, SCWeightNoiseModel):
        return noise_model
    if isinstance(noise_model, str):
        return SCWeightNoiseModel(mode=noise_model)
    return SCWeightNoiseModel(**dict(noise_model))


def _normalise_sc_weight_tensor(weight: torch.Tensor, encoding: str = "unipolar") -> torch.Tensor:
    if encoding not in {"unipolar", "bipolar"}:
        raise ValueError("encoding must be 'unipolar' or 'bipolar'")
    if encoding == "bipolar":
        abs_max = weight.abs().max()
        if abs_max > 0:
            return weight / abs_max
        return torch.zeros_like(weight)

    w_min, w_max = weight.min(), weight.max()
    if w_max > w_min:
        return (weight - w_min) / (w_max - w_min)
    return torch.zeros_like(weight)


def _sc_weight_scale(weight: torch.Tensor, encoding: str = "unipolar") -> torch.Tensor:
    if encoding == "bipolar":
        abs_max = weight.abs().max()
        if abs_max > 0:
            return abs_max
        return torch.ones((), dtype=weight.dtype, device=weight.device)
    return torch.ones((), dtype=weight.dtype, device=weight.device)


def _normalise_sc_bias_tensor(
    bias: torch.Tensor,
    scale: torch.Tensor,
    encoding: str = "unipolar",
) -> torch.Tensor:
    if encoding == "bipolar":
        return bias.detach() / scale
    return bias.detach()


def _apply_sc_weight_noise(
    weight: torch.Tensor,
    model: SCWeightNoiseModel | None,
    layer_index: int,
    encoding: str = "unipolar",
) -> torch.Tensor:
    if model is None or model.mode == "none":
        return weight

    generator = torch.Generator(device=weight.device)
    generator.manual_seed(model.seed + layer_index)
    if encoding == "bipolar":
        probabilities = ((weight.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        probabilities = weight.clamp(0.0, 1.0)

    if model.mode == "binomial":
        samples = torch.rand(
            (*probabilities.shape, model.bitstream_length),
            generator=generator,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        noisy = (samples < probabilities.unsqueeze(-1)).to(probabilities.dtype).mean(dim=-1)
        return 2.0 * noisy - 1.0 if encoding == "bipolar" else noisy

    noise = torch.randn(
        probabilities.shape,
        generator=generator,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    noisy = (probabilities + noise * model.sigma).clamp(0.0, 1.0)
    return 2.0 * noisy - 1.0 if encoding == "bipolar" else noisy


class LIFCell(nn.Module):
    """Single-step Leaky Integrate-and-Fire with surrogate backward.

    v[t] = beta * v[t-1] + I[t]
    spike[t] = H(v[t] - threshold)
    v[t] -= spike[t] * threshold  (subtract reset)
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        if learn_beta:
            # Store log-odds so sigmoid maps to (0, 1)
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta

        if learn_threshold:
            # Store log so exp maps to (0, inf)
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold

        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return cast(
            torch.Tensor, self._beta_logit.sigmoid() if self._learn_beta else self._beta_val
        )

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_next = self.beta * v + current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, v_next


class IFCell(nn.Module):
    """Integrate-and-Fire (no leak, beta=1).

    Simplest spiking model: v[t] = v[t-1] + I[t], fire when v >= threshold.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_threshold: bool = False,
    ):
        super().__init__()
        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_next = v + current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, v_next


class SynapticCell(nn.Module):
    """Dual-exponential synaptic LIF. Two state variables: synapse current + membrane.

    i_syn[t] = alpha * i_syn[t-1] + I[t]
    v[t] = beta * v[t-1] + i_syn[t]
    """

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 0.8,
        threshold: float = 1.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        if learn_beta:
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta

        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return cast(
            torch.Tensor, self._beta_logit.sigmoid() if self._learn_beta else self._beta_val
        )

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(
        self,
        current: torch.Tensor,
        i_syn: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (spike, i_syn_next, v_next)."""
        i_syn_next = self.alpha * i_syn + current
        v_next = self.beta * v + i_syn_next
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, i_syn_next, v_next


class ALIFCell(nn.Module):
    """Adaptive LIF. Bellec et al. 2020.

    Threshold adapts based on recent spiking: theta[t] = theta_0 + beta_adapt * a[t]
    where a[t] = rho * a[t-1] + spike[t-1].
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        rho: float = 0.99,
        beta_adapt: float = 1.8,
        surrogate_fn: SurrogateFn = atan_surrogate,
    ):
        super().__init__()
        self.beta = beta
        self.threshold_0 = threshold
        self.rho = rho
        self.beta_adapt = beta_adapt
        self.surrogate_fn = surrogate_fn

    def forward(
        self,
        current: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (spike, v_next, a_next)."""
        v_next = self.beta * v + current
        theta = self.threshold_0 + self.beta_adapt * a
        spike = self.surrogate_fn(v_next - theta)
        v_next = v_next - spike.detach() * theta
        a_next = self.rho * a + spike.detach()
        return spike, v_next, a_next


class ExpIFCell(nn.Module):
    """Exponential Integrate-and-Fire. Fourcaud-Trocmé et al. 2003.

    v[t] = beta * v[t-1] + delta_T * exp((v[t-1] - v_rh) / delta_T) + I[t]
    Exponential term creates sharp upstroke near threshold.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        delta_t: float = 0.5,
        v_rh: float = 0.8,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        if learn_beta:
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta
        self.delta_t = delta_t
        self.v_rh = v_rh
        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return cast(
            torch.Tensor, self._beta_logit.sigmoid() if self._learn_beta else self._beta_val
        )

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        exp_term = self.delta_t * torch.exp(torch.clamp((v - self.v_rh) / self.delta_t, max=5.0))
        v_next = self.beta * v + exp_term + current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, v_next


class AdExCell(nn.Module):
    """Adaptive Exponential IF. Brette & Gerstner 2005.

    v[t] = beta * v[t-1] + delta_T * exp((v - v_rh) / delta_T) - w[t-1] + I[t]
    w[t] = rho * w[t-1] + a * (v[t-1] - v_rest) + b * spike[t]
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        delta_t: float = 0.5,
        v_rh: float = 0.8,
        a: float = 0.01,
        b: float = 0.1,
        rho: float = 0.99,
        v_rest: float = 0.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        if learn_beta:
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta
        self.delta_t = delta_t
        self.v_rh = v_rh
        self.a = a
        self.b = b
        self.rho = rho
        self.v_rest = v_rest
        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return cast(
            torch.Tensor, self._beta_logit.sigmoid() if self._learn_beta else self._beta_val
        )

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(
        self,
        current: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exp_term = self.delta_t * torch.exp(torch.clamp((v - self.v_rh) / self.delta_t, max=5.0))
        v_next = self.beta * v + exp_term - w + current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        w_next = self.rho * w + self.a * (v - self.v_rest) + self.b * spike.detach()
        return spike, v_next, w_next


class LapicqueCell(nn.Module):
    """Lapicque IF with membrane resistance. Lapicque 1907.

    tau * dv/dt = -(v - v_rest) + R * I
    Discretised: v[t] = (1 - dt/tau) * v[t-1] + (R * dt / tau) * I[t]
    """

    def __init__(
        self,
        tau: float = 20.0,
        r: float = 1.0,
        dt: float = 1.0,
        threshold: float = 1.0,
        v_rest: float = 0.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.decay = 1.0 - dt / tau
        self.gain = r * dt / tau
        self.v_rest = v_rest
        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_next = self.decay * (v - self.v_rest) + self.v_rest + self.gain * current
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * (self.threshold - self.v_rest)
        return spike, v_next


class AlphaCell(nn.Module):
    """Alpha synapse neuron. Rall 1967.

    Two-state alpha function: i_exc and i_inh with separate time constants.
    v[t] = beta * v[t-1] + i_exc[t] - i_inh[t]
    """

    def __init__(
        self,
        alpha_exc: float = 0.9,
        alpha_inh: float = 0.85,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.alpha_exc = alpha_exc
        self.alpha_inh = alpha_inh
        if learn_beta:
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta

        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return cast(
            torch.Tensor, self._beta_logit.sigmoid() if self._learn_beta else self._beta_val
        )

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(
        self,
        exc_current: torch.Tensor,
        inh_current: torch.Tensor,
        i_exc: torch.Tensor,
        i_inh: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i_exc_next = self.alpha_exc * i_exc + exc_current
        i_inh_next = self.alpha_inh * i_inh + inh_current
        v_next = self.beta * v + i_exc_next - i_inh_next
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, i_exc_next, i_inh_next, v_next


class SecondOrderLIFCell(nn.Module):
    """Second-order LIF with inertial term. Dayan & Abbott 2001.

    Adds a second state variable (acceleration) for smoother dynamics:
    a[t] = alpha * a[t-1] + I[t]
    v[t] = beta * v[t-1] + a[t]
    """

    def __init__(
        self,
        alpha: float = 0.95,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        if learn_beta:
            self._beta_logit = nn.Parameter(torch.tensor(_logit(beta)))
        else:
            self.register_buffer("_beta_val", torch.tensor(beta))
        self._learn_beta = learn_beta

        if learn_threshold:
            self._threshold_log = nn.Parameter(torch.tensor(float(threshold)).log())
        else:
            self.register_buffer("_threshold_val", torch.tensor(threshold))
        self._learn_threshold = learn_threshold
        self.surrogate_fn = surrogate_fn

    @property
    def beta(self) -> torch.Tensor:
        return cast(
            torch.Tensor, self._beta_logit.sigmoid() if self._learn_beta else self._beta_val
        )

    @property
    def threshold(self) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._threshold_log.exp() if self._learn_threshold else self._threshold_val,
        )

    def forward(
        self,
        current: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_next = self.alpha * a + current
        v_next = self.beta * v + a_next
        spike = self.surrogate_fn(v_next - self.threshold)
        v_next = v_next - spike.detach() * self.threshold
        return spike, a_next, v_next


class RecurrentLIFCell(nn.Module):
    """LIF with trainable recurrent weights."""

    def __init__(
        self,
        n_neurons: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.lif = LIFCell(beta, threshold, surrogate_fn, learn_beta, learn_threshold)
        self.recurrent = nn.Linear(n_neurons, n_neurons, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.5)

    def forward(
        self,
        current: torch.Tensor,
        v: torch.Tensor,
        spike_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lif.forward(current + self.recurrent(spike_prev), v)


class SpikingNet(nn.Module):
    """Multi-layer feedforward SNN for classification.

    Architecture: [Linear -> LIF] x (n_layers+1)
    Readout: spike count and membrane accumulation over T timesteps.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 2,
        beta: float = 0.9,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.n_output = n_output
        sizes = [n_input] + [n_hidden] * n_layers + [n_output]
        self.linears = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(
            LIFCell(
                beta=beta,
                surrogate_fn=surrogate_fn,
                learn_beta=learn_beta,
                learn_threshold=learn_threshold,
            )
            for _ in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, batch, n_input). Returns (spike_counts, membrane_acc)."""
        T, batch, _ = x.shape
        device = x.device
        n_cells = len(self.lifs)
        v = [
            torch.zeros(batch, cast(torch.nn.Linear, lin).out_features, device=device)
            for lin in self.linears
        ]

        spike_sum = torch.zeros(batch, self.n_output, device=device)
        mem_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            h = x[t]
            for i in range(n_cells):
                h = cast(torch.nn.Linear, self.linears[i])(h)
                spike, v[i] = cast(LIFCell, self.lifs[i])(h, v[i])
                h = spike
            spike_sum = spike_sum + spike
            mem_sum = mem_sum + v[-1]

        return spike_sum, mem_sum

    def to_sc_weights(
        self,
        include_bias: bool = True,
        noise_model: SCWeightNoiseSpec = None,
        encoding: str = "unipolar",
    ) -> List[dict[str, Any]]:
        """Export weight matrices for SC bitstream deployment.

        Parameters
        ----------
        include_bias : bool
            If True (default), include bias vectors in the output dicts.
        noise_model : SCWeightNoiseModel | str | Mapping | None
            Optional deterministic export-time SC noise model. ``"binomial"``
            samples Bernoulli bitstreams and stores realised probabilities;
            ``"gaussian"`` adds clamped probability noise with ``sigma``.
        encoding : {"unipolar", "bipolar"}
            ``"unipolar"`` preserves the legacy [0, 1] export. ``"bipolar"``
            preserves sign by normalising weights to [-1, 1] for XNOR SC.

        Returns
        -------
        List of dicts with keys "weight" and optionally "bias" (Tensor).
        """
        layers: list[dict[str, Any]] = []
        model = _coerce_sc_weight_noise_model(noise_model)
        for layer_index, lin in enumerate(self.linears):
            lin_typed = cast(torch.nn.Linear, lin)
            scale = _sc_weight_scale(lin_typed.weight.detach(), encoding=encoding)
            w = _normalise_sc_weight_tensor(lin_typed.weight.detach(), encoding=encoding)
            w = _apply_sc_weight_noise(w, model, layer_index, encoding=encoding)
            entry: dict[str, Any] = {"weight": w, "encoding": encoding}
            if encoding == "bipolar":
                entry["weight_scale"] = scale.detach()
            if model is not None:
                entry["noise_model"] = model.metadata()
            if include_bias and lin_typed.bias is not None:
                entry["bias"] = _normalise_sc_bias_tensor(lin_typed.bias, scale, encoding)
            layers.append(entry)
        return layers


class ConvSpikingNet(nn.Module):
    """Convolutional SNN for image classification.

    Conv2d(1,32,5)→LIF→AvgPool→Conv2d(32,64,5)→LIF→AvgPool→Flatten→Linear→LIF→Linear→LIF
    """

    def __init__(
        self,
        n_output: int = 10,
        beta: float = 0.9,
        surrogate_fn: SurrogateFn = atan_surrogate,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.n_output = n_output
        lif_kw: dict[str, Any] = dict(
            beta=beta,
            surrogate_fn=surrogate_fn,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.lif1 = LIFCell(**lif_kw)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.lif2 = LIFCell(**lif_kw)
        self.pool2 = nn.AvgPool2d(2)
        # MNIST: 28→conv5→24→pool2→12→conv5→8→pool2→4, so 64*4*4=1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.lif3 = LIFCell(**lif_kw)
        self.fc2 = nn.Linear(128, n_output)
        self.lif4 = LIFCell(**lif_kw)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (T, batch, 1, 28, 28). Returns (spike_counts, membrane_acc)."""
        T, batch = x.shape[:2]
        device = x.device

        v1 = torch.zeros(batch, 32, 24, 24, device=device)
        v2 = torch.zeros(batch, 64, 8, 8, device=device)
        v3 = torch.zeros(batch, 128, device=device)
        v4 = torch.zeros(batch, self.n_output, device=device)

        spike_sum = torch.zeros(batch, self.n_output, device=device)
        mem_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            h = self.conv1(x[t])
            spk, v1 = self.lif1(h, v1)
            h = self.pool1(spk)

            h = self.conv2(h)
            spk, v2 = self.lif2(h, v2)
            h = self.pool2(spk)

            h = h.flatten(1)
            h = self.fc1(h)
            spk, v3 = self.lif3(h, v3)

            h = self.fc2(spk)
            spk, v4 = self.lif4(h, v4)

            spike_sum = spike_sum + spk
            mem_sum = mem_sum + v4

        return spike_sum, mem_sum

    def to_sc_weights(
        self,
        include_bias: bool = True,
        noise_model: SCWeightNoiseSpec = None,
        encoding: str = "unipolar",
    ) -> List[dict[str, Any]]:
        """Export weight matrices for SC bitstream deployment.

        Returns list of dicts with "weight" and optionally "bias" keys,
        matching SpikingNet.to_sc_weights() format.
        """
        layers: list[dict[str, Any]] = []
        model = _coerce_sc_weight_noise_model(noise_model)
        for layer_index, mod in enumerate([self.conv1, self.conv2, self.fc1, self.fc2]):
            w = (
                mod.weight.detach().flatten(1)
                if isinstance(mod, nn.Conv2d)
                else cast(torch.Tensor, mod.weight).detach()
            )
            scale = _sc_weight_scale(w, encoding=encoding)
            w = _normalise_sc_weight_tensor(w, encoding=encoding)
            w = _apply_sc_weight_noise(w, model, layer_index, encoding=encoding)
            entry: dict[str, Any] = {"weight": w, "encoding": encoding}
            if encoding == "bipolar":
                entry["weight_scale"] = scale.detach()
            if model is not None:
                entry["noise_model"] = model.metadata()
            if include_bias and mod.bias is not None:
                entry["bias"] = _normalise_sc_bias_tensor(
                    cast(torch.Tensor, mod.bias), scale, encoding
                )
            layers.append(entry)
        return layers


def _logit(p: float) -> float:
    """Inverse sigmoid: logit(p) = log(p / (1 - p))."""
    import math

    p = max(min(p, 1.0 - 1e-7), 1e-7)
    return math.log(p / (1.0 - p))
