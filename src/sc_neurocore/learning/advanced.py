# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Advanced plasticity: BPTT, eligibility traces, R-STDP, meta-learning, STP

"""Advanced plasticity rules beyond STDP.

Provides BPTT with surrogate gradients, three-factor eligibility traces,
reward-modulated STDP, MAML-style meta-learning, homeostatic scaling,
Tsodyks-Markram STP, and structural plasticity (synapse grow/prune).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# --- Surrogate gradient for BPTT ---

SURROGATE_BETA = 25.0  # steepness of fast-sigmoid surrogate


def _fast_sigmoid_surrogate(
    v: np.ndarray[Any, Any], threshold: float = 1.0
) -> np.ndarray[Any, Any]:
    """Surrogate gradient: d/dv of fast-sigmoid spike function.

    Neftci et al. 2019, Eq. 5.
    """
    return SURROGATE_BETA / (1.0 + SURROGATE_BETA * np.abs(v - threshold)) ** 2


def _direct_readout_projection(network: Any) -> Any:
    """Return the projection supported by the array-based learners.

    These lightweight learners operate on one directly connected input/output
    pair. Reject hidden, recurrent, and delayed graphs instead of silently
    computing a gradient for the wrong population or timestep.
    """
    if len(network.populations) != 2 or len(network.projections) != 1:
        raise NotImplementedError(
            "array-based BPTT learners require exactly two populations and one projection"
        )
    projection = network.projections[0]
    if (
        projection.source is not network.populations[0]
        or projection.target is not network.populations[1]
    ):
        raise NotImplementedError(
            "array-based BPTT learners require a direct input-to-output projection"
        )
    if getattr(projection, "_delay_mode", "none") != "none":
        raise NotImplementedError("array-based BPTT learners do not support delayed projections")
    return projection


def _validate_training_arrays(
    network: Any,
    inputs: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
) -> Any:
    """Validate a training batch and return its direct readout projection."""
    projection = _direct_readout_projection(network)
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must both be two-dimensional")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same number of timesteps")
    if inputs.shape[1] != projection.source.n:
        raise ValueError("input width must equal the input population size")
    if targets.shape[1] != projection.target.n:
        raise ValueError("target width must equal the output population size")
    if inputs.shape[0] == 0:
        raise ValueError("training sequences must contain at least one timestep")
    return projection


def _reset_training_network(network: Any) -> None:
    """Reset neuron state before an independent training rollout."""
    for population in network.populations:
        population.reset_all()


def _forward_direct_window(
    projection: Any,
    inputs: np.ndarray[Any, Any],
    previous_source_spikes: np.ndarray[Any, Any],
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    """Run a window with the same one-timestep projection semantics as ``Network``."""
    output_spikes: list[np.ndarray[Any, Any]] = []
    output_voltages: list[np.ndarray[Any, Any]] = []
    driving_source_spikes: list[np.ndarray[Any, Any]] = []
    source_spikes = previous_source_spikes.copy()

    for currents in inputs:
        driving_source_spikes.append(source_spikes.copy())
        output_current = projection.propagate(source_spikes)
        next_source_spikes = projection.source.step_all(currents)
        next_output_spikes = projection.target.step_all(output_current)
        output_spikes.append(next_output_spikes.copy())
        output_voltages.append(projection.target.voltages.copy())
        source_spikes = next_source_spikes

    return (
        np.stack(output_spikes),
        np.stack(output_voltages),
        np.stack(driving_source_spikes),
        source_spikes,
    )


def _direct_surrogate_gradient(
    projection: Any,
    output_spikes: np.ndarray[Any, Any],
    output_voltages: np.ndarray[Any, Any],
    driving_source_spikes: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Return the sparse direct-readout surrogate gradient for one window."""
    gradient: np.ndarray[Any, Any] = np.zeros_like(projection.data)
    output_error = output_spikes - targets
    for timestep in range(output_spikes.shape[0]):
        post_delta = output_error[timestep] * _fast_sigmoid_surrogate(output_voltages[timestep])
        for source_index in range(projection.source.n):
            pre = driving_source_spikes[timestep, source_index]
            if pre == 0:
                continue
            start = projection.indptr[source_index]
            stop = projection.indptr[source_index + 1]
            target_indices = projection.indices[start:stop]
            gradient[start:stop] += pre * post_delta[target_indices]
    return gradient


class BPTTLearner:
    """Backpropagation Through Time for spiking networks.

    Uses fast-sigmoid surrogate gradient (Neftci et al. 2019) to handle
    the spike non-differentiability.
    """

    def __init__(self, network: Any, loss_fn: Callable[..., float], lr: float = 1e-3) -> None:
        self.network = network
        self.loss_fn = loss_fn
        self.lr = lr

    def train_step(self, inputs: np.ndarray[Any, Any], targets: np.ndarray[Any, Any]) -> float:
        """One BPTT step: forward pass, loss, backward with surrogate gradients.

        Parameters
        ----------
        inputs : np.ndarray[Any, Any]
            Shape (n_steps, n_input) input currents.
        targets : np.ndarray[Any, Any]
            Shape (n_steps, n_output) target spike trains.

        Returns
        -------
        float
            Scalar loss value.
        """
        projection = _validate_training_arrays(self.network, inputs, targets)
        n_steps = inputs.shape[0]
        _reset_training_network(self.network)
        initial_spikes = np.zeros(projection.source.n, dtype=np.int8)
        spike_arr, recorded_v, driving_spikes, _ = _forward_direct_window(
            projection, inputs, initial_spikes
        )
        loss = float(self.loss_fn(spike_arr, targets))
        gradient = _direct_surrogate_gradient(
            projection, spike_arr, recorded_v, driving_spikes, targets
        )
        projection.data -= self.lr * gradient / n_steps

        return loss


class TBPTTLearner:
    """Truncated Backpropagation Through Time for long sequences.

    Splits input into chunks of ``k`` timesteps, backpropagating gradients
    only within each chunk while carrying forward state (membrane voltage)
    across boundaries. Reduces memory from O(T) to O(k).

    Williams & Peng 1990.
    """

    def __init__(
        self, network: Any, loss_fn: Callable[..., float], lr: float = 1e-3, k: int = 50
    ) -> None:
        self.network = network
        self.loss_fn = loss_fn
        self.lr = lr
        self.k = k

    def train_step(self, inputs: np.ndarray[Any, Any], targets: np.ndarray[Any, Any]) -> float:
        """One TBPTT step over the full sequence, chunked into windows of k.

        Parameters
        ----------
        inputs : np.ndarray[Any, Any]
            Shape (n_steps, n_input).
        targets : np.ndarray[Any, Any]
            Shape (n_steps, n_output).

        Returns
        -------
        float
            Total loss summed across chunks.
        """
        projection = _validate_training_arrays(self.network, inputs, targets)
        if self.k <= 0:
            raise ValueError("k must be positive")
        n_steps = inputs.shape[0]
        total_loss = 0.0
        _reset_training_network(self.network)
        previous_source_spikes = np.zeros(projection.source.n, dtype=np.int8)

        for chunk_start in range(0, n_steps, self.k):
            chunk_end = min(chunk_start + self.k, n_steps)
            chunk_len = chunk_end - chunk_start

            spike_arr, recorded_v, driving_spikes, previous_source_spikes = _forward_direct_window(
                projection,
                inputs[chunk_start:chunk_end],
                previous_source_spikes,
            )
            chunk_targets = targets[chunk_start:chunk_end]
            chunk_loss = float(self.loss_fn(spike_arr, chunk_targets))
            total_loss += chunk_loss

            # Backward within this chunk only
            gradient = _direct_surrogate_gradient(
                projection, spike_arr, recorded_v, driving_spikes, chunk_targets
            )
            projection.data -= self.lr * gradient / chunk_len

            # State (voltages) carries forward — no reset between chunks

        return total_loss


class EligibilityTrace:
    """E-prop eligibility trace: three-factor learning (pre x post x error).

    Bellec et al. 2020.
    """

    def __init__(self, tau_e: float = 20.0, dt: float = 1.0) -> None:
        self.decay = float(np.exp(-dt / tau_e))
        self._trace: np.ndarray[Any, Any] | None = None

    def update(
        self,
        pre_spike: np.ndarray[Any, Any],
        post_spike: np.ndarray[Any, Any],
        error_signal: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute weight delta from three-factor rule.

        Parameters
        ----------
        pre_spike, post_spike : np.ndarray[Any, Any]
            Binary (0/1) vectors of length n_pre, n_post.
        error_signal : np.ndarray[Any, Any]
            Error signal of length n_post.

        Returns
        -------
        np.ndarray[Any, Any]
            Weight delta matrix of shape (n_pre, n_post).
        """
        outer = np.outer(pre_spike, post_spike)
        if self._trace is None:
            self._trace = np.zeros_like(outer)
        self._trace = self.decay * self._trace + outer
        delta: np.ndarray[Any, Any] = self._trace * error_signal[np.newaxis, :]
        return delta


class RewardModulatedLearner:
    """Reward-modulated STDP (R-STDP).

    Maintains per-synapse eligibility traces and applies weight updates
    scaled by a global reward signal.
    """

    def __init__(self, network: Any, tau_reward: float = 100.0) -> None:
        self.network = network
        self.reward_decay = np.exp(-1.0 / tau_reward)
        self._elig: dict[int, np.ndarray[Any, Any]] = {}
        self._pre_trace: dict[int, np.ndarray[Any, Any]] = {}
        self._post_trace: dict[int, np.ndarray[Any, Any]] = {}
        self._init_traces()

    def _init_traces(self) -> None:
        for proj in self.network.projections:
            pid = id(proj)
            self._elig[pid] = np.zeros_like(proj.data)
            self._pre_trace[pid] = np.zeros(proj.source.n)
            self._post_trace[pid] = np.zeros(proj.target.n)

    def step(self, reward: float) -> None:
        """Apply reward-modulated weight update.

        Parameters
        ----------
        reward : float
            Scalar reward signal.
        """
        tau_trace = 20.0
        trace_decay = np.exp(-1.0 / tau_trace)
        for proj in self.network.projections:
            pid = id(proj)
            pre_sp = proj.source.voltages > 0.9
            post_sp = proj.target.voltages > 0.9
            self._pre_trace[pid] = trace_decay * self._pre_trace[pid] + pre_sp
            self._post_trace[pid] = trace_decay * self._post_trace[pid] + post_sp

            for i in range(proj.source.n):
                for k in range(proj.indptr[i], proj.indptr[i + 1]):
                    j = proj.indices[k]
                    self._elig[pid][k] = (
                        self.reward_decay * self._elig[pid][k]
                        + self._pre_trace[pid][i] * self._post_trace[pid][j]
                    )
            proj.data += 0.01 * reward * self._elig[pid]
            np.clip(proj.data, 0.0, None, out=proj.data)


class MetaLearner:
    """MAML-style meta-learning for spiking networks.

    Finn et al. 2017. Inner loop: fast adaptation on a task.
    Outer loop: meta-gradient across tasks.
    """

    def __init__(self, network: Any, inner_lr: float = 0.01, outer_lr: float = 0.001) -> None:
        self.network = network
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

    def _snapshot_weights(self) -> list[np.ndarray[Any, Any]]:
        return [proj.data.copy() for proj in self.network.projections]

    def _restore_weights(self, snapshot: list[np.ndarray[Any, Any]]) -> None:
        for proj, w in zip(self.network.projections, snapshot):
            proj.data[:] = w

    def inner_loop(
        self, task_data: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], n_steps: int = 5
    ) -> None:
        """Fast adaptation: n_steps of gradient descent on task_data.

        Parameters
        ----------
        task_data : tuple
            (inputs, targets) arrays.
        n_steps : int
            Number of inner-loop updates.
        """
        inputs, targets = task_data
        projection = _validate_training_arrays(self.network, inputs, targets)
        for _ in range(n_steps):
            _reset_training_network(self.network)
            n_t = inputs.shape[0]
            initial_spikes = np.zeros(projection.source.n, dtype=np.int8)
            spike_arr, recorded_v, driving_spikes, _ = _forward_direct_window(
                projection, inputs, initial_spikes
            )
            gradient = _direct_surrogate_gradient(
                projection, spike_arr, recorded_v, driving_spikes, targets
            )
            projection.data -= self.inner_lr * gradient / n_t

    def outer_step(self, tasks: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]) -> None:
        """Meta-gradient update across multiple tasks.

        Parameters
        ----------
        tasks : list of tuple
            Each element is (inputs, targets).
        """
        meta_grad = [np.zeros_like(proj.data) for proj in self.network.projections]
        base_weights = self._snapshot_weights()

        for task in tasks:
            self._restore_weights(base_weights)
            pre_weights = self._snapshot_weights()
            self.inner_loop(task)
            for idx, proj in enumerate(self.network.projections):
                meta_grad[idx] += proj.data - pre_weights[idx]

        self._restore_weights(base_weights)
        for idx, proj in enumerate(self.network.projections):
            proj.data += self.outer_lr * meta_grad[idx] / max(len(tasks), 1)


class HomeostaticPlasticity:
    """Homeostatic synaptic scaling to maintain target firing rate.

    Turrigiano 2008. Multiplicatively scales all incoming weights to keep
    the population mean rate near target_rate.
    """

    def __init__(self, target_rate: float = 10.0, tau: float = 1000.0) -> None:
        self.target_rate = target_rate
        self.tau = tau
        self._rate_estimate: float | None = None

    def update(self, population: Any) -> None:
        """Scale weights of all incoming projections to *population*.

        Parameters
        ----------
        population : Population
            Target population whose rate should be regulated.
        """
        current_rate = np.mean(population.voltages > 0.9) * 1000.0
        if self._rate_estimate is None:
            self._rate_estimate = current_rate
        alpha = 1.0 / self.tau
        self._rate_estimate += alpha * (current_rate - self._rate_estimate)
        if self._rate_estimate <= 0:
            return
        scale = self.target_rate / self._rate_estimate
        scale = np.clip(scale, 0.9, 1.1)
        for proj in getattr(population, "_projections", []):
            if hasattr(proj, "data"):
                proj.data *= scale
        self._last_scale = float(scale)


class ShortTermPlasticity:
    """Tsodyks-Markram short-term plasticity (STP).

    Tsodyks & Markram 1997. Models depression (tau_d) and facilitation (tau_f)
    with use parameter u_se.
    """

    def __init__(self, tau_d: float = 200.0, tau_f: float = 600.0, u_se: float = 0.2) -> None:
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.u_se = u_se
        self._x: np.ndarray[Any, Any] | None = None
        self._u: np.ndarray[Any, Any] | None = None

    def update(self, pre_spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute effective weight scaling given pre-synaptic spikes.

        Parameters
        ----------
        pre_spikes : np.ndarray[Any, Any]
            Binary (0/1) vector of length n_pre.

        Returns
        -------
        np.ndarray[Any, Any]
            Effective weight multiplier per pre-synaptic neuron.
        """
        n = pre_spikes.shape[0]
        if self._x is None:
            self._x = np.ones(n)
            self._u = np.full(n, self.u_se)
        assert self._x is not None and self._u is not None

        dt = 1.0
        self._x += dt / self.tau_d * (1.0 - self._x)
        self._u += dt / self.tau_f * (self.u_se - self._u)

        mask = pre_spikes.astype(bool)
        self._u[mask] += self.u_se * (1.0 - self._u[mask])
        release: np.ndarray[Any, Any] = self._u * self._x
        self._x[mask] -= release[mask]

        return release


class StructuralPlasticity:
    """Activity-dependent synapse creation and elimination.

    Grows new synapses between correlated neurons and prunes weak ones.
    """

    def __init__(self, growth_rate: float = 0.001, prune_threshold: float = 0.01) -> None:
        self.growth_rate = growth_rate
        self.prune_threshold = prune_threshold

    def update(self, projection: Any) -> None:
        """Grow or prune synapses in a Projection based on activity.

        Parameters
        ----------
        projection : Projection
            Target projection to modify.
        """
        prune_mask = np.abs(projection.data) < self.prune_threshold
        projection.data[prune_mask] = 0.0

        n_src = projection.source.n
        n_pruned = int(prune_mask.sum())
        n_grow = min(n_pruned, max(1, int(self.growth_rate * len(projection.data))))
        if n_grow > 0:
            zero_indices = np.where(projection.data == 0.0)[0]
            if zero_indices.size > 0:
                chosen = np.random.choice(
                    zero_indices, size=min(n_grow, zero_indices.size), replace=False
                )
                projection.data[chosen] = np.random.uniform(0.001, 0.05, size=chosen.size)
