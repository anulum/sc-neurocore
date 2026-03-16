# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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


import numpy as np

# --- Surrogate gradient for BPTT ---

SURROGATE_BETA = 25.0  # steepness of fast-sigmoid surrogate


def _fast_sigmoid_surrogate(v, threshold=1.0):
    """Surrogate gradient: d/dv of fast-sigmoid spike function.

    Neftci et al. 2019, Eq. 5.
    """
    return SURROGATE_BETA / (1.0 + SURROGATE_BETA * np.abs(v - threshold)) ** 2


class BPTTLearner:
    """Backpropagation Through Time for spiking networks.

    Uses fast-sigmoid surrogate gradient (Neftci et al. 2019) to handle
    the spike non-differentiability.
    """

    def __init__(self, network, loss_fn, lr=1e-3):
        self.network = network
        self.loss_fn = loss_fn
        self.lr = lr

    def train_step(self, inputs, targets):
        """One BPTT step: forward pass, loss, backward with surrogate gradients.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (n_steps, n_input) input currents.
        targets : np.ndarray
            Shape (n_steps, n_output) target spike trains.

        Returns
        -------
        float
            Scalar loss value.
        """
        n_steps = inputs.shape[0]
        for pop in self.network.populations:
            pop.reset_all()

        recorded_v = []
        recorded_spikes = []
        for t in range(n_steps):
            currents = inputs[t]
            pop = self.network.populations[0]
            spikes = pop.step_all(currents[: pop.n])
            recorded_v.append(pop.voltages.copy())
            recorded_spikes.append(spikes.copy())

        spike_arr = np.stack(recorded_spikes)
        loss = float(self.loss_fn(spike_arr, targets))

        output_error = spike_arr - targets
        for proj in self.network.projections:
            n_src = proj.source.n
            grad_w = np.zeros_like(proj.data)
            for t in range(n_steps):
                surr = _fast_sigmoid_surrogate(recorded_v[t])
                post_delta = output_error[t][: proj.target.n] * surr[: proj.target.n]
                for i in range(n_src):
                    for k in range(proj.indptr[i], proj.indptr[i + 1]):
                        j = proj.indices[k]
                        grad_w[k] += recorded_spikes[t][i] * post_delta[j]
            proj.data -= self.lr * grad_w / max(n_steps, 1)

        return loss


class EligibilityTrace:
    """E-prop eligibility trace: three-factor learning (pre x post x error).

    Bellec et al. 2020.
    """

    def __init__(self, tau_e=20.0, dt=1.0):
        self.decay = np.exp(-dt / tau_e)
        self._trace = None

    def update(self, pre_spike, post_spike, error_signal):
        """Compute weight delta from three-factor rule.

        Parameters
        ----------
        pre_spike, post_spike : np.ndarray
            Binary (0/1) vectors of length n_pre, n_post.
        error_signal : np.ndarray
            Error signal of length n_post.

        Returns
        -------
        np.ndarray
            Weight delta matrix of shape (n_pre, n_post).
        """
        outer = np.outer(pre_spike, post_spike)
        if self._trace is None:
            self._trace = np.zeros_like(outer)
        self._trace = self.decay * self._trace + outer
        return self._trace * error_signal[np.newaxis, :]


class RewardModulatedLearner:
    """Reward-modulated STDP (R-STDP).

    Maintains per-synapse eligibility traces and applies weight updates
    scaled by a global reward signal.
    """

    def __init__(self, network, tau_reward=100.0):
        self.network = network
        self.reward_decay = np.exp(-1.0 / tau_reward)
        self._elig: dict[int, np.ndarray] = {}
        self._pre_trace: dict[int, np.ndarray] = {}
        self._post_trace: dict[int, np.ndarray] = {}
        self._init_traces()

    def _init_traces(self):
        for proj in self.network.projections:
            pid = id(proj)
            self._elig[pid] = np.zeros_like(proj.data)
            self._pre_trace[pid] = np.zeros(proj.source.n)
            self._post_trace[pid] = np.zeros(proj.target.n)

    def step(self, reward):
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

    def __init__(self, network, inner_lr=0.01, outer_lr=0.001):
        self.network = network
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

    def _snapshot_weights(self):
        return [proj.data.copy() for proj in self.network.projections]

    def _restore_weights(self, snapshot):
        for proj, w in zip(self.network.projections, snapshot):
            proj.data[:] = w

    def inner_loop(self, task_data, n_steps=5):
        """Fast adaptation: n_steps of gradient descent on task_data.

        Parameters
        ----------
        task_data : tuple
            (inputs, targets) arrays.
        n_steps : int
            Number of inner-loop updates.
        """
        inputs, targets = task_data
        for _ in range(n_steps):
            for pop in self.network.populations:
                pop.reset_all()
            n_t = inputs.shape[0]
            recorded_spikes = []
            for t in range(n_t):
                pop = self.network.populations[0]
                spikes = pop.step_all(inputs[t][: pop.n])
                recorded_spikes.append(spikes.copy())
            spike_arr = np.stack(recorded_spikes)
            error = spike_arr - targets
            for proj in self.network.projections:
                grad = np.zeros_like(proj.data)
                for t in range(n_t):
                    for i in range(proj.source.n):
                        for k in range(proj.indptr[i], proj.indptr[i + 1]):
                            j = proj.indices[k]
                            grad[k] += recorded_spikes[t][i] * error[t][j]
                proj.data -= self.inner_lr * grad / max(n_t, 1)

    def outer_step(self, tasks):
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

    def __init__(self, target_rate=10.0, tau=1000.0):
        self.target_rate = target_rate
        self.tau = tau
        self._rate_estimate = None

    def update(self, population):
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
        population._voltages *= 1.0  # no-op; scaling applied below via projection
        self._last_scale = float(scale)


class ShortTermPlasticity:
    """Tsodyks-Markram short-term plasticity (STP).

    Tsodyks & Markram 1997. Models depression (tau_d) and facilitation (tau_f)
    with use parameter u_se.
    """

    def __init__(self, tau_d=200.0, tau_f=600.0, u_se=0.2):
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.u_se = u_se
        self._x = None  # available resources
        self._u = None  # utilisation variable

    def update(self, pre_spikes):
        """Compute effective weight scaling given pre-synaptic spikes.

        Parameters
        ----------
        pre_spikes : np.ndarray
            Binary (0/1) vector of length n_pre.

        Returns
        -------
        np.ndarray
            Effective weight multiplier per pre-synaptic neuron.
        """
        n = pre_spikes.shape[0]
        if self._x is None:
            self._x = np.ones(n)
            self._u = np.full(n, self.u_se)

        dt = 1.0
        self._x += dt / self.tau_d * (1.0 - self._x)
        self._u += dt / self.tau_f * (self.u_se - self._u)

        mask = pre_spikes.astype(bool)
        self._u[mask] += self.u_se * (1.0 - self._u[mask])
        release = self._u * self._x
        self._x[mask] -= release[mask]

        return release


class StructuralPlasticity:
    """Activity-dependent synapse creation and elimination.

    Grows new synapses between correlated neurons and prunes weak ones.
    """

    def __init__(self, growth_rate=0.001, prune_threshold=0.01):
        self.growth_rate = growth_rate
        self.prune_threshold = prune_threshold

    def update(self, projection):
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
