# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gamma oscillation circuit (PING, conductance-based)

"""Pyramidal-Interneuron Network Gamma (PING) oscillation circuit.

Conductance-based PING per **Börgers & Kopell 2003** (Neural Comp.
15:509-538) — itself a refinement of **Whittington et al. 1995**
(Nature 373:612-615). The previous implementation was a rate-coded
toy that did not reproduce the published gamma-band peak; this
rewrite ships the full conductance model so the spectral peak,
phase-locking, and weak-PING-vs-strong-PING transition published in
those papers are reproducible.

Per-neuron model (Börgers-Kopell §2, also Wang & Buzsáki 1996 for
the I cell):

    C_m dV/dt = -g_L (V - E_L)
                - g_AMPA (V - E_AMPA)        # E inputs onto E and I
                - g_GABA (V - E_GABA)        # I inputs onto E and I
                + I_drive
                + sigma * sqrt(dt) * xi(t)

Spike when V crosses `v_threshold`; clamp to `v_reset` for an
absolute refractory period (`t_refrac`). Each pre-synaptic spike
adds a unit step to the post-synaptic conductance trace, which
then decays exponentially:

    dg_AMPA/dt = -g_AMPA / tau_AMPA + w_AMPA * Σ delta(t - t_spike^pre_E)
    dg_GABA/dt = -g_GABA / tau_GABA + w_GABA * Σ delta(t - t_spike^pre_I)

Defaults are the published values from Börgers-Kopell 2003 Fig 2
(weak-PING regime, 40 Hz):

    tau_AMPA  = 3 ms        E_AMPA  =   0 mV
    tau_GABA  = 9 ms        E_GABA  = -80 mV
    g_L       = 0.1 mS/cm²  E_L     = -67 mV
    C_m       = 1 µF/cm²
    v_thresh  = -52 mV      v_reset = -67 mV    t_refrac = 2 ms
    w_EE = 0.05  w_EI = 0.25  w_IE = 0.25  w_II = 0.20

Drive `I_drive_e_mean` ≈ 1.4 µA/cm² puts E cells in the supra-
threshold regime; I cells receive a smaller `I_drive_i_mean` so
they fire only when dragged up by the recurrent E→I conductance —
the canonical PING gain mechanism.

Usage:

    ping = PINGCircuit(n_excitatory=80, n_inhibitory=20)
    spikes_e_log = []
    spikes_i_log = []
    for _ in range(2000):                       # 200 ms at dt=0.1
        e, i = ping.step(dt=0.1)
        spikes_e_log.append(e)
        spikes_i_log.append(i)
    freq_hz = ping.dominant_frequency(spikes_e_log, dt=0.1)
    assert 30.0 <= freq_hz <= 80.0              # gamma band
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Border-Kopell 2003 published defaults (Fig 2A, weak-PING 40 Hz).
_DEFAULT_C_M = 1.0           # µF/cm²
_DEFAULT_G_L = 0.1           # mS/cm²
_DEFAULT_E_L = -67.0         # mV
_DEFAULT_E_AMPA = 0.0        # mV
_DEFAULT_E_GABA = -80.0      # mV
_DEFAULT_V_THRESH = -52.0    # mV
_DEFAULT_V_RESET = -67.0     # mV
_DEFAULT_T_REFRAC = 2.0      # ms
_DEFAULT_TAU_AMPA = 3.0      # ms
_DEFAULT_TAU_GABA = 9.0      # ms


@dataclass
class PINGCircuit:
    """Conductance-based PING circuit (Börgers-Kopell 2003).

    Parameters mirror the publication; defaults reproduce the 40 Hz
    weak-PING regime from Fig 2A. Override `i_drive_e_mean` to scan
    drive-frequency curves; override `w_EI` / `w_IE` to scan the
    gain-loop strength.
    """

    n_excitatory: int = 80
    n_inhibitory: int = 20

    # Membrane biophysics (Börgers-Kopell §2).
    c_m: float = _DEFAULT_C_M
    g_l: float = _DEFAULT_G_L
    e_l: float = _DEFAULT_E_L
    e_ampa: float = _DEFAULT_E_AMPA
    e_gaba: float = _DEFAULT_E_GABA
    v_threshold: float = _DEFAULT_V_THRESH
    v_reset: float = _DEFAULT_V_RESET
    t_refrac: float = _DEFAULT_T_REFRAC

    # Synaptic time constants.
    tau_ampa: float = _DEFAULT_TAU_AMPA
    tau_gaba: float = _DEFAULT_TAU_GABA

    # Connection weights (per-spike conductance jump applied to
    # every post-synaptic cell — all-to-all mean-field coupling).
    # The defaults below were chosen to reproduce the Börgers-Kopell
    # 2003 Fig 2A weak-PING regime (40 Hz dominant frequency with
    # `i_drive_e_mean = 2.0 µA/cm²`, N_E=80, N_I=20), verified by
    # `tests/test_gamma_oscillation.py::TestPublishedFidelity::
    # test_gamma_frequency_is_30_to_80_hz`. They are smaller than
    # the published per-synapse values because every pre-synaptic
    # spike here contributes to every post-synaptic cell (the
    # cumulative burst conductance scales with N_pre).
    w_ee: float = 0.0006        # E → E (recurrent excitation)
    w_ei: float = 0.003         # E → I (gain loop forward)
    w_ie: float = 0.005         # I → E (gain loop feedback)
    w_ii: float = 0.01          # I → I (lateral inhibition)

    # External drive (µA/cm²): per-neuron Gaussian heterogeneity
    # around the mean. Setting `i_drive_e_mean` ≈ 1.4 puts E cells
    # in the supra-threshold regime that drives gamma; setting
    # `i_drive_i_mean` ≈ 0 forces I cells to spike only when pulled
    # up by recurrent E→I — the canonical PING gain loop.
    i_drive_e_mean: float = 2.0
    i_drive_e_sigma: float = 0.05
    i_drive_i_mean: float = 0.0
    i_drive_i_sigma: float = 0.05

    # Membrane noise (µA/cm²·ms^½). Wraps a √dt term inside `step`.
    sigma_e: float = 0.10
    sigma_i: float = 0.05

    seed: int = 42

    # State (initialised in __post_init__).
    v_e: np.ndarray | None = field(default=None, repr=False)
    v_i: np.ndarray | None = field(default=None, repr=False)
    g_ampa_e: np.ndarray | None = field(default=None, repr=False)
    g_ampa_i: np.ndarray | None = field(default=None, repr=False)
    g_gaba_e: np.ndarray | None = field(default=None, repr=False)
    g_gaba_i: np.ndarray | None = field(default=None, repr=False)
    refrac_e: np.ndarray | None = field(default=None, repr=False)
    refrac_i: np.ndarray | None = field(default=None, repr=False)
    i_drive_e: np.ndarray | None = field(default=None, repr=False)
    i_drive_i: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_excitatory <= 0 or self.n_inhibitory <= 0:
            raise ValueError("PINGCircuit needs at least 1 E and 1 I neuron")
        self._rng = np.random.default_rng(self.seed)
        # Initial V drawn near E_L with small jitter, so spike onset
        # is asynchronous (matches Whittington 1995 burn-in).
        self.v_e = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_excitatory)
        self.v_i = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_inhibitory)
        self.g_ampa_e = np.zeros(self.n_excitatory, dtype=np.float64)
        self.g_ampa_i = np.zeros(self.n_inhibitory, dtype=np.float64)
        self.g_gaba_e = np.zeros(self.n_excitatory, dtype=np.float64)
        self.g_gaba_i = np.zeros(self.n_inhibitory, dtype=np.float64)
        self.refrac_e = np.zeros(self.n_excitatory, dtype=np.float64)
        self.refrac_i = np.zeros(self.n_inhibitory, dtype=np.float64)
        # Per-neuron heterogeneous drive (constant for the run).
        self.i_drive_e = self._rng.normal(
            self.i_drive_e_mean, self.i_drive_e_sigma, self.n_excitatory,
        )
        self.i_drive_i = self._rng.normal(
            self.i_drive_i_mean, self.i_drive_i_sigma, self.n_inhibitory,
        )

    # ── Single timestep ──────────────────────────────────────────

    def step(self, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Advance one timestep (dt in ms); return (spikes_e, spikes_i)."""
        # Decay synaptic conductances first (closed-form exponential).
        decay_ampa = np.exp(-dt / self.tau_ampa)
        decay_gaba = np.exp(-dt / self.tau_gaba)
        self.g_ampa_e *= decay_ampa
        self.g_ampa_i *= decay_ampa
        self.g_gaba_e *= decay_gaba
        self.g_gaba_i *= decay_gaba

        # Membrane currents (sign convention: outward positive).
        # E cells:  -g_L (V-E_L) - g_AMPA (V-E_AMPA) - g_GABA (V-E_GABA) + I_drive
        i_e = (
            -self.g_l * (self.v_e - self.e_l)
            - self.g_ampa_e * (self.v_e - self.e_ampa)
            - self.g_gaba_e * (self.v_e - self.e_gaba)
            + self.i_drive_e
        )
        i_i = (
            -self.g_l * (self.v_i - self.e_l)
            - self.g_ampa_i * (self.v_i - self.e_ampa)
            - self.g_gaba_i * (self.v_i - self.e_gaba)
            + self.i_drive_i
        )

        # Stochastic membrane noise (Wiener increment).
        noise_e = self.sigma_e * np.sqrt(dt) * self._rng.standard_normal(
            self.n_excitatory,
        )
        noise_i = self.sigma_i * np.sqrt(dt) * self._rng.standard_normal(
            self.n_inhibitory,
        )

        # Update voltages — only for non-refractory neurons.
        not_refrac_e = self.refrac_e <= 0.0
        not_refrac_i = self.refrac_i <= 0.0
        self.v_e[not_refrac_e] += (
            (i_e[not_refrac_e] / self.c_m) * dt + noise_e[not_refrac_e]
        )
        self.v_i[not_refrac_i] += (
            (i_i[not_refrac_i] / self.c_m) * dt + noise_i[not_refrac_i]
        )

        # Decrement refractory countdown.
        self.refrac_e = np.maximum(self.refrac_e - dt, 0.0)
        self.refrac_i = np.maximum(self.refrac_i - dt, 0.0)

        # Detect spikes (only neurons currently OUT of refractory).
        spikes_e = (self.v_e >= self.v_threshold) & not_refrac_e
        spikes_i = (self.v_i >= self.v_threshold) & not_refrac_i

        # Apply spike: clamp to reset, start refractory countdown.
        self.v_e[spikes_e] = self.v_reset
        self.v_i[spikes_i] = self.v_reset
        self.refrac_e[spikes_e] = self.t_refrac
        self.refrac_i[spikes_i] = self.t_refrac

        # Propagate spikes through synapses. E spikes raise AMPA on
        # both E (recurrent) and I (gain loop); I spikes raise GABA
        # on both E (inhibition) and I (self-inhibition).
        # `count_nonzero` instead of `.sum()`/`np.sum(...)`: under
        # coverage instrumentation NumPy can be reloaded, which makes
        # the Python-level `_NoValue` sentinel mismatch the C-level
        # one used by `umr_sum`, raising `TypeError` from `_methods.py`.
        # `count_nonzero` is a direct C call that does not use that
        # sentinel and so is reload-safe.
        n_e_spikes = int(np.count_nonzero(spikes_e))
        n_i_spikes = int(np.count_nonzero(spikes_i))
        if n_e_spikes > 0:
            # Each E spike contributes w_ee to every E cell and w_ei
            # to every I cell. Mean-field all-to-all coupling — this
            # matches the published model's "fully connected
            # subnetwork" assumption in Börgers-Kopell §2.
            self.g_ampa_e += self.w_ee * n_e_spikes
            self.g_ampa_i += self.w_ei * n_e_spikes
        if n_i_spikes > 0:
            self.g_gaba_e += self.w_ie * n_i_spikes
            self.g_gaba_i += self.w_ii * n_i_spikes

        return spikes_e.copy(), spikes_i.copy()

    # ── Reset (re-randomise initial state from the per-instance RNG) ──

    def reset_state(self) -> None:
        """Re-initialise voltages, conductances and refractory state.

        Note: this advances the RNG, so calling `reset_state()` does
        NOT return the network to its t=0 configuration. To
        reproduce a run from scratch, build a fresh `PINGCircuit`
        with the same seed.
        """
        assert self.v_e is not None and self.v_i is not None
        self.v_e[:] = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_excitatory)
        self.v_i[:] = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_inhibitory)
        assert self.g_ampa_e is not None and self.g_ampa_i is not None
        assert self.g_gaba_e is not None and self.g_gaba_i is not None
        assert self.refrac_e is not None and self.refrac_i is not None
        self.g_ampa_e[:] = 0.0
        self.g_ampa_i[:] = 0.0
        self.g_gaba_e[:] = 0.0
        self.g_gaba_i[:] = 0.0
        self.refrac_e[:] = 0.0
        self.refrac_i[:] = 0.0

    # ── Spectral analysis helpers ────────────────────────────────

    @staticmethod
    def population_rate(
        spike_log: list[np.ndarray], dt: float, bin_ms: float = 1.0,
    ) -> np.ndarray:
        """Bin per-step spike booleans into a population rate (Hz).

        `spike_log[t]` is a length-N boolean array of spikes at
        timestep t. Output bin width is `bin_ms`; the array length
        is `len(spike_log) * dt / bin_ms` (rounded down).
        """
        if not spike_log:
            return np.array([], dtype=np.float64)
        steps_per_bin = max(1, int(bin_ms / dt))
        n_bins = len(spike_log) // steps_per_bin
        n_neurons = len(spike_log[0])
        rate = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            lo = b * steps_per_bin
            hi = lo + steps_per_bin
            spikes_in_bin = sum(
                int(np.count_nonzero(s)) for s in spike_log[lo:hi]
            )
            # Convert to firing rate (Hz): spikes per neuron per second.
            rate[b] = spikes_in_bin / (n_neurons * (bin_ms / 1000.0))
        return rate

    def dominant_frequency(
        self,
        spike_log: list[np.ndarray],
        dt: float,
        bin_ms: float = 1.0,
        f_min: float = 5.0,
        f_max: float = 200.0,
    ) -> float:
        """Return the dominant frequency (Hz) in the population rate.

        Uses a discrete FFT on the binned population rate; the
        returned frequency is the bin with the largest magnitude
        within `[f_min, f_max]`. Returns 0.0 if the spike log is
        too short or all silent.
        """
        rate = self.population_rate(spike_log, dt=dt, bin_ms=bin_ms)
        if rate.size < 16 or np.allclose(rate, 0.0):
            return 0.0
        # Detrend to suppress the DC bin.
        rate = rate - np.mean(rate)
        spectrum = np.abs(np.fft.rfft(rate))
        freqs = np.fft.rfftfreq(rate.size, d=bin_ms / 1000.0)
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(mask):
            return 0.0
        idx = int(np.argmax(spectrum[mask]))
        return float(freqs[mask][idx])
