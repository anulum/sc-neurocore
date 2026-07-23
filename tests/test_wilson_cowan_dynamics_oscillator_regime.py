# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOscillatorRegime from former test_wilson_cowan_dynamics.py

"""Focused suite: TestOscillatorRegime from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403

class TestOscillatorRegime:
    """With strong recurrent coupling + cross-inhibition, Wilson-Cowan
    supports limit cycles. Detecting them via zero-crossings of
    (E - mean(E))."""

    # Oscillator parameter set derived from Wilson-Cowan 1972 Fig 3.
    # Requires the published two-term sigmoid (Hopf bifurcation).
    OSC_PARAMS = dict(
        w_ee=16.0,
        w_ei=12.0,
        w_ie=15.0,
        w_ii=3.0,
        tau_e=1.0,
        tau_i=2.0,
        a=1.2,
        theta=2.8,
        dt=0.1,
    )
    OSC_DRIVE = 1.25
    OSC_N_STEPS = 4_000
    OSC_TRANSIENT = 1_000

    def _run_oscillator(self) -> np.ndarray:
        u = WilsonCowanUnit(**self.OSC_PARAMS)
        trace_e = np.empty(self.OSC_N_STEPS, dtype=np.float64)
        for t in range(self.OSC_N_STEPS):
            u.step(self.OSC_DRIVE)
            trace_e[t] = u.e
        return trace_e[self.OSC_TRANSIENT :]

    def test_limit_cycle_shows_repeated_zero_crossings(self):
        """Canonical regression guard: with the corrected two-term sigmoid,
        the Wilson-Cowan 1972 Fig 3 oscillator parameters produce repeated
        E oscillations around the post-transient mean."""
        trace = self._run_oscillator()
        centred = trace - trace.mean()
        zero_crossings = int(np.sum(np.abs(np.diff(np.sign(centred))) > 0))
        assert zero_crossings >= 4, (
            f"Oscillator regime should show repeated E oscillations "
            f"around the mean; got {zero_crossings} zero-crossings"
        )

    def test_limit_cycle_fft_peak_in_published_band(self):
        """FFT peak verification. The Wilson-Cowan 1972 oscillator at this
        parameter set has a dominant period of roughly 60–140 timesteps
        (dt=0.1) corresponding to ~0.7–1.6 Hz in time-units of the Euler
        step. This is the actual published-frequency check requested by
        `feedback_no_fabricated_benchmarks`."""
        trace = self._run_oscillator()
        centred = trace - trace.mean()
        # Discrete FFT of the detrended trace; bin 0 (DC) is now near 0.
        spec = np.abs(np.fft.rfft(centred))
        freqs = np.fft.rfftfreq(len(centred), d=self.OSC_PARAMS["dt"])
        peak_bin = int(np.argmax(spec[1:])) + 1  # skip DC
        peak_freq = float(freqs[peak_bin])
        peak_period_steps = 1.0 / (peak_freq * self.OSC_PARAMS["dt"])
        assert 40.0 <= peak_period_steps <= 200.0, (
            f"Oscillator dominant period {peak_period_steps:.1f} steps "
            f"(f={peak_freq:.3f} Hz) outside the expected "
            f"Wilson-Cowan 1972 Fig 3 range 40–200 steps"
        )
        # Peak must be significantly above the noise floor: at least 5× the
        # median non-DC bin.
        noise_floor = float(np.median(spec[1:]))
        assert spec[peak_bin] > 5.0 * noise_floor, (
            f"FFT peak ({spec[peak_bin]:.2f}) should stand out against "
            f"noise floor ({noise_floor:.2f}); ratio={spec[peak_bin] / noise_floor:.2f}"
        )

    def test_limit_cycle_amplitude_nontrivial(self):
        """Amplitude of the oscillation must exceed pure noise."""
        trace = self._run_oscillator()
        amplitude = float(trace.max() - trace.min())
        assert amplitude > 0.1, (
            f"Limit-cycle amplitude should be well above numerical noise; "
            f"got E range {amplitude:.4f}"
        )
