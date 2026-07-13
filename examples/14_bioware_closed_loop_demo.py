# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Experimental Bioware closed-loop demonstration

"""Run a deterministic synthetic demonstration of the experimental interface."""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sc_neurocore.bioware.bioware import (
    AERToSCConverter,
    BioAuditEntry,
    BioAuditLog,
    BioHybridSession,
    CultureHealth,
    HomeostaticPlasticity,
    LatencyBudget,
    MEAConfig,
    MEALayout,
    MEAToAERTranscoder,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
    detect_network_bursts,
    extract_lfp_power,
)
from sc_neurocore.arcane_zenith import ArcaneZenithCognitiveCore


class SyntheticMEA:
    """Realistic synthetic MEA generator for demos and tests."""

    def __init__(
        self,
        config: MEAConfig,
        active_fraction: float = 0.25,
        seed: int = 20260713,
    ):
        self.cfg = config
        self.active = int(config.num_channels * active_fraction)
        self.t_s = 0.0
        self.rng = np.random.default_rng(seed)

    def reset_time(self) -> None:
        """Reset the internal time counter."""
        self.t_s = 0.0

    def generate_frame(self, duration_s: float = 0.05) -> np.ndarray[Any, Any]:
        """Generate one synthetic multi-electrode voltage frame.

        Parameters
        ----------
        duration_s : float, optional
            Frame duration in seconds. The duration must contain more than 30
            samples at the configured MEA sampling rate.

        Returns
        -------
        numpy.ndarray
            Voltage samples in microvolts with shape ``(samples, channels)``.

        Raises
        ------
        ValueError
            If the requested frame is too short for the synthetic waveform.
        """
        n_samples = int(duration_s * self.cfg.sample_rate_hz)
        if n_samples <= 30:
            raise ValueError("duration_s must contain more than 30 MEA samples")
        voltage = self.rng.normal(
            0.0,
            self.cfg.noise_floor_uv,
            size=(n_samples, self.cfg.num_channels),
        )
        for ch in range(self.active):
            if self.rng.random() < 0.35:
                num_spikes = int(self.rng.integers(2, 6))
                idx = self.rng.integers(0, n_samples - 30, num_spikes)
                for i in idx:
                    voltage[i : i + 12, ch] += np.array(
                        [25, 60, 110, 55, -35, -75, -25, 5, 12, 8, 3, 0]
                    )
        self.t_s += duration_s
        return voltage


def run_bio_hybrid_demo() -> None:
    """Run and plot a deterministic synthetic MEA-to-optogenetic workflow."""
    print("=== SC-NeuroCore: experimental Bioware closed-loop demo ===\n")

    mea_cfg = MEAConfig.from_layout(MEALayout.MEA_60)
    mea_hardware = SyntheticMEA(mea_cfg)
    sorter = SpikeSorter(num_units=4, n_components=3)
    zenith_core = ArcaneZenithCognitiveCore(backend="torch")

    session = BioHybridSession(
        mea_config=mea_cfg,
        detector=SpikeDetector(mea_cfg),
        transcoder=MEAToAERTranscoder(),
        sc_converter=AERToSCConverter(
            window_ticks=0x10000,
            num_neurons=mea_cfg.num_channels,
        ),
        opto_encoder=SCToOptoEncoder(),
        health_monitor=CultureHealth(),
        sorter=sorter,
        pharm_model=PharmModel(onset_delay_s=0.5, gain=1.15),
        homeostatic=HomeostaticPlasticity(target_rate_hz=8.0),
        latency_budget=LatencyBudget(max_latency_us=10_000.0),
        zenith_core=zenith_core,
    )

    # Train sorter
    print("[1] Training SpikeSorter (PCA+KMeans)...")
    training_voltage = np.vstack([mea_hardware.generate_frame(0.05) for _ in range(8)])
    training_spikes = session.detector.detect(training_voltage)
    sorter.fit(training_spikes)
    print(f"   → Fitted on {len(training_spikes)} waveforms.")
    mea_hardware.reset_time()
    if session.pharm_model is not None:
        session.pharm_model.apply(0.0)

    print("\n[2] Starting 100-frame closed-loop experiment...")
    audit_log = BioAuditLog(experiment_id="DEMO-2026-CLOSED-LOOP")
    health_trace, drift_trace, lfp_gamma, bursts = [], [], [], []
    threshold_q88 = 256

    t_start = time.time()
    for frame_id in range(1, 101):
        frame_start_s = mea_hardware.t_s
        raw_voltage = mea_hardware.generate_frame(0.05)

        result = session.process_frame(
            voltage_data=raw_voltage,
            t_start_s=frame_start_s,
            stim_times_s=[0.025] if frame_id % 20 == 0 else None,
        )

        if session.homeostatic is not None:
            mean_rate_hz = result.num_spikes / (mea_cfg.num_channels * 0.05)
            threshold_q88 = session.homeostatic.update_threshold(
                threshold_q88,
                observed_rate_hz=mean_rate_hz,
                dt_ms=50.0,
            )

        health_trace.append(result.health["health_score"])
        drift_trace.append(zenith_core.neuron.identity_drift)

        # Additional diagnostic metrics
        lfp = extract_lfp_power(raw_voltage, mea_cfg.sample_rate_hz)
        lfp_gamma.append(float(np.mean(lfp["gamma"])))
        bursts.extend(detect_network_bursts(result.spikes))

        if frame_id % 20 == 0:
            print(
                f"  Frame {frame_id:03d} | Spikes: {result.num_spikes:3d} | "
                f"Opto: {result.num_opto_pulses} | Health: {result.health['health_score']:.3f} | "
                f"Drift: {zenith_core.neuron.identity_drift:.4f} | Latency: {result.latency_us:.1f} μs"
            )

        audit_log.log(
            BioAuditEntry(
                round_number=result.round,
                timestamp_iso="2026-04-20T00:00:00Z",
                num_spikes=result.num_spikes,
                num_opto_pulses=result.num_opto_pulses,
                latency_us=result.latency_us,
                health_score=result.health["health_score"],
            )
        )

    print(f"\n[3] Experiment complete in {time.time() - t_start:.2f}s.")
    print(f"    Final checksum: {audit_log.checksum()}")
    print(f"    Total network bursts detected: {len(bursts)}")
    print(f"    Final ArcaneZenith identity drift: {zenith_core.neuron.identity_drift:.4f}")
    print(f"    Caller-managed homeostatic threshold (Q8.8): {threshold_q88}")

    # Plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    axs[0].plot(health_trace, color="green", label="Culture Health Score")
    axs[0].set_title("Bioware Closed-Loop: Tissue Health")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend()

    axs[1].plot(drift_trace, color="purple", label="Cognitive Identity Drift")
    axs[1].set_title("ArcaneZenith: Self-Referential Plasticity")
    axs[1].legend()

    axs[2].plot(lfp_gamma, color="orange", label="Gamma-band LFP power")
    axs[2].set_title("LFP Dynamics")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_bio_hybrid_demo()
