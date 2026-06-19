# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware closed-loop demo (PCA SpikeSorter + SyntheticMEA + ArcaneZenith)

"""Closed-loop bio-hybrid experiment demo (production version)."""

import time
import numpy as np
import matplotlib.pyplot as plt

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
    extract_lfp_power,
    detect_network_bursts,
)
from sc_neurocore.arcane_zenith import ArcaneZenithCognitiveCore


class SyntheticMEA:
    """Realistic synthetic MEA generator for demos and tests."""

    def __init__(self, config: MEAConfig, active_fraction: float = 0.25):
        self.cfg = config
        self.active = int(config.num_channels * active_fraction)
        self.t_s = 0.0

    def reset_time(self) -> None:
        """Reset the internal time counter."""
        self.t_s = 0.0

    def generate_frame(self, duration_s: float = 0.1) -> np.ndarray:
        n_samples = int(duration_s * self.cfg.sample_rate_hz)
        voltage = np.random.randn(n_samples, self.cfg.num_channels) * self.cfg.noise_floor_uv
        for ch in range(self.active):
            if np.random.random() < 0.35:
                num_spikes = np.random.randint(2, 6)
                idx = np.random.randint(0, n_samples - 30, num_spikes)
                for i in idx:
                    voltage[i : i + 12, ch] += np.array(
                        [25, 60, 110, 55, -35, -75, -25, 5, 12, 8, 3, 0]
                    )
        self.t_s += duration_s
        return voltage


def run_bio_hybrid_demo() -> None:
    print("=== SC-NeuroCore: Bioware × ArcaneZenith Closed-Loop Demo (SOTA) ===\n")

    mea_cfg = MEAConfig.from_layout(MEALayout.MEA_60)
    mea_hardware = SyntheticMEA(mea_cfg)

    session = BioHybridSession(
        mea_config=mea_cfg,
        detector=SpikeDetector(mea_cfg),
        transcoder=MEAToAERTranscoder(),
        sc_converter=AERToSCConverter(num_neurons=mea_cfg.num_channels),
        opto_encoder=SCToOptoEncoder(),
        health_monitor=CultureHealth(),
        sorter=SpikeSorter(num_units=4, n_components=3),
        pharm_model=PharmModel(onset_delay_s=0.5, gain=1.15),
        homeostatic=HomeostaticPlasticity(target_rate_hz=8.0),
        latency_budget=LatencyBudget(max_latency_us=800.0),
        zenith_core=ArcaneZenithCognitiveCore(backend="torch"),
    )

    # Train sorter
    print("[1] Training SpikeSorter (PCA+KMeans)...")
    training_voltage = np.vstack([mea_hardware.generate_frame(0.2) for _ in range(8)])
    training_spikes = session.detector.detect(training_voltage)
    session.sorter.fit(training_spikes)
    print(f"   → Fitted on {len(training_spikes)} waveforms.")

    print("\n[2] Starting 100-frame closed-loop experiment...")
    audit_log = BioAuditLog(experiment_id="DEMO-2026-SOTA")
    health_trace, drift_trace, lfp_gamma, bursts = [], [], [], []

    t_start = time.time()
    for frame_id in range(1, 101):
        raw_voltage = mea_hardware.generate_frame(0.1)

        result = session.process_frame(
            voltage_data=raw_voltage,
            t_start_s=mea_hardware.t_s,
            stim_times_s=[0.05] if frame_id % 20 == 0 else None,  # occasional stim artifact
        )

        health_trace.append(result.health["health_score"])
        drift_trace.append(session.zenith_core.neuron.identity_drift)

        # Extra SOTA metrics
        lfp = extract_lfp_power(raw_voltage, mea_cfg.sample_rate_hz)
        lfp_gamma.append(float(np.mean(lfp["gamma"])))
        bursts.extend(detect_network_bursts(result.spikes))

        if frame_id % 20 == 0:
            print(
                f"  Frame {frame_id:03d} | Spikes: {result.num_spikes:3d} | "
                f"Opto: {result.num_opto_pulses} | Health: {result.health['health_score']:.3f} | "
                f"Drift: {session.zenith_core.neuron.identity_drift:.4f} | Latency: {result.latency_us:.1f} μs"
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
    print(f"    Final ArcaneZenith identity drift: {session.zenith_core.neuron.identity_drift:.4f}")

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
