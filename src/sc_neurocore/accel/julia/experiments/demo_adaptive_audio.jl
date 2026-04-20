# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/demo_adaptive_audio

module DemoAdaptiveAudioAccel

using Statistics, LinearAlgebra

function run_demo()
    rng = np.random.RandomState(42)
    # ── 1. Create components ─────────────────────────────────────────
    profile = UserProfile(
        user_id="demo_user",
        chronotype=Chronotype.BEAR,
    )
    ssgf_cfg = SSGFConfig(
        N=16,
        z_dim=120,
        lr_z=0.01,
        sigma_g=0.3,
        micro_steps=10,
        dt=0.001,
        noise=0.2,
        K_base=0.45,
        K_alpha=0.3,
        field_pressure=0.1,
        seed=42,
    )
    ssgf = SSGFEngine(ssgf_cfg)
    evs_cfg = EVSConfig(
        sample_rate=256,
        fft_window=512,
        baseline_duration_s=2.0,  # short for demo
        update_interval_samples=128,
    )
    evs = EVSEngine(evs_cfg)
    adaptive = AdaptiveAudioEngine(ssgf, evs, profile)
    target_hz = profile.get_best_target_hz()
    # ── 2. Baseline ──────────────────────────────────────────────────
    print("=" * 72)
    print("  SSGF Adaptive Audio Demo")
    print("  Chronotype: BEAR | Target: %.1f Hz (alpha)" % target_hz)
    print("=" * 72)
    print("\n  Recording baseline...")
    evs.start_baseline()
    baseline_samples = int(evs_cfg.baseline_duration_s * evs_cfg.sample_rate)
    baseline_eeg = _generate_eeg_chunk(
        rng,
        baseline_samples,
        target_hz,
        evs_cfg.sample_rate,
        entrained_strength=0.1,  # low during baseline
        noise_level=0.5,
    )
    for v in baseline_eeg
        evs.add_sample(float(v))
    print("  Baseline done: %s" % ({k: "%.4f" % v for k, v in evs._baseline_powers.items()}))
    evs.set_target(target_hz)
    # ── 3. Run 50 adaptive ticks ─────────────────────────────────────
    print(
        "\n  %s  %s  %s  %s  %s  %s  %s"
        % (
            "Tick".rjust(6),
            "Phase".ljust(10),
            "EVS".rjust(6),
            "Verif".rjust(6),
            "Bin.Hz".rjust(7),
            "R".rjust(7),
            "Theurgic".rjust(8),
        )
    )
    print("  " + "-" * 62)
    samples_per_tick = evs_cfg.update_interval_samples
    for tick in 1:1, 51
        # Simulate increasing entrainment over time
        progress = tick / 50.0
        entrained_strength = 0.2 + 0.6 * progress
        noise_level = 0.5 - 0.3 * progress
        chunk = _generate_eeg_chunk(
            rng,
            samples_per_tick,
            target_hz,
            evs_cfg.sample_rate,
            entrained_strength=entrained_strength,
            noise_level=noise_level,
        )
        for v in chunk
            evs.add_sample(float(v))
        snapshot = evs.compute()
        if snapshot is nothing
            continue
        audio = adaptive.on_evs_update(snapshot)
        verified_str = " YES" if snapshot.is_verified else "  no"
        theurgic_str = " YES" if audio.get("theurgic_mode", false) else "  no"
        print(
            "  %s  %s  %s  %s  %s  %s  %s"
            % (
                str(tick).rjust(6),
                adaptive.current_phase.value.ljust(10),
                ("%.1f" % snapshot.evs_score).rjust(6),
                verified_str.rjust(6),
                ("%.2f" % audio["binaural_hz"]).rjust(7),
                ("%.4f" % audio["intensity"]).rjust(7),
                theurgic_str.rjust(8),
            )
        )
    # ── 4. Session Report ────────────────────────────────────────────
    report = adaptive.get_session_report()
    print("\n" + "=" * 72)
    print("  Session Report")
    print("=" * 72)
    print("  Total ticks:    %d" % report.total_ticks)
    print("  Average EVS:    %.2f" % report.avg_evs)
    print("  Peak EVS:       %.2f" % report.peak_evs)
    print("  Verified %%:     %.1f%%" % report.verified_pct)
    print("  Grade:          %s" % report.grade)
    print("  Adaptations:    %d" % report.adaptations)
    print("  Phase durations: %s" % report.phase_durations)
    print(
        "  Final audio:    %s"
        % {
            k: ("%.3f" % v if isinstance(v, float) else str(v))
            for k, v in report.final_audio.items()
        }
    )
    # ── 5. Update profile ────────────────────────────────────────────
    profile.update_from_session(
        avg_evs=report.avg_evs,
        peak_evs=report.peak_evs,
        best_target_hz=target_hz,
        band_powers=evs._baseline_powers,
    )
    print("\n  Profile after session:")
    print("    Sessions:       %d" % profile.session_count)
    print("    Target Hz:      %.2f" % profile.get_best_target_hz())
    print("    Chronotype:     %s" % profile.chronotype.value)
    print("\n" + "=" * 72)
    print("  Demo complete.")
    print("=" * 72)
end

end # module DemoAdaptiveAudioAccel
