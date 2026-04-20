# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/demo_sleep_optimization

module DemoSleepOptimizationAccel

using Statistics, LinearAlgebra

function generate_eeg_epoch(stage, n_samples, sample_rate, rng)
    stage: SleepStage,
    n_samples: int = 256,
    sample_rate: int = 256,
    rng: np.random.Generator | nothing = nothing,
    ) -> np.ndarray[Any, Any]
    if rng is nothing
        rng = np.random.default_rng()
    pri_hz, sec_hz, noise_scale = _STAGE_FREQ[stage]
    t = collect(n_samples) / sample_rate
    signal = (
        1.0 * sin(2.0 * pi * pri_hz * t)
        + 0.4 * sin(2.0 * pi * sec_hz * t)
        + noise_scale * rng.standard_normal(n_samples)
    )
    return signal
end

function run_demo()
    rng = np.random.default_rng(42)
    protocol = get_protocol("insomnia_relief")
    config = SleepOptimizerConfig(
        sample_rate=256,
        fft_window=512,
        stage_check_interval=256,
        max_reinduction_attempts=3,
    )
    optimizer = SleepOptimizer(protocol, config)
    optimizer.start_session()
    n_epochs = 100
    schedule = _night_schedule(n_epochs)
    # header
    print("=" * 88)
    print("  SLEEP OPTIMISATION DEMO -- insomnia_relief protocol, 100 epochs")
    print("=" * 88)
    print(
        f"{'Epoch':>5}  {'Elapsed':>8}  {'Stage':>5}  {'Target':>6}  "
        f"{'Match':>5}  {'Binaural':>8}  {'Noise':>6}  {'Reind':>5}"
    )
    print("-" * 88)
    for epoch_idx, true_stage in enumerate(schedule)
        eeg = generate_eeg_epoch(true_stage, n_samples=256, sample_rate=256, rng=rng)
        optimizer.add_samples(eeg)
        tick = optimizer.check_and_adapt()
        if tick is ! nothing && epoch_idx % 5 == 0
            print(
                f"{tick.tick:5d}  "
                f"{tick.elapsed_min:7.2f}m  "
                f"{tick.current_stage.name:>5}  "
                f"{tick.target_stage.name:>6}  "
                f"{'  Y' if tick.stage_match else '  N':>5}  "
                f"{tick.audio_params.binaural_hz:7.1f}Hz  "
                f"{tick.audio_params.noise_color:>6}  "
                f"{'YES' if tick.reinduction_active else '  -':>5}"
            )
    optimizer.stop_session()
    # --- morning report ---
    report = SleepReportGenerator.generate(optimizer)
    print()
    print("=" * 88)
    print("  MORNING SLEEP REPORT")
    print("=" * 88)
    print(f"  Total duration   : {report.total_duration_min:.1f} min")
    print(f"  Sleep onset      : {report.sleep_onset_latency_min:.1f} min")
    print(f"  Sleep efficiency : {report.sleep_efficiency_pct:.1f}%")
    print(f"  Quality score    : {report.quality_score:.1f}/100")
    print(f"  Grade            : {report.grade}")
    print(f"  Wakeups          : {report.wakeups}")
    print(f"  Reinductions     : {report.reinductions}")
    print()
    print("  Stage Breakdown:")
    print(f"  {'Stage':<8} {'Actual':>8} {'Target':>8} {'Duration':>10}")
    print("  " + "-" * 38)
    for stage_name in ["WAKE", "N1", "N2", "N3", "REM"]
        actual = report.stage_percentages.get(stage_name, 0.0)
        target = report.stage_targets.get(stage_name, 0.0)
        dur = report.stage_durations_min.get(stage_name, 0.0)
        print(f"  {stage_name:<8} {actual:7.1f}% {target:7.1f}% {dur:9.2f}m")
    print()
    print("  Recommendations:")
    for i, rec in enumerate(report.recommendations, 1)
        print(f"  {i}. {rec}")
    print()
    print("=" * 88)
    print("  Demo complete.")
    print("=" * 88)
end

end # module DemoSleepOptimizationAccel
