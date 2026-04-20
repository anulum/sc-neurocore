# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/demo_tcbo_consciousness

module DemoTcboConsciousnessAccel

using Statistics, LinearAlgebra

function run_demo()
    engine = TCBODemoEngine(N=16, dt=0.001, seed=42)
    print("=" * 72)
    print("  TCBO Consciousness Detection Demo")
    print("  16-layer Kuramoto → persistent homology proxy → consciousness gate")
    print("=" * 72)
    for name in ScenarioName
        cfg = SCENARIOS[name]
        print(f"\n{'─' * 72}")
        print(f"  Scenario: {name.value}")
        print(f"  {cfg.description}")
        print(f"  Duration: {cfg.duration_s}s | K_scale: {cfg.K_scale}")
        print(f"{'─' * 72}")
        snapshots = engine.run_scenario(
            name.value,
            duration_s=min(cfg.duration_s, 5.0),
            subsample=500,
        )
        print(f"  {'Step':>6}  {'p_h1':>7}  {'R':>7}  {'kappa':>7}  {'Gate':>6}")
        for s in snapshots
            gate_str = " OPEN" if s.gate_open else "CLOSE"
            bar = "█" * int(s.p_h1 * 20) + "░" * (20 - int(s.p_h1 * 20))
            print(
                f"  {s.step:>6}  {s.p_h1:>7.3f}  {s.R_global:>7.3f}  "
                f"{s.kappa:>7.3f}  {gate_str}  |{bar}|"
            )
        final = snapshots[-1] if snapshots else nothing
        if final
            print(
                f"\n  Final: p_h1={final.p_h1:.3f}, R={final.R_global:.3f}, "
                f"gate={'OPEN' if final.gate_open else 'CLOSED'}"
            )
    print(f"\n{'=' * 72}")
    print("  Demo complete.")
    print("=" * 72)
end

end # module DemoTcboConsciousnessAccel
