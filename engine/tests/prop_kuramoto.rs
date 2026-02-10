use proptest::prelude::*;
use sc_neurocore_engine::scpn::KuramotoSolver;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn order_parameter_in_range(
        n in 2usize..=50,
        _seed in 0u64..=1000,
    ) {
        let omega = vec![1.0; n];
        let coupling = vec![0.0; n * n];
        let phases: Vec<f64> = (0..n)
            .map(|i| std::f64::consts::TAU * (i as f64) / (n as f64) * 0.99)
            .collect();
        let solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
        let r = solver.order_parameter();
        prop_assert!(
            (0.0..=1.0 + 1e-10).contains(&r),
            "R out of range: {}",
            r
        );
    }

    #[test]
    fn step_preserves_phase_range(
        n in 2usize..=30,
        dt in 0.001f64..=0.1,
    ) {
        let omega = vec![1.0; n];
        let coupling = vec![0.1; n * n];
        let phases: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
        let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
        solver.step(dt, 42);
        for &p in solver.get_phases() {
            prop_assert!(
                (0.0..std::f64::consts::TAU).contains(&p),
                "Phase out of [0, 2pi): {}",
                p
            );
        }
    }

    #[test]
    fn identical_phases_stay_coherent(
        n in 2usize..=50,
        coupling_strength in 0.0f64..=5.0,
    ) {
        let omega = vec![1.0; n];
        let coupling = vec![coupling_strength; n * n];
        let phases = vec![1.0; n];
        let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);

        for _ in 0..10 {
            solver.step(0.01, 0);
        }
        let r = solver.order_parameter();
        prop_assert!(r > 0.99, "Identical phases should stay coherent, R={}", r);
    }
}
