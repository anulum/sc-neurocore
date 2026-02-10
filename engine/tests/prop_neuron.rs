use proptest::prelude::*;
use sc_neurocore_engine::neuron::FixedPointLif;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn lif_voltage_bounded(
        leak_k in 1i16..=100,
        gain_k in 1i16..=512,
        i_t in -500i16..=500i16,
        n_steps in 1usize..=500,
    ) {
        let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        for _ in 0..n_steps {
            let (spike, v) = lif.step(leak_k, gain_k, i_t, 0);
            prop_assert!(spike == 0 || spike == 1);
            prop_assert!((-32768..=32767).contains(&v));
        }
    }

    #[test]
    fn lif_zero_input_decays_to_rest(
        leak_k in 10i16..=100,
    ) {
        let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        for _ in 0..10 {
            lif.step(leak_k, 256, 200, 0);
        }
        for _ in 0..1000 {
            lif.step(leak_k, 256, 0, 0);
        }
        let (_, v) = lif.step(leak_k, 256, 0, 0);
        prop_assert!(v.abs() < 10, "Expected v near 0, got {}", v);
    }
}
