use proptest::prelude::*;
use sc_neurocore_engine::layer::DenseLayer;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn forward_output_non_negative(
        n_inputs in 2usize..=32,
        n_neurons in 1usize..=16,
    ) {
        let layer = DenseLayer::new(n_inputs, n_neurons, 256, 42);
        let inputs = vec![0.5; n_inputs];
        let out = layer.forward(&inputs, 42).unwrap();
        prop_assert_eq!(out.len(), n_neurons);
        for &v in &out {
            prop_assert!(v >= 0.0, "Negative output: {}", v);
        }
    }

    #[test]
    fn forward_deterministic(seed in 1u64..=10000) {
        let layer = DenseLayer::new(8, 4, 512, 42);
        let inputs = vec![0.5; 8];
        let out1 = layer.forward(&inputs, seed).unwrap();
        let out2 = layer.forward(&inputs, seed).unwrap();
        prop_assert_eq!(out1, out2, "Same seed should give same output");
    }

    #[test]
    fn forward_zero_input_gives_near_zero(
        n_neurons in 1usize..=16,
    ) {
        let layer = DenseLayer::new(8, n_neurons, 1024, 42);
        let inputs = vec![0.0; 8];
        let out = layer.forward(&inputs, 42).unwrap();
        for &v in &out {
            prop_assert!(
                v < 0.02,
                "Expected near-zero output for zero input, got {}",
                v
            );
        }
    }
}
