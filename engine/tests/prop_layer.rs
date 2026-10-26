// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense layer public contracts and property tests

//! Public dense construction, inference, refusal and recovery contracts.

use proptest::prelude::*;
use sc_neurocore_engine::layer::{DenseLayer, DenseLayerError};

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

/// Reject capacity overflow through the public fallible constructor.
#[test]
fn dense_capacity_refusal_keeps_construction_usable() {
    for dimensions in [
        (isize::MAX as usize, 1, 65),
        (1, isize::MAX as usize, 65),
        (usize::MAX, usize::MAX, 1),
        (64, 64, usize::MAX),
    ] {
        let error = DenseLayer::try_new(dimensions.0, dimensions.1, dimensions.2, 24301)
            .expect_err("overflowing dense capacity must be refused");
        assert_eq!(error, DenseLayerError::Allocation);
        assert_eq!(error.to_string(), "dense storage cannot be allocated");
    }
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    layer
        .try_set_weights(vec![vec![1.0, 0.0], vec![0.0, 1.0]])
        .expect("valid weights");
    assert_eq!(
        layer
            .try_forward_fast(&[1.0, 0.0], 42)
            .expect("valid forward"),
        vec![1.0, 0.0]
    );
}

/// Preserve both weight representations on row-count and row-width refusal.
#[test]
fn dense_weight_shape_refusal_is_atomic() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    let original = layer.try_get_weights().expect("weight copy");
    let packed = layer.packed_weights_flat().to_vec();
    let output = layer
        .try_forward_fast(&[0.25, 0.75], 42)
        .expect("valid forward");
    for (replacement, message) in [
        (vec![vec![1.0, 0.0]], "Expected 2 rows, got 1."),
        (
            vec![vec![1.0], vec![0.0, 1.0]],
            "Row 0 has length 1, expected 2.",
        ),
    ] {
        assert_eq!(
            layer.try_set_weights(replacement.clone()),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(layer.set_weights(replacement), Err(message.into()));
        assert_eq!(layer.try_get_weights().expect("weight copy"), original);
        assert_eq!(layer.packed_weights_flat(), packed);
        assert_eq!(
            layer
                .try_forward_fast(&[0.25, 0.75], 42)
                .expect("valid forward"),
            output
        );
    }
}

/// Refuse incorrect batch shapes before changing a caller-owned output buffer.
#[test]
fn dense_batch_refusal_preserves_caller_output() {
    let layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    let mut output = [17.0; 4];
    assert_eq!(
        layer.try_forward_batch_into(&[1.0], 2, 42, &mut output),
        Err(DenseLayerError::Invalid(
            "Expected 4 values (2×2), got 1.".into()
        ))
    );
    assert_eq!(output, [17.0; 4]);
    let mut short_output = [23.0; 3];
    assert_eq!(
        layer.try_forward_batch_into(&[1.0, 0.0, 0.0, 1.0], 2, 42, &mut short_output),
        Err(DenseLayerError::Invalid(
            "Expected output length 4 (2×2), got 3.".into()
        ))
    );
    assert_eq!(short_output, [23.0; 3]);
    layer
        .try_forward_batch_into(&[1.0, 0.0, 0.0, 1.0], 2, 42, &mut output)
        .expect("valid batch");
    assert_eq!(
        &output[..2],
        layer.try_forward_fast(&[1.0, 0.0], 42).expect("sample 0")
    );
    assert_eq!(
        &output[2..],
        layer
            .try_forward_fast(&[0.0, 1.0], 1_000_042)
            .expect("sample 1")
    );
}

/// Preserve batch overflow priority, caller storage and recovery after public geometry edits.
#[test]
fn dense_batch_overflow_refusals_preserve_output_and_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    let inputs = [1.0, 0.0, 0.0, 1.0];
    let before = layer
        .try_forward_batch(&inputs, 2, 42)
        .expect("valid batch");
    let weights = layer.get_weights();
    let packed = layer.packed_weights_flat().to_vec();
    let allocating_message = "Output size overflow when allocating n_samples * n_neurons.";
    assert_eq!(
        layer.try_forward_batch(&[], usize::MAX, 42),
        Err(DenseLayerError::Invalid(allocating_message.into()))
    );
    assert_eq!(
        layer.forward_batch(&[], usize::MAX, 42),
        Err(allocating_message.into())
    );
    let mut output = [17.0; 4];
    let input_message = "Input size overflow when validating n_samples * n_inputs.";
    assert_eq!(
        layer.try_forward_batch_into(&[], usize::MAX, 42, &mut output),
        Err(DenseLayerError::Invalid(input_message.into()))
    );
    assert_eq!(output, [17.0; 4]);
    assert_eq!(
        layer.forward_batch_into(&[], usize::MAX, 42, &mut output),
        Err(input_message.into())
    );
    assert_eq!(output, [17.0; 4]);
    layer.n_neurons = usize::MAX;
    let output_message = "Output size overflow when validating n_samples * n_neurons.";
    assert_eq!(
        layer.try_forward_batch_into(&inputs, 2, 42, &mut output),
        Err(DenseLayerError::Invalid(output_message.into()))
    );
    assert_eq!(output, [17.0; 4]);
    assert_eq!(
        layer.forward_batch_into(&inputs, 2, 42, &mut output),
        Err(output_message.into())
    );
    assert_eq!(output, [17.0; 4]);
    layer.n_neurons = 2;
    assert_eq!(layer.get_weights(), weights);
    assert_eq!(layer.packed_weights_flat(), packed);
    layer
        .try_forward_batch_into(&inputs, 2, 42, &mut output)
        .expect("healthy retry after dimension restoration");
    assert_eq!(output.as_slice(), before);
    assert_eq!(layer.forward_batch(&inputs, 2, 42), Ok(before));
}

/// Repack public weight edits and preserve packed storage across a documented panic.
#[test]
fn dense_convenience_weight_refresh_preserves_refusal_and_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    layer.weights = weights.clone();
    layer.refresh_packed_weights();
    assert_eq!(layer.forward_fast(&[1.0, 0.0], 42), Ok(vec![1.0, 0.0]));
    let packed = layer.packed_weights_flat().to_vec();
    layer.weights[0].pop();
    let malformed = layer.weights.clone();
    let refusal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        layer.refresh_packed_weights();
    }));
    assert!(refusal.is_err());
    assert_eq!(layer.weights, malformed);
    assert_eq!(layer.packed_weights_flat(), packed);
    layer.weights = weights.clone();
    layer.refresh_packed_weights();
    assert_eq!(layer.weights, weights);
    assert_eq!(layer.packed_weights_flat(), packed);
    assert_eq!(layer.forward_fast(&[1.0, 0.0], 42), Ok(vec![1.0, 0.0]));
}

/// Reject zero dimensions in the documented order through the public Rust API.
#[test]
fn dense_zero_dimensions_preserve_refusal_priority() {
    for (inputs, neurons, length, message) in [
        (0, 0, 0, "bitstream length must be > 0"),
        (0, 0, 1, "n_inputs must be > 0"),
        (1, 0, 1, "n_neurons must be > 0"),
    ] {
        match DenseLayer::try_new(inputs, neurons, length, 24301) {
            Err(DenseLayerError::Invalid(actual)) => assert_eq!(actual, message),
            _ => panic!("zero dimensions did not produce the documented refusal"),
        }
    }
    assert!(DenseLayer::try_new(1, 1, 1, 24301).is_ok());
}

/// Preserve documented convenience-constructor panics and subsequent construction.
#[test]
fn dense_convenience_constructor_panics_preserve_refusal_priority() {
    for (inputs, neurons, length, message) in [
        (0, 0, 0, "bitstream length must be > 0"),
        (0, 0, 1, "n_inputs must be > 0"),
        (1, 0, 1, "n_neurons must be > 0"),
        (
            1,
            usize::MAX,
            1,
            "dense storage cannot be allocated: Allocation",
        ),
    ] {
        let refused = std::panic::catch_unwind(|| DenseLayer::new(inputs, neurons, length, 24301));
        let payload = match refused {
            Err(payload) => payload,
            Ok(_) => panic!("invalid convenience construction succeeded"),
        };
        let actual = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert_eq!(actual, Some(message));
        assert_eq!(
            DenseLayer::new(1, 1, 1, 24301).forward_fast(&[0.0], 42),
            Ok(vec![0.0])
        );
    }
}

/// Preserve String-returning forward refusals without changing weights or caller output.
#[test]
fn dense_convenience_forward_refusals_preserve_healthy_retry() {
    let layer = DenseLayer::new(2, 3, 65, 24301);
    let input = [0.25, 0.75];
    let weights = layer.get_weights();
    let packed = layer.packed_weights_flat().to_vec();
    let before = layer.forward_fast(&input, 42).expect("valid input");
    let message = "Expected input of length 2, got 0.".to_string();
    assert_eq!(layer.forward(&[], 42), Err(message.clone()));
    assert_eq!(layer.forward_fast(&[], 42), Err(message.clone()));
    assert_eq!(layer.forward_fused(&[], 42), Err(message));
    assert_eq!(
        layer.forward_batch(&[], 1, 42),
        Err("Expected 2 values (1×2), got 0.".into())
    );
    let mut caller_output = [991.0];
    assert_eq!(
        layer.forward_batch_into(&input, 1, 42, &mut caller_output),
        Err("Expected output length 3 (1×3), got 1.".into())
    );
    assert_eq!(caller_output, [991.0]);
    assert_eq!(layer.get_weights(), weights);
    assert_eq!(layer.packed_weights_flat(), packed);
    assert_eq!(layer.forward_fast(&input, 42), Ok(before));
}

/// Reject incomplete packed storage while preserving weights and a healthy retry.
#[test]
fn dense_packed_extent_refusals_preserve_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    layer.set_weights(weights.clone()).expect("valid weights");
    let packed_weights = layer.packed_weights_flat().to_vec();
    for (packed, inputs, words, message) in [
        (vec![u64::MAX, 1], 1, 2, "Expected 2 packed inputs, got 1."),
        (
            vec![u64::MAX, 0],
            2,
            1,
            "Expected 2 words per input, got 1.",
        ),
        (
            vec![u64::MAX, 1, 0],
            2,
            2,
            "Flat buffer length 3 != n_inputs(2) * words(2).",
        ),
    ] {
        assert_eq!(
            layer.try_forward_prepacked_2d(&packed, inputs, words),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(
            layer.forward_prepacked_2d(&packed, inputs, words),
            Err(message.into())
        );
        assert_eq!(layer.get_weights(), weights);
        assert_eq!(layer.packed_weights_flat(), packed_weights);
        assert_eq!(
            layer.forward_prepacked_2d(&[u64::MAX, 1, 0, 0], 2, 2),
            Ok(vec![1.0, 0.0])
        );
        assert_eq!(
            layer.forward_prepacked(&[vec![u64::MAX, 1], vec![0, 0]]),
            Ok(vec![1.0, 0.0])
        );
    }
}

/// Refuse inconsistent public packing geometry and recover both forward protocols.
#[test]
fn dense_packing_geometry_refusals_preserve_weights_and_retry() {
    for inputs_count in [2, 128] {
        let mut layer = DenseLayer::try_new(inputs_count, 2, 65, 24301).expect("valid storage");
        let inputs = vec![0.5; inputs_count];
        let scalar = layer.try_forward(&inputs, 42).expect("valid scalar input");
        let fast = layer
            .try_forward_fast(&inputs, 42)
            .expect("valid SIMD input");
        let weights = layer.get_weights();
        let packed = layer.packed_weights_flat().to_vec();
        layer.words_per_input = 1;
        let message = "packed output length must equal ceil(input length / 64)";
        assert_eq!(
            layer.try_forward(&inputs, 42),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(layer.forward(&inputs, 42), Err(message.into()));
        assert_eq!(
            layer.try_forward_fast(&inputs, 42),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(layer.forward_fast(&inputs, 42), Err(message.into()));
        assert_eq!(
            layer.try_refresh_packed_weights(),
            Err(DenseLayerError::Invalid(message.into()))
        );
        let refusal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            layer.refresh_packed_weights();
        }));
        assert!(refusal.is_err());
        assert_eq!(layer.get_weights(), weights);
        assert_eq!(layer.packed_weights_flat(), packed);
        layer.words_per_input = 2;
        assert_eq!(layer.try_forward(&inputs, 42), Ok(scalar));
        assert_eq!(layer.try_forward_fast(&inputs, 42), Ok(fast));
        layer
            .try_refresh_packed_weights()
            .expect("healthy repacking");
        assert_eq!(layer.get_weights(), weights);
        assert_eq!(layer.packed_weights_flat(), packed);
    }
}

/// Preserve nested packed input, weights and inference across both shape refusals.
#[test]
fn dense_nested_packed_refusals_preserve_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).expect("valid storage");
    let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    layer.set_weights(weights.clone()).expect("valid weights");
    let packed = layer.packed_weights_flat().to_vec();
    for (inputs, message) in [
        (vec![vec![u64::MAX, 1]], "Expected 2 packed inputs, got 1."),
        (
            vec![vec![u64::MAX], vec![0, 0]],
            "Packed input 0 has 1 words, expected 2.",
        ),
    ] {
        let before = inputs.clone();
        assert_eq!(
            layer.try_forward_prepacked(&inputs),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(layer.forward_prepacked(&inputs), Err(message.into()));
        assert_eq!(inputs, before);
        assert_eq!(layer.get_weights(), weights);
        assert_eq!(layer.packed_weights_flat(), packed);
        assert_eq!(
            layer.forward_prepacked(&[vec![u64::MAX, 1], vec![0, 0]]),
            Ok(vec![1.0, 0.0])
        );
    }
}

/// Exercise Rust convenience and fused kernels against the independent reference.
#[test]
fn dense_rust_kernels_match_recorded_python_counts() {
    // Counts captured from the unchanged installed reference in
    // tests/fixtures/dense_layer_forward_golden.json (native 8ed41318d0cf8be0).
    type DenseGoldenCase<'a> = (usize, usize, usize, u64, u64, &'a [u64]);
    let cases: [DenseGoldenCase<'_>; 9] = [
        (4, 2, 1, 42, 42, &[1, 2]),
        (4, 2, 63, 24301, 44257, &[43, 85]),
        (4, 2, 64, 24301, 44257, &[44, 86]),
        (4, 2, 65, 24301, 44257, &[46, 84]),
        (
            4,
            2,
            130,
            18446744073709551615,
            18446744073709551614,
            &[74, 191],
        ),
        (
            16,
            8,
            65,
            24301,
            18446744073709551611,
            &[243, 277, 202, 222, 229, 194, 269, 214],
        ),
        (
            128,
            8,
            65,
            18446744073709551613,
            18446744073709551552,
            &[1940, 1833, 1809, 1853, 1915, 1845, 1870, 1871],
        ),
        (4, 2, 1024, 12345, 99999, &[715, 1455]),
        (4, 2, 1025, 42, 18446744073709551613, &[724, 1521]),
    ];
    for (inputs, neurons, length, weight_seed, seed, counts) in cases {
        let mut layer = DenseLayer::try_new(inputs, neurons, length, weight_seed).unwrap();
        let weights: Vec<Vec<f64>> = (0..neurons)
            .map(|row| {
                (0..inputs)
                    .map(|column| ((7 * row + 3 * column) % 17) as f64 / 16.0)
                    .collect()
            })
            .collect();
        layer.set_weights(weights).unwrap();
        let values: Vec<f64> = (0..inputs)
            .map(|input| ((5 * input + 1) % 19) as f64 / 20.0)
            .collect();
        let expected: Vec<f64> = counts
            .iter()
            .map(|count| *count as f64 * (1.0 / length as f64))
            .collect();
        assert_eq!(layer.forward_fast(&values, seed).unwrap(), expected);
        assert_eq!(layer.try_forward_fused(&values, seed).unwrap(), expected);
        assert_eq!(layer.forward_fused(&values, seed).unwrap(), expected);
        assert_eq!(layer.forward_numpy_inner(&values, seed).unwrap(), expected);
    }
}

/// Propagate per-sample encoding refusal without publishing a partial batch.
#[test]
fn dense_batch_encoding_refusals_preserve_output_and_retry() {
    for inputs_count in [2, 128] {
        let mut layer = DenseLayer::try_new(inputs_count, 2, 65, 24301).unwrap();
        let inputs = vec![0.5; 2 * inputs_count];
        let before = layer.try_forward_batch(&inputs, 2, 42).unwrap();
        let weights = layer.get_weights();
        let packed = layer.packed_weights_flat().to_vec();
        let mut output = [17.0; 4];
        layer.words_per_input = 1;
        let message = "packed output length must equal ceil(input length / 64)";
        assert_eq!(
            layer.try_forward_batch_into(&inputs, 2, 42, &mut output),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(output, [17.0; 4]);
        assert_eq!(
            layer.forward_batch_into(&inputs, 2, 42, &mut output),
            Err(message.into())
        );
        assert_eq!(output, [17.0; 4]);
        assert_eq!(
            layer.try_forward_batch(&inputs, 2, 42),
            Err(DenseLayerError::Invalid(message.into()))
        );
        assert_eq!(layer.forward_batch(&inputs, 2, 42), Err(message.into()));
        assert_eq!(layer.get_weights(), weights);
        assert_eq!(layer.packed_weights_flat(), packed);
        layer.words_per_input = 2;
        layer
            .try_forward_batch_into(&inputs, 2, 42, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), before);
        assert_eq!(layer.forward_batch(&inputs, 2, 42), Ok(before));
    }
}

/// Reject unrepresentable packing capacities before allocating or replacing weights.
#[test]
fn dense_packing_capacity_refusals_preserve_weights_and_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).unwrap();
    let inputs = [0.25, 0.75];
    let scalar = layer.try_forward(&inputs, 42).unwrap();
    let fast = layer.try_forward_fast(&inputs, 42).unwrap();
    let weights = layer.get_weights();
    let packed = layer.packed_weights_flat().to_vec();
    let replacement = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    for words in [usize::MAX, usize::MAX / 2] {
        layer.words_per_input = words;
        assert_eq!(
            layer.try_forward(&inputs, 42),
            Err(DenseLayerError::Allocation)
        );
        assert_eq!(
            layer.try_forward_fast(&inputs, 42),
            Err(DenseLayerError::Allocation)
        );
        assert_eq!(
            layer.forward(&inputs, 42),
            Err("dense storage cannot be allocated".into())
        );
        assert_eq!(
            layer.forward_fast(&inputs, 42),
            Err("dense storage cannot be allocated".into())
        );
        assert_eq!(
            layer.try_refresh_packed_weights(),
            Err(DenseLayerError::Allocation)
        );
        assert_eq!(
            layer.try_set_weights(replacement.clone()),
            Err(DenseLayerError::Allocation)
        );
        assert_eq!(layer.weights, weights);
        assert_eq!(layer.packed_weights_flat(), packed);
    }
    layer.words_per_input = isize::MAX as usize / 16;
    let words = layer.words_per_input.checked_mul(4).unwrap();
    assert!(words.checked_mul(std::mem::size_of::<u64>()).unwrap() > isize::MAX as usize);
    assert_eq!(
        layer.try_refresh_packed_weights(),
        Err(DenseLayerError::Allocation)
    );
    assert_eq!(
        layer.try_set_weights(replacement),
        Err(DenseLayerError::Allocation)
    );
    assert_eq!(layer.weights, weights);
    assert_eq!(layer.packed_weights_flat(), packed);
    layer.words_per_input = 2;
    assert_eq!(layer.try_forward(&inputs, 42), Ok(scalar));
    assert_eq!(layer.try_forward_fast(&inputs, 42), Ok(fast));
    layer.try_refresh_packed_weights().unwrap();
    assert_eq!(layer.weights, weights);
    assert_eq!(layer.packed_weights_flat(), packed);
}

/// Refuse impossible activation capacity before executing scalar or packed kernels.
#[test]
fn dense_activation_capacity_refusals_preserve_healthy_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).unwrap();
    let inputs = [0.25, 0.75];
    let weights = layer.get_weights();
    let packed_weights = layer.packed_weights_flat().to_vec();
    let before = layer.try_forward_fast(&inputs, 42).unwrap();
    let packed = [vec![u64::MAX, 1], vec![0, 0]];
    layer.n_neurons = usize::MAX;
    for result in [
        layer.try_forward(&inputs, 42),
        layer.try_forward_fast(&inputs, 42),
        layer.try_forward_fused(&inputs, 42),
        layer.try_forward_prepacked(&packed),
        layer.try_forward_prepacked_2d(&[u64::MAX, 1, 0, 0], 2, 2),
    ] {
        assert_eq!(result, Err(DenseLayerError::Allocation));
    }
    for result in [
        layer.forward(&inputs, 42),
        layer.forward_fast(&inputs, 42),
        layer.forward_fused(&inputs, 42),
        layer.forward_numpy_inner(&inputs, 42),
        layer.forward_prepacked(&packed),
        layer.forward_prepacked_2d(&[u64::MAX, 1, 0, 0], 2, 2),
    ] {
        assert_eq!(result, Err("dense storage cannot be allocated".into()));
    }
    layer.n_neurons = 2;
    assert_eq!(layer.get_weights(), weights);
    assert_eq!(layer.packed_weights_flat(), packed_weights);
    assert_eq!(layer.try_forward_fast(&inputs, 42), Ok(before));
    assert_eq!(
        layer.try_forward_prepacked(&packed),
        layer.try_forward_prepacked_2d(&[u64::MAX, 1, 0, 0], 2, 2)
    );
}

/// Reject flat packed extent overflow before accepting a borrowed input slice.
#[test]
fn dense_prepacked_extent_capacity_refusal_preserves_retry() {
    let mut layer = DenseLayer::try_new(2, 2, 65, 24301).unwrap();
    let weights = layer.get_weights();
    let packed_weights = layer.packed_weights_flat().to_vec();
    let input = [u64::MAX, 1, 0, 0];
    let before = layer.try_forward_prepacked_2d(&input, 2, 2).unwrap();
    layer.n_inputs = usize::MAX;
    assert_eq!(
        layer.try_forward_prepacked_2d(&[], usize::MAX, 2),
        Err(DenseLayerError::Allocation)
    );
    assert_eq!(
        layer.forward_prepacked_2d(&[], usize::MAX, 2),
        Err("dense storage cannot be allocated".into())
    );
    layer.n_inputs = 2;
    assert_eq!(layer.get_weights(), weights);
    assert_eq!(layer.packed_weights_flat(), packed_weights);
    assert_eq!(layer.try_forward_prepacked_2d(&input, 2, 2), Ok(before));
    assert_eq!(input, [u64::MAX, 1, 0, 0]);
}

/// Refuse batch byte capacity even when the element-count multiplication fits.
#[test]
fn dense_batch_byte_capacity_refusal_preserves_retry() {
    let layer = DenseLayer::try_new(2, 2, 65, 24301).unwrap();
    let inputs = [0.25, 0.75];
    let weights = layer.get_weights();
    let packed = layer.packed_weights_flat().to_vec();
    let before = layer.try_forward_batch(&inputs, 1, 42).unwrap();
    let samples = usize::MAX / 2;
    assert!(samples.checked_mul(2).is_some());
    assert_eq!(
        layer.try_forward_batch(&[], samples, 42),
        Err(DenseLayerError::Allocation)
    );
    assert_eq!(
        layer.forward_batch(&[], samples, 42),
        Err("dense storage cannot be allocated".into())
    );
    assert_eq!(layer.get_weights(), weights);
    assert_eq!(layer.packed_weights_flat(), packed);
    assert_eq!(layer.try_forward_batch(&inputs, 1, 42), Ok(before));
}
