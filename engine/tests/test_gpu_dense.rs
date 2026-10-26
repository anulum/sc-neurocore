// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU dense allocation and retry contracts

//! Real GPU dense entry points, capacity refusals and shared-buffer isolation.

#![cfg(feature = "gpu")]

use sc_neurocore_engine::gpu::dense::GpuDenseError;
use sc_neurocore_engine::gpu::{context, GpuDenseLayer};
use sc_neurocore_engine::layer::DenseLayerError;

#[test]
fn constructor_rejects_zero_dimensions_in_contract_order() {
    for (inputs, neurons, length, batch, message) in [
        (0, 0, 0, 0, "bitstream length must be > 0"),
        (0, 0, 1, 0, "n_inputs must be > 0"),
        (1, 0, 1, 0, "n_neurons must be > 0"),
        (1, 1, 1, 0, "max_batch must be > 0"),
    ] {
        match GpuDenseLayer::try_new_checked(inputs, neurons, length, 42, batch) {
            Err(GpuDenseError::Dense(DenseLayerError::Invalid(actual))) => {
                assert_eq!(actual, message)
            }
            _ => panic!("zero dimension did not produce an authored input refusal"),
        }
    }
}

#[test]
fn constructor_rejects_wrapping_storage_before_allocation() {
    for (inputs, neurons, length, batch) in [
        (usize::MAX, 2, 64, 1),
        (1, usize::MAX, 64, 1),
        (1, 1, 64, usize::MAX),
        (64, 2, usize::MAX, 2),
    ] {
        assert!(matches!(
            GpuDenseLayer::try_new_checked(inputs, neurons, length, 42, batch),
            Err(GpuDenseError::Dense(DenseLayerError::Allocation))
        ));
    }
}

#[test]
fn constructor_rejects_shader_index_truncation() {
    let too_large = u32::MAX as usize + 1;
    match GpuDenseLayer::try_new_checked(1, 1, too_large, 42, 1) {
        Err(GpuDenseError::Dense(DenseLayerError::Invalid(message))) => {
            assert_eq!(message, "GPU dense dimensions exceed u32 shader indexing");
        }
        _ => panic!("shader index truncation was admitted"),
    }
}

#[test]
fn constructor_obeys_actual_device_buffer_and_dispatch_limits() {
    let ctx = context::get_context().expect("physical GPU context is required");
    let limits = ctx.device.limits();
    let oversized_batch = limits.max_storage_buffer_binding_size as usize / 4 + 1;
    match GpuDenseLayer::try_new_checked(1, 1, 64, 42, oversized_batch) {
        Err(GpuDenseError::Dense(DenseLayerError::Invalid(message))) => {
            assert_eq!(message, "GPU dense storage exceeds device buffer limits");
        }
        _ => panic!("oversized storage reached device allocation"),
    }
    let oversized_weights = limits.max_buffer_size as usize / 8 + 1;
    match GpuDenseLayer::try_new_checked(1, oversized_weights, 64, 42, 1) {
        Err(GpuDenseError::Dense(DenseLayerError::Invalid(message))) => {
            assert_eq!(message, "GPU dense storage exceeds device buffer limits");
        }
        _ => panic!("oversized weight buffer reached device allocation"),
    }
    let beyond_dispatch = limits.max_compute_workgroups_per_dimension as usize + 1;
    for (inputs, neurons, batch) in [
        (1, beyond_dispatch, 1),
        (1, 1, beyond_dispatch),
        ((beyond_dispatch - 1) * 256 + 1, 1, 1),
    ] {
        match GpuDenseLayer::try_new_checked(inputs, neurons, 64, 42, batch) {
            Err(GpuDenseError::Dense(DenseLayerError::Invalid(message))) => {
                assert_eq!(
                    message,
                    "GPU dense dimensions exceed device dispatch limits"
                );
            }
            _ => panic!("oversized dispatch reached device submission"),
        }
    }
}

#[test]
fn shape_and_capacity_refusals_preserve_weights_and_healthy_retry() {
    let layer = GpuDenseLayer::try_new_checked(2, 3, 65, 24301, 3)
        .expect("construction must succeed")
        .expect("physical GPU required");
    let input = [0.25, 0.75];
    let weights = layer.cpu.get_weights();
    let packed = layer.cpu.packed_weights_flat().to_vec();
    let before = layer.try_forward_gpu(&input, 42).unwrap();
    assert_eq!(before.len(), 3);
    for (values, samples, message) in [
        (vec![0.25], 1, "Expected 2 values (1×2), got 1."),
        (
            vec![],
            usize::MAX,
            "Input size overflow when validating n_samples * n_inputs.",
        ),
        (vec![0.5; 8], 4, "Batch size 4 exceeds max 3"),
    ] {
        match layer.try_forward_batch_gpu(&values, samples, 42) {
            Err(GpuDenseError::Dense(DenseLayerError::Invalid(actual))) => {
                assert_eq!(actual, message)
            }
            _ => panic!("invalid batch was admitted"),
        }
        assert_eq!(layer.cpu.weights, weights);
        assert_eq!(layer.cpu.packed_weights_flat(), packed);
        assert_eq!(layer.try_forward_gpu(&input, 42).unwrap(), before);
    }
    assert!(layer.try_forward_batch_gpu(&[], 0, 42).unwrap().is_empty());
    assert_eq!(layer.forward_gpu(&input, 42), before);
    assert_eq!(layer.forward_batch_gpu(&input, 1, 42), before);
}

#[test]
fn altered_cpu_geometry_is_refused_before_gpu_submission() {
    let mut layer = GpuDenseLayer::try_new_checked(2, 3, 65, 24301, 1)
        .unwrap()
        .expect("physical GPU required");
    let before = layer.try_forward_gpu(&[0.25, 0.75], 42).unwrap();
    layer.cpu.n_neurons = usize::MAX;
    match layer.try_forward_gpu(&[0.25, 0.75], 42) {
        Err(GpuDenseError::Dense(DenseLayerError::Invalid(message))) => {
            assert_eq!(message, "GPU layer dimensions changed after construction");
        }
        _ => panic!("changed geometry reached device buffers"),
    }
    layer.cpu.n_neurons = 3;
    assert_eq!(layer.try_forward_gpu(&[0.25, 0.75], 42).unwrap(), before);
}

#[test]
fn concurrent_samples_keep_their_own_seeded_results() {
    let layer = GpuDenseLayer::try_new_checked(3, 4, 129, 24301, 1)
        .unwrap()
        .expect("physical GPU required");
    let first_input = [0.125, 0.5, 0.875];
    let second_input = [0.875, 0.25, 0.625];
    let first = layer.try_forward_gpu(&first_input, 7).unwrap();
    let second = layer.try_forward_gpu(&second_input, u64::MAX).unwrap();
    std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            for _ in 0..8 {
                assert_eq!(layer.try_forward_gpu(&first_input, 7).unwrap(), first);
            }
        });
        let right = scope.spawn(|| {
            for _ in 0..8 {
                assert_eq!(
                    layer.try_forward_gpu(&second_input, u64::MAX).unwrap(),
                    second
                );
            }
        });
        left.join().unwrap();
        right.join().unwrap();
    });
}
