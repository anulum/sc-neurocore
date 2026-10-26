// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native dense allocation refusal and recovery

//! Exercise public dense storage under real, process-local Linux address limits.

#![cfg(target_os = "linux")]

use std::ffi::{c_int, c_ulong};
use std::fs;
use std::io;
use std::process::Command;

use sc_neurocore_engine::layer::{DenseLayer, DenseLayerError};

#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceLimit {
    current: c_ulong,
    maximum: c_ulong,
}

unsafe extern "C" {
    fn getrlimit(resource: c_int, limit: *mut ResourceLimit) -> c_int;
    fn setrlimit(resource: c_int, limit: *const ResourceLimit) -> c_int;
}

struct AddressSpaceLimit(ResourceLimit);

impl AddressSpaceLimit {
    fn enter() -> Self {
        let status = fs::read_to_string("/proc/self/status").unwrap();
        let virtual_kib: c_ulong = status
            .lines()
            .find_map(|line| line.strip_prefix("VmSize:"))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let mut previous = ResourceLimit {
            current: 0,
            maximum: 0,
        };
        // SAFETY: Linux RLIMIT_AS is 9; repr(C) matches its two rlim_t fields.
        assert_eq!(unsafe { getrlimit(9, &mut previous) }, 0);
        let requested = ResourceLimit {
            current: virtual_kib.checked_mul(1024).unwrap() + 1024 * 1024,
            maximum: previous.maximum,
        };
        assert!(requested.current <= previous.maximum);
        // SAFETY: The valid limit preserves the original hard ceiling.
        assert_eq!(unsafe { setrlimit(9, &requested) }, 0);
        Self(previous)
    }
}

impl Drop for AddressSpaceLimit {
    fn drop(&mut self) {
        // SAFETY: This child owns the saved limit and never changes its hard ceiling.
        let result = unsafe { setrlimit(9, &self.0) };
        assert_eq!(result, 0, "{}", io::Error::last_os_error());
    }
}

fn isolated(case: &str, exercise: impl FnOnce()) {
    if std::env::var("SC_DENSE_ALLOCATION_CASE").as_deref() == Ok(case) {
        exercise();
        return;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", case, "--nocapture", "--test-threads=1"])
        .env("SC_DENSE_ALLOCATION_CASE", case)
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{case}: {}\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("1 passed; 0 failed; 0 ignored"));
}

fn copy_refusal_and_retry(n_inputs: usize, n_neurons: usize) {
    let layer = DenseLayer::try_new(n_inputs, n_neurons, 1, 24301).unwrap();
    let probe = DenseLayer::try_new(524288, 1, 1, 24301).unwrap();
    let mut retained = Vec::with_capacity(32);
    let limit = AddressSpaceLimit::enter();
    let mut exhausted = false;
    for _ in 0..32 {
        match probe.try_get_weights() {
            Ok(copy) => retained.push(copy),
            Err(error) => {
                assert_eq!(error, DenseLayerError::Allocation);
                exhausted = true;
                break;
            }
        }
    }
    let refused = layer.try_get_weights();
    drop(limit);
    assert!(
        exhausted,
        "bounded public copies did not exhaust mapped storage"
    );
    assert!(
        matches!(refused, Err(DenseLayerError::Allocation)),
        "public weight copy did not refuse after a smaller row allocation failed"
    );
    assert!(retained.len() < 32);

    let control = DenseLayer::try_new(n_inputs, n_neurons, 1, 24301).unwrap();
    assert_eq!(layer.weights, control.weights);
    assert_eq!(layer.packed_weights_flat(), control.packed_weights_flat());
    let mut copied = layer.try_get_weights().unwrap();
    assert_eq!(copied, layer.weights);
    copied[0][0] = -1.0;
    assert_eq!(layer.weights, control.weights);
    assert_eq!(layer.packed_weights_flat(), control.packed_weights_flat());
}

/// Refuse the large outer row table without changing either weight representation.
#[test]
fn native_weight_copy_outer_refusal_preserves_state_and_retry() {
    isolated(
        "native_weight_copy_outer_refusal_preserves_state_and_retry",
        || copy_refusal_and_retry(1, 524288),
    );
}

/// Refuse a probability-row copy after the small outer table fits.
#[test]
fn native_weight_copy_inner_refusal_preserves_state_and_retry() {
    isolated(
        "native_weight_copy_inner_refusal_preserves_state_and_retry",
        || copy_refusal_and_retry(524288, 2),
    );
}

/// Propagate packed-constructor refusal and retain a seeded healthy constructor.
#[test]
fn native_constructor_packed_refusal_preserves_healthy_retry() {
    isolated(
        "native_constructor_packed_refusal_preserves_healthy_retry",
        || {
            let healthy = DenseLayer::try_new(2, 2, 65, 24301).unwrap();
            let weights = healthy.try_get_weights().unwrap();
            let packed = healthy.packed_weights_flat().to_vec();
            let limit = AddressSpaceLimit::enter();
            let refused = DenseLayer::try_new(1, 2, 1 << 29, 24301);
            drop(limit);
            assert!(matches!(refused, Err(DenseLayerError::Allocation)));
            assert_eq!(healthy.weights, weights);
            assert_eq!(healthy.packed_weights_flat(), packed);
            let retry = DenseLayer::try_new(2, 2, 65, 24301).unwrap();
            assert_eq!(retry.weights, weights);
            assert_eq!(retry.packed_weights_flat(), packed);
            assert_eq!(
                retry.try_forward_fast(&[1.0, 0.0], 42),
                healthy.try_forward_fast(&[1.0, 0.0], 42)
            );
        },
    );
}
