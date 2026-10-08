// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sliding-window bitstream estimator tests

//! Exercise the actual sliding-window estimator and its fallible public methods.

use super::BitstreamAverager;

#[test]
fn fallible_construction_refuses_zero_and_impossible_capacity() {
    assert_eq!(
        BitstreamAverager::try_new(0).unwrap_err(),
        "window must be > 0"
    );
    assert_eq!(
        BitstreamAverager::try_new(usize::MAX).unwrap_err(),
        "Bitstream averager buffer cannot be allocated"
    );
    let mut avg = BitstreamAverager::try_new(1).unwrap();
    avg.try_push(1).unwrap();
    assert_eq!(avg.estimate(), 1.0);
}

#[test]
fn invalid_bit_preserves_partial_and_wrapped_windows() {
    for prefix in [&[1_u8][..], &[1, 0, 1, 1, 0][..]] {
        let mut avg = BitstreamAverager::new(4);
        let mut control = BitstreamAverager::new(4);
        for &bit in prefix {
            avg.push(bit);
            control.push(bit);
        }
        for bit in [2, 127, 255] {
            assert_eq!(avg.try_push(bit), Err("Bit must be 0 or 1."));
            assert_eq!(avg.estimate(), control.estimate());
        }
        for bit in [0, 1, 0, 0, 1] {
            avg.try_push(bit).unwrap();
            control.push(bit);
            assert_eq!(avg.estimate(), control.estimate());
        }
    }
}

#[test]
fn all_ones_estimate_one() {
    let mut avg = BitstreamAverager::new(100);
    for _ in 0..100 {
        avg.push(1);
    }
    assert!((avg.estimate() - 1.0).abs() < 1e-12);
}

#[test]
fn all_zeros_estimate_zero() {
    let mut avg = BitstreamAverager::new(50);
    for _ in 0..50 {
        avg.push(0);
    }
    assert!(avg.estimate().abs() < 1e-12);
}

#[test]
fn alternating_bits_estimate_half() {
    let mut avg = BitstreamAverager::new(100);
    for i in 0..100 {
        avg.push((i % 2) as u8);
    }
    assert!((avg.estimate() - 0.5).abs() < 1e-12);
}

#[test]
fn sliding_window_replaces_oldest_bits() {
    let mut avg = BitstreamAverager::new(4);
    for &bit in &[1_u8, 1, 0, 0] {
        avg.push(bit);
    }
    assert!((avg.estimate() - 0.5).abs() < 1e-12);

    avg.push(1);
    assert!((avg.estimate() - 0.5).abs() < 1e-12);
    avg.push(1);
    assert!((avg.estimate() - 0.5).abs() < 1e-12);
    avg.push(1);
    assert!((avg.estimate() - 0.75).abs() < 1e-12);
}

#[test]
fn partial_window_uses_observed_count() {
    let mut avg = BitstreamAverager::new(100);
    avg.push(1);
    avg.push(0);
    assert!((avg.estimate() - 0.5).abs() < 1e-12);
}

#[test]
fn empty_window_estimate_is_zero() {
    let avg = BitstreamAverager::new(10);
    assert!(avg.estimate().abs() < 1e-12);
}

#[test]
fn reset_clears_window_state() {
    let mut avg = BitstreamAverager::new(10);
    for _ in 0..10 {
        avg.push(1);
    }
    avg.reset();
    assert!(avg.estimate().abs() < 1e-12);
}
