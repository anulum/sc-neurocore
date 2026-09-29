// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actual dense IF core and resource admission tests

//! Exercise native core recurrence, state custody and exact buffer limits.

use super::*;

#[test]
fn bias_shift_and_complete_trace() {
    let layer = DenseLayer::new(1, 1, vec![1.0], Some(vec![0.25]), 1.0).unwrap();
    let model = ConvertedSNN::new(vec![layer], 0.5, OutputMode::Spikes).unwrap();
    let result = model
        .replay(
            &[0.0, 0.25, 0.75, 1.0, 0.0, 0.5],
            (6, 1),
            None,
            true,
            false,
            1024,
        )
        .unwrap();
    assert_eq!(result.output, vec![4.0]);
    assert_eq!(result.final_state, vec![vec![0.5]]);
    assert_eq!(
        result.state_trace,
        vec![vec![0.75, 0.25, 0.25, 0.5, 0.75, 0.5]]
    );
    assert_eq!(result.spike_trace, vec![vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0]]);
    assert_eq!(model.classify(&result, 1).unwrap(), vec![0]);
}

#[test]
fn signed_linear_and_empty_time_state_identity() {
    let layer = DenseLayer::new(1, 1, vec![-2.0], Some(vec![-0.25]), 1.0).unwrap();
    let model = ConvertedSNN::new(vec![layer], 0.5, OutputMode::Linear).unwrap();
    let result = model
        .replay(&[1.0, 0.0], (2, 1), None, true, true, 1024)
        .unwrap();
    assert_eq!(result.output, vec![-2.5]);
    assert_eq!(result.state_trace, vec![vec![-2.25, -2.5]]);
    assert!(result.spike_trace.is_empty());
    let initial = vec![vec![-2.5]];
    let empty = model
        .replay(&[], (0, 1), Some(&initial), true, true, 1024)
        .unwrap();
    assert_eq!(empty.output, vec![-2.5]);
    assert_eq!(initial, vec![vec![-2.5]]);
}

#[test]
fn late_bad_input_and_large_state_refuse() {
    let layer = DenseLayer::new(1, 1, vec![1.0], None, 1.0).unwrap();
    let model = ConvertedSNN::new(vec![layer], 0.0, OutputMode::Spikes).unwrap();
    assert!(matches!(
        model.replay(&[1.0, f64::NAN], (2, 1), None, true, true, 1024),
        Err(ReplayError::InvalidInput)
    ));
    assert!(matches!(
        model.replay(&[], (0, usize::MAX), None, false, true, 1024),
        Err(ReplayError::ResourceLimit)
    ));
}
#[test]
fn replay_budget_boundaries_and_empty_batch() {
    let snn = ConvertedSNN::new(
        vec![DenseLayer::new(1, 1, vec![1.0], None, 1.0).unwrap()],
        0.0,
        OutputMode::Spikes,
    )
    .unwrap();
    assert_eq!(
        snn.replay(&[1.0], (1, 1), None, false, true, 111)
            .unwrap_err(),
        ReplayError::ResourceLimit
    );
    assert_eq!(
        snn.replay(&[1.0], (1, 1), None, false, true, 112)
            .unwrap()
            .output,
        vec![1.0]
    );
    assert_eq!(
        snn.replay(&[1.0], (1, 1), None, true, true, 127)
            .unwrap_err(),
        ReplayError::ResourceLimit
    );
    assert_eq!(
        snn.replay(&[1.0], (1, 1), None, true, true, 128)
            .unwrap()
            .output,
        vec![1.0]
    );
    assert_eq!(
        snn.replay(&[], (0, 1usize << 30), None, true, true, 256usize << 20)
            .unwrap_err(),
        ReplayError::ResourceLimit
    );
    assert!(snn
        .replay(&[], (1usize << 53, 0), None, true, true, 16)
        .unwrap()
        .output
        .is_empty());
    assert_eq!(
        snn.replay(&[], (0, 0), None, false, true, 0).unwrap_err(),
        ReplayError::InvalidInput
    );
}

#[test]
fn layer_preloads_preserve_mixed_activation_recurrence() {
    let first = DenseLayer::new(1, 1, vec![1.0], Some(vec![0.25]), 1.0)
        .unwrap()
        .with_initial_fraction(0.0)
        .unwrap();
    let last = DenseLayer::new(1, 1, vec![1.0], Some(vec![0.25]), 1.0)
        .unwrap()
        .with_initial_fraction(0.5)
        .unwrap();
    let snn = ConvertedSNN::new(vec![first, last], 0.5, OutputMode::Spikes).unwrap();
    let r = snn
        .replay(&[0.5, 0.5], (2, 1), None, true, false, 1024)
        .unwrap();
    assert_eq!(r.output, vec![1.0]);
    assert_eq!(r.state_trace, vec![vec![0.75, 0.5], vec![0.75, 1.0]]);
    assert_eq!(
        DenseLayer::new(1, 1, vec![1.0], None, 1.0)
            .unwrap()
            .with_initial_fraction(f64::NAN)
            .unwrap_err(),
        ReplayError::InvalidInput
    );
}
