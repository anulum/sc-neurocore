// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Checked dense IF numeric buffer reservation

//! Portable numeric reservation; caller buffers and allocator overhead excluded.

/// Calculate eight bytes per element of 2P + 2F + 2S + 2O + H + 5M + I.
///
/// P counts coefficients, F frames, S states, O output, H optional state/event
/// traces, M the largest layer and I one input frame. None denotes usize overflow.
pub fn replay_bytes(
    coefficients: usize,
    inputs: usize,
    outputs: &[usize],
    steps: usize,
    batch: usize,
    trace: bool,
    linear: bool,
) -> Option<usize> {
    let nodes = outputs
        .iter()
        .try_fold(0usize, |n, width| n.checked_add(*width))?;
    let states = batch.checked_mul(nodes)?;
    let output = batch.checked_mul(*outputs.last()?)?;
    let largest = batch.checked_mul(*outputs.iter().max()?)?;
    let frame = batch.checked_mul(inputs)?;
    let frames = steps.checked_mul(frame)?;
    let spiking_nodes = if linear {
        nodes.checked_sub(*outputs.last()?)?
    } else {
        nodes
    };
    let traces = if trace {
        steps
            .checked_mul(batch)?
            .checked_mul(nodes.checked_add(spiking_nodes)?)?
    } else {
        0
    };
    coefficients
        .checked_mul(2)?
        .checked_add(frames.checked_mul(2)?)?
        .checked_add(states.checked_mul(2)?)?
        .checked_add(output.checked_mul(2)?)?
        .checked_add(traces)?
        .checked_add(largest.checked_mul(5)?)?
        .checked_add(frame)?
        .checked_mul(8)
}

#[cfg(test)]
mod tests {
    use super::replay_bytes;

    #[test]
    fn exact_reservations_and_zero_geometry() {
        assert_eq!(replay_bytes(1, 1, &[1], 1, 1, false, false), Some(112));
        assert_eq!(replay_bytes(1, 1, &[1], 1, 1, true, false), Some(128));
        assert_eq!(replay_bytes(1, 1, &[1], 1, 1, true, true), Some(120));
        assert_eq!(
            replay_bytes(1, 1, &[1], 1usize << 53, 0, true, false),
            Some(16)
        );
        assert_eq!(replay_bytes(usize::MAX, 1, &[1], 1, 1, true, false), None);
    }
}
