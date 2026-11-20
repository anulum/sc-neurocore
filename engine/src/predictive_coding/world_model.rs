// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Online autoregressive predictor weight update

//! Update one observed world-model step without materialising an outer product.

/// Apply the least-mean-squares weight and bias update in row-major channel order.
///
/// The caller provides the probabilities used to predict the current step and
/// the same oldest-first history vector. The two multiplications have the same
/// order as the scalar operations in the reference NumPy outer product.
pub(crate) fn update_world_model(
    weights: &mut [f64],
    bias: &mut [f64],
    features: &[f64],
    probabilities: &[f64],
    observed: &[f64],
    learning_rate: f64,
) -> Result<(), &'static str> {
    let channels = bias.len();
    let columns = features.len();
    if probabilities.len() != channels
        || observed.len() != channels
        || channels.checked_mul(columns) != Some(weights.len())
    {
        return Err("world-model update dimensions must match");
    }

    for channel in 0..channels {
        let error = observed[channel] - probabilities[channel];
        let start = channel * columns;
        for (weight, &feature) in weights[start..start + columns].iter_mut().zip(features) {
            *weight += learning_rate * (error * feature);
        }
        bias[channel] += learning_rate * error;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_errors_update_weights_and_bias_independently() {
        let mut weights = [0.0, 0.0, 0.0, 0.0];
        let mut bias = [0.0, 0.125];
        update_world_model(
            &mut weights,
            &mut bias,
            &[1.0, 0.0],
            &[0.25, 0.75],
            &[1.0, 0.0],
            0.5,
        )
        .unwrap();
        assert_eq!(weights, [0.375, 0.0, -0.375, 0.0]);
        assert_eq!(bias, [0.375, -0.25]);
    }

    #[test]
    fn empty_channels_leave_the_model_empty() {
        update_world_model(&mut [], &mut [], &[], &[], &[], 0.1).unwrap();
    }

    #[test]
    fn shape_refusal_does_not_change_state() {
        let mut weights = [0.25];
        let mut bias = [0.5];
        for (features, probabilities, observed) in [
            (&[0.0][..], &[][..], &[1.0][..]),
            (&[0.0][..], &[0.5][..], &[][..]),
            (&[][..], &[0.5][..], &[1.0][..]),
        ] {
            assert!(update_world_model(
                &mut weights,
                &mut bias,
                features,
                probabilities,
                observed,
                0.1,
            )
            .is_err());
            assert_eq!(weights, [0.25]);
            assert_eq!(bias, [0.5]);
        }
    }
}
