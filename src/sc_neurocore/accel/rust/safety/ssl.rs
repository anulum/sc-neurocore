// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for spike contrastive SSL

#[derive(Debug, Clone, Copy)]
pub struct SpikeContrastiveLoss {
    pub temperature: f64,
}

impl SpikeContrastiveLoss {
    pub fn new(temperature: f64) -> Result<Self, String> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("temperature must be finite and positive".to_string());
        }
        Ok(Self { temperature })
    }

    pub fn compute(&self, view_a: &[Vec<f64>], view_b: &[Vec<f64>]) -> Result<f64, String> {
        validate_views(view_a, view_b)?;
        let batch = view_a.len();
        if batch < 2 {
            return Ok(0.0);
        }

        let a_norm = normalise_rows(view_a);
        let b_norm = normalise_rows(view_b);
        let mut total = 0.0;

        for row in 0..batch {
            let mut logits = Vec::with_capacity(batch);
            for rhs in &b_norm {
                logits.push(dot(&a_norm[row], rhs) / self.temperature);
            }
            let row_max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = logits.iter().map(|value| (value - row_max).exp()).collect();
            let denom: f64 = exp_logits.iter().sum();
            let prob = (exp_logits[row] / denom).max(1e-10);
            total += prob.ln();
        }

        Ok(-total / batch as f64)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CSDPRule {
    pub lr: f64,
    pub decay: f64,
}

impl CSDPRule {
    pub fn new(lr: f64, decay: f64) -> Result<Self, String> {
        validate_non_negative(lr, "lr")?;
        validate_non_negative(decay, "decay")?;
        Ok(Self { lr, decay })
    }

    pub fn positive_update(
        &self,
        weights: &[Vec<f64>],
        pre_spikes: &[f64],
        post_spikes: &[f64],
    ) -> Result<Vec<Vec<f64>>, String> {
        validate_update_inputs(weights, pre_spikes, post_spikes)?;
        let mut updated = weights.to_vec();
        for post_idx in 0..post_spikes.len() {
            for pre_idx in 0..pre_spikes.len() {
                updated[post_idx][pre_idx] += self.lr * post_spikes[post_idx] * pre_spikes[pre_idx]
                    - self.decay * weights[post_idx][pre_idx];
            }
        }
        Ok(updated)
    }

    pub fn negative_update(
        &self,
        weights: &[Vec<f64>],
        pre_spikes: &[f64],
        post_spikes: &[f64],
    ) -> Result<Vec<Vec<f64>>, String> {
        validate_update_inputs(weights, pre_spikes, post_spikes)?;
        let mut updated = weights.to_vec();
        for post_idx in 0..post_spikes.len() {
            for pre_idx in 0..pre_spikes.len() {
                updated[post_idx][pre_idx] -= self.lr * post_spikes[post_idx] * pre_spikes[pre_idx];
            }
        }
        Ok(updated)
    }

    pub fn contrastive_step(
        &self,
        weights: &[Vec<f64>],
        pos_pre: &[f64],
        pos_post: &[f64],
        neg_pre: &[f64],
        neg_post: &[f64],
    ) -> Result<Vec<Vec<f64>>, String> {
        let after_positive = self.positive_update(weights, pos_pre, pos_post)?;
        self.negative_update(&after_positive, neg_pre, neg_post)
    }

    pub fn goodness(&self, activations: &[f64]) -> Result<f64, String> {
        validate_vector(activations, "activations")?;
        Ok(activations.iter().map(|value| value * value).sum())
    }
}

impl Default for CSDPRule {
    fn default() -> Self {
        Self {
            lr: 0.01,
            decay: 0.001,
        }
    }
}

pub fn validate_ssl() -> bool {
    let loss = match SpikeContrastiveLoss::new(0.5) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let view_a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let view_b = view_a.clone();
    let rule = CSDPRule::default();
    let weights = vec![vec![0.2, 0.4], vec![0.1, 0.3]];
    loss.compute(&view_a, &view_b).is_ok()
        && rule
            .contrastive_step(
                &weights,
                &[1.0, 0.5],
                &[0.25, 1.0],
                &[0.0, 1.0],
                &[0.5, 0.5],
            )
            .is_ok()
        && matches!(rule.goodness(&[1.0, -2.0, 0.5]), Ok(value) if (value - 5.25).abs() < 1e-12)
}

fn validate_views(view_a: &[Vec<f64>], view_b: &[Vec<f64>]) -> Result<(), String> {
    if view_a.len() != view_b.len() {
        return Err("view_a and view_b must have the same shape".to_string());
    }
    if view_a.is_empty() {
        return Ok(());
    }
    let n_features = view_a[0].len();
    if n_features == 0 {
        return Err("views must contain at least one feature".to_string());
    }
    validate_matrix(view_a, n_features, "view_a")?;
    validate_matrix(view_b, n_features, "view_b")
}

fn validate_matrix(matrix: &[Vec<f64>], n_features: usize, name: &str) -> Result<(), String> {
    for row in matrix {
        if row.len() != n_features {
            return Err(format!("{name} rows must have the same feature count"));
        }
        validate_vector(row, name)?;
    }
    Ok(())
}

fn validate_update_inputs(
    weights: &[Vec<f64>],
    pre_spikes: &[f64],
    post_spikes: &[f64],
) -> Result<(), String> {
    validate_vector(pre_spikes, "pre_spikes")?;
    validate_vector(post_spikes, "post_spikes")?;
    if weights.len() != post_spikes.len() {
        return Err("weights must have len(post_spikes) rows".to_string());
    }
    for row in weights {
        if row.len() != pre_spikes.len() {
            return Err("weights rows must have len(pre_spikes) columns".to_string());
        }
        validate_vector(row, "weights")?;
    }
    Ok(())
}

fn validate_vector(values: &[f64], name: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn validate_non_negative(value: f64, name: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(())
}

fn normalise_rows(values: &[Vec<f64>]) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|row| {
            let denom = row
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                .max(1e-8);
            row.iter().map(|value| value / denom).collect()
        })
        .collect()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(left, right)| left * right).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssl_compute_prefers_identical_views() {
        let loss = SpikeContrastiveLoss::new(0.25).unwrap();
        let view = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let permuted = vec![
            view[1].clone(),
            view[2].clone(),
            view[3].clone(),
            view[0].clone(),
        ];

        assert!(loss.compute(&view, &view).unwrap() < loss.compute(&view, &permuted).unwrap());
    }

    #[test]
    fn test_ssl_rejects_bad_temperature() {
        assert!(SpikeContrastiveLoss::new(0.0).is_err());
        assert!(SpikeContrastiveLoss::new(f64::NAN).is_err());
    }

    #[test]
    fn test_csdp_updates_match_rule() {
        let rule = CSDPRule::new(0.1, 0.01).unwrap();
        let weights = vec![vec![0.2, 0.4], vec![0.1, 0.3]];
        let updated = rule
            .positive_update(&weights, &[1.0, 0.5], &[0.25, 1.0])
            .unwrap();

        assert!((updated[0][0] - 0.223).abs() < 1e-12);
        assert!((updated[1][1] - 0.347).abs() < 1e-12);
    }

    #[test]
    fn test_validate_ssl() {
        assert!(validate_ssl());
    }
}
