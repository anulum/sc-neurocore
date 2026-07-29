// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for spike-domain few-shot learning

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrototypeMetric {
    Cosine,
    Euclidean,
    Hamming,
}

#[derive(Debug, Clone)]
pub struct HebbianFewShot {
    pub n_features: usize,
    pub n_classes: usize,
    pub lr_hebbian: f64,
    memory: Vec<f64>,
    counts: Vec<usize>,
}

impl HebbianFewShot {
    pub fn new(n_features: usize, n_classes: usize, lr_hebbian: f64) -> Result<Self, String> {
        if n_features == 0 {
            return Err("n_features must be positive".to_string());
        }
        if n_classes == 0 {
            return Err("n_classes must be positive".to_string());
        }
        if !lr_hebbian.is_finite() || lr_hebbian < 0.0 {
            return Err("lr_hebbian must be finite and non-negative".to_string());
        }

        Ok(Self {
            n_features,
            n_classes,
            lr_hebbian,
            memory: vec![0.0; n_features * n_classes],
            counts: vec![0; n_classes],
        })
    }

    pub fn store(&mut self, pattern: &[f64], label: usize) -> Result<(), String> {
        self.validate_label(label)?;
        validate_pattern(pattern, self.n_features, "spike_pattern")?;

        let row_start = label * self.n_features;
        for (idx, value) in pattern.iter().enumerate() {
            self.memory[row_start + idx] += self.lr_hebbian * value;
        }
        self.counts[label] += 1;
        Ok(())
    }

    pub fn query_scores(&self, pattern: &[f64]) -> Result<Vec<f64>, String> {
        validate_pattern(pattern, self.n_features, "spike_pattern")?;
        let mut scores = vec![0.0; self.n_classes];

        for (class_idx, score) in scores.iter_mut().enumerate() {
            if self.counts[class_idx] == 0 {
                continue;
            }
            let row_start = class_idx * self.n_features;
            let memory = &self.memory[row_start..row_start + self.n_features];
            *score = cosine_score(memory, pattern);
        }
        Ok(scores)
    }

    pub fn query(&self, pattern: &[f64]) -> Result<usize, String> {
        if self.counts.iter().all(|count| *count == 0) {
            return Err("at least one support example must be stored before query".to_string());
        }

        let scores = self.query_scores(pattern)?;
        scores
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(class_idx, _)| class_idx)
            .ok_or_else(|| "no class scores were computed".to_string())
    }

    pub fn reset(&mut self) {
        self.memory.fill(0.0);
        self.counts.fill(0);
    }

    pub fn export_weights(&self) -> Vec<f64> {
        self.memory.clone()
    }

    fn validate_label(&self, label: usize) -> Result<(), String> {
        if label >= self.n_classes {
            return Err(format!("label must be in [0, {})", self.n_classes));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SpikePrototypeNet {
    pub n_features: usize,
    pub metric: PrototypeMetric,
    pub prototypes: Vec<(usize, Vec<f64>)>,
}

impl SpikePrototypeNet {
    pub fn new(n_features: usize, metric: PrototypeMetric) -> Result<Self, String> {
        if n_features == 0 {
            return Err("n_features must be positive".to_string());
        }
        Ok(Self {
            n_features,
            metric,
            prototypes: Vec::new(),
        })
    }

    pub fn classify(
        &mut self,
        support_x: &[Vec<f64>],
        support_y: &[usize],
        query_x: &[Vec<f64>],
    ) -> Result<Vec<usize>, String> {
        self.prototypes = build_prototypes(support_x, support_y, self.n_features)?;
        let mut predictions = Vec::with_capacity(query_x.len());

        for query in query_x {
            validate_pattern(query, self.n_features, "query")?;
            let mut best_class = self.prototypes[0].0;
            let mut best_score = f64::NEG_INFINITY;
            for (class_idx, prototype) in &self.prototypes {
                let score = metric_score(self.metric, query, prototype);
                if score > best_score {
                    best_score = score;
                    best_class = *class_idx;
                }
            }
            predictions.push(best_class);
        }
        Ok(predictions)
    }

    pub fn export_prototypes(&self) -> Vec<(usize, Vec<f64>)> {
        self.prototypes.clone()
    }
}

fn validate_pattern(pattern: &[f64], n_features: usize, name: &str) -> Result<(), String> {
    if pattern.len() != n_features {
        return Err(format!("{name} must resolve to {n_features} features"));
    }
    if pattern.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn cosine_score(lhs: &[f64], rhs: &[f64]) -> f64 {
    let dot: f64 = lhs.iter().zip(rhs).map(|(a, b)| a * b).sum();
    let lhs_norm = lhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = lhs_norm * rhs_norm;
    if denom <= 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

fn metric_score(metric: PrototypeMetric, query: &[f64], prototype: &[f64]) -> f64 {
    match metric {
        PrototypeMetric::Cosine => cosine_score(query, prototype),
        PrototypeMetric::Euclidean => -query
            .iter()
            .zip(prototype)
            .map(|(lhs, rhs)| {
                let diff = lhs - rhs;
                diff * diff
            })
            .sum::<f64>()
            .sqrt(),
        PrototypeMetric::Hamming => {
            let disagreements = query
                .iter()
                .zip(prototype)
                .filter(|(lhs, rhs)| (**lhs > 0.0) != (**rhs > 0.0))
                .count();
            -(disagreements as f64) / (query.len() as f64)
        }
    }
}

fn build_prototypes(
    support_x: &[Vec<f64>],
    support_y: &[usize],
    n_features: usize,
) -> Result<Vec<(usize, Vec<f64>)>, String> {
    if support_x.is_empty() {
        return Err("support_x must contain at least one support pattern".to_string());
    }
    if support_x.len() != support_y.len() {
        return Err("support_x and support_y must have the same length".to_string());
    }

    let mut labels = support_y.to_vec();
    labels.sort_unstable();
    labels.dedup();

    let mut prototypes = Vec::with_capacity(labels.len());
    for label in labels {
        let mut prototype = vec![0.0; n_features];
        let mut count = 0usize;
        for (pattern, pattern_label) in support_x.iter().zip(support_y) {
            if *pattern_label != label {
                continue;
            }
            validate_pattern(pattern, n_features, "support pattern")?;
            for (idx, value) in pattern.iter().enumerate() {
                prototype[idx] += value;
            }
            count += 1;
        }
        for value in &mut prototype {
            *value /= count as f64;
        }
        prototypes.push((label, prototype));
    }

    Ok(prototypes)
}

pub fn validate_haam() -> bool {
    let mut learner = match HebbianFewShot::new(4, 2, 0.1) {
        Ok(value) => value,
        Err(_) => return false,
    };
    learner.store(&[1.0, 0.0, 0.0, 0.0], 0).is_ok() && learner.query(&[0.9, 0.0, 0.0, 0.0]) == Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hebbian_memory_scores_and_resets() {
        let mut learner = HebbianFewShot::new(4, 2, 0.5).unwrap();
        learner.store(&[1.0, 0.0, 1.0, 0.0], 0).unwrap();
        learner.store(&[0.0, 1.0, 0.0, 1.0], 1).unwrap();

        assert_eq!(learner.query(&[0.8, 0.0, 0.9, 0.0]).unwrap(), 0);
        assert!(learner.query_scores(&[0.8, 0.0, 0.9, 0.0]).unwrap()[0] > 0.99);
        assert_eq!(learner.export_weights()[0], 0.5);

        learner.reset();
        assert!(learner.query(&[1.0, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn hebbian_validation_rejects_bad_contracts() {
        assert!(HebbianFewShot::new(0, 2, 0.1).is_err());
        assert!(HebbianFewShot::new(2, 0, 0.1).is_err());
        assert!(HebbianFewShot::new(2, 2, f64::NAN).is_err());

        let mut learner = HebbianFewShot::new(2, 2, 0.1).unwrap();
        assert!(learner.store(&[1.0], 0).is_err());
        assert!(learner.store(&[1.0, f64::INFINITY], 0).is_err());
        assert!(learner.store(&[1.0, 0.0], 2).is_err());
    }

    #[test]
    fn prototype_classifier_supports_all_metrics() {
        let support_x = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        let support_y = vec![0, 1];
        let query_x = vec![vec![0.9, 0.1, 0.0]];

        for metric in [
            PrototypeMetric::Cosine,
            PrototypeMetric::Euclidean,
            PrototypeMetric::Hamming,
        ] {
            let mut net = SpikePrototypeNet::new(3, metric).unwrap();
            assert_eq!(
                net.classify(&support_x, &support_y, &query_x).unwrap(),
                vec![0]
            );
            assert_eq!(net.export_prototypes().len(), 2);
        }
    }

    #[test]
    fn prototype_validation_rejects_malformed_support() {
        let mut net = SpikePrototypeNet::new(2, PrototypeMetric::Cosine).unwrap();
        assert!(SpikePrototypeNet::new(0, PrototypeMetric::Cosine).is_err());
        assert!(net.classify(&[], &[], &[vec![1.0, 0.0]]).is_err());
        assert!(net
            .classify(&[vec![1.0, 0.0]], &[], &[vec![1.0, 0.0]])
            .is_err());
        assert!(net
            .classify(&[vec![1.0, 0.0]], &[0], &[vec![1.0, f64::NAN]])
            .is_err());
    }

    #[test]
    fn validate_haam_accepts_reference_episode() {
        assert!(validate_haam());
    }
}
