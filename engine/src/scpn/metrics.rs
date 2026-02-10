//! # SCPN Metrics
//!
//! Composite metrics computed from multi-layer SCPN outputs.

/// SCPN-wide metrics computed from the 7-layer outputs.
pub struct SCPNMetrics;

impl SCPNMetrics {
    /// Compute weighted global coherence across all layers.
    /// weights: per-layer importance weights (7,)
    /// metrics: per-layer global metric values (7,)
    /// Returns: weighted average coherence ∈ [0, 1]
    pub fn global_coherence(weights: &[f64; 7], metrics: &[f64; 7]) -> f64 {
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        for idx in 0..7 {
            let weight = weights[idx].max(0.0);
            let metric = metrics[idx].clamp(0.0, 1.0);
            weighted_sum += weight * metric;
            weight_total += weight;
        }

        if weight_total == 0.0 {
            0.0
        } else {
            (weighted_sum / weight_total).clamp(0.0, 1.0)
        }
    }

    /// Compute the "consciousness index" — a composite score
    /// based on cross-layer synchronization.
    /// phases_l4: Kuramoto phases from L4
    /// glyph_l7: Glyph vector from L7
    /// Returns: index ∈ [0, 1]
    pub fn consciousness_index(phases_l4: &[f64], glyph_l7: &[f64; 6]) -> f64 {
        let phase_sync = if phases_l4.is_empty() {
            0.0
        } else {
            let n_inv = 1.0 / phases_l4.len() as f64;
            let mean_cos = phases_l4.iter().map(|theta| theta.cos()).sum::<f64>() * n_inv;
            let mean_sin = phases_l4.iter().map(|theta| theta.sin()).sum::<f64>() * n_inv;
            (mean_cos * mean_cos + mean_sin * mean_sin)
                .sqrt()
                .clamp(0.0, 1.0)
        };

        let glyph_norm =
            (glyph_l7.iter().map(|v| v * v).sum::<f64>().sqrt() / (6.0_f64).sqrt()).clamp(0.0, 1.0);

        (0.7 * phase_sync + 0.3 * glyph_norm).clamp(0.0, 1.0)
    }
}
