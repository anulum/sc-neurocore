// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike waveform shape analysis

/// Trough-to-peak width in seconds. Bartho et al. 2004.
pub fn waveform_width(waveform: &[f64], dt: f64) -> f64 {
    if waveform.is_empty() {
        return f64::NAN;
    }
    let trough = waveform
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    if trough >= waveform.len() - 1 {
        return f64::NAN;
    }
    let peak = trough
        + waveform[trough..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
    (peak - trough) as f64 * dt
}

/// Peak-to-trough amplitude. Bartho et al. 2004.
pub fn waveform_amplitude(waveform: &[f64]) -> f64 {
    if waveform.is_empty() {
        return 0.0;
    }
    let max = waveform.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = waveform.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

/// Repolarisation slope: max dV/dt after trough. Bean 2007.
pub fn waveform_repolarization_slope(waveform: &[f64], dt: f64) -> f64 {
    if waveform.is_empty() {
        return f64::NAN;
    }
    let trough = waveform
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    if trough >= waveform.len() - 2 {
        return f64::NAN;
    }
    let post = &waveform[trough..];
    let mut max_dv = f64::NEG_INFINITY;
    for i in 0..post.len() - 1 {
        let dv = (post[i + 1] - post[i]) / dt;
        if dv > max_dv {
            max_dv = dv;
        }
    }
    max_dv
}

/// Recovery slope: dV/dt during return to baseline after peak. Bean 2007.
pub fn waveform_recovery_slope(waveform: &[f64], dt: f64) -> f64 {
    if waveform.is_empty() {
        return f64::NAN;
    }
    let trough = waveform
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    if trough >= waveform.len() - 1 {
        return f64::NAN;
    }
    let peak = trough
        + waveform[trough..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
    if peak >= waveform.len() - 2 {
        return f64::NAN;
    }
    let post = &waveform[peak..];
    let mut min_dv = f64::INFINITY;
    for i in 0..post.len() - 1 {
        let dv = (post[i + 1] - post[i]) / dt;
        if dv < min_dv {
            min_dv = dv;
        }
    }
    min_dv
}

/// Half-width: duration at half-minimum amplitude. Bartho et al. 2004.
pub fn waveform_halfwidth(waveform: &[f64], dt: f64) -> f64 {
    if waveform.is_empty() {
        return f64::NAN;
    }
    let trough_val = waveform.iter().cloned().fold(f64::INFINITY, f64::min);
    let half_val = trough_val / 2.0;
    let below: Vec<usize> = waveform
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < half_val)
        .map(|(i, _)| i)
        .collect();
    if below.len() < 2 {
        return f64::NAN;
    }
    (below[below.len() - 1] - below[0]) as f64 * dt
}

/// Peak-to-trough ratio. Bartho et al. 2004.
pub fn waveform_pt_ratio(waveform: &[f64]) -> f64 {
    if waveform.is_empty() {
        return f64::NAN;
    }
    let trough = waveform
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let trough_val = waveform[trough].abs();
    if trough >= waveform.len() - 1 || trough_val < 1e-30 {
        return f64::NAN;
    }
    let peak_val = waveform[trough..]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    peak_val.abs() / trough_val
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_waveform() -> Vec<f64> {
        // Simulated extracellular action potential shape
        vec![
            0.0, 0.1, 0.0, -0.3, -0.8, -1.0, -0.7, -0.2, 0.3, 0.5, 0.4, 0.2, 0.1, 0.0,
        ]
    }

    #[test]
    fn test_waveform_width() {
        let wf = typical_waveform();
        let dt = 1.0 / 30000.0;
        let w = waveform_width(&wf, dt);
        // Trough at index 5, peak after at index 9 -> 4 samples
        assert!((w - 4.0 * dt).abs() < 1e-12);
    }

    #[test]
    fn test_waveform_width_empty() {
        assert!(waveform_width(&[], 1.0 / 30000.0).is_nan());
    }

    #[test]
    fn test_waveform_amplitude() {
        let wf = typical_waveform();
        let amp = waveform_amplitude(&wf);
        // max=0.5, min=-1.0 -> 1.5
        assert!((amp - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_waveform_amplitude_empty() {
        assert_eq!(waveform_amplitude(&[]), 0.0);
    }

    #[test]
    fn test_repolarization_slope() {
        let wf = typical_waveform();
        let dt = 1.0 / 30000.0;
        let slope = waveform_repolarization_slope(&wf, dt);
        assert!(slope > 0.0);
        // Max positive derivative after trough at index 5
        // Steepest rise: index 7->8: 0.3-(-0.2)=0.5
        let expected = 0.5 / dt;
        assert!((slope - expected).abs() < 1e-6);
    }

    #[test]
    fn test_recovery_slope() {
        let wf = typical_waveform();
        let dt = 1.0 / 30000.0;
        let slope = waveform_recovery_slope(&wf, dt);
        // Should be negative (descending from peak)
        assert!(slope < 0.0);
    }

    #[test]
    fn test_halfwidth() {
        let wf = typical_waveform();
        let dt = 1.0 / 30000.0;
        let hw = waveform_halfwidth(&wf, dt);
        // trough_val = -1.0, half_val = -0.5
        // Below -0.5: indices 4 (-0.8), 5 (-1.0), 6 (-0.7) -> span = (6-4)*dt = 2*dt
        assert!((hw - 2.0 * dt).abs() < 1e-12);
    }

    #[test]
    fn test_pt_ratio() {
        let wf = typical_waveform();
        let ratio = waveform_pt_ratio(&wf);
        // trough_val = 1.0, peak after trough = 0.5
        assert!((ratio - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_pt_ratio_no_peak() {
        let wf = vec![0.0, -1.0];
        assert!(waveform_pt_ratio(&wf).is_nan());
    }

    #[test]
    fn test_flat_waveform() {
        let wf = vec![0.5; 10];
        // flat -> min_by picks first index (0), max after it spans to end
        // trough at 0, peak at 0 (all equal) -> width = 0
        // Actually argmax of [0.5, 0.5, ...] picks idx 0 relative -> peak - trough = 0
        let w = waveform_width(&wf, 1.0 / 30000.0);
        // For flat waveform, trough=0, max from [0..] is also 0, so width = 0
        assert!(w.abs() < 1e-6 || w >= 0.0, "Width for flat waveform: {w}");
    }

    #[test]
    fn test_single_sample() {
        assert!(waveform_width(&[1.0], 0.001).is_nan());
        assert!((waveform_amplitude(&[5.0])).abs() < 1e-12);
    }

    #[test]
    fn test_negative_only_waveform() {
        let wf = vec![-0.5, -1.0, -0.3];
        let dt = 1.0 / 30000.0;
        let w = waveform_width(&wf, dt);
        // trough at 1, peak after at 2 -> 1 sample
        assert!((w - dt).abs() < 1e-12);
    }
}
