// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety acceleration for analysis

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisConfig {
    pub params: Vec<(String, f64)>,
    pub dt: f64,
    pub duration: f64,
    pub current: f64,
    pub protocol: Option<String>,
    pub frequency_hz: Option<f64>,
}

impl AnalysisConfig {
    pub fn with_param(&self, name: &str, value: f64) -> Self {
        let mut out = self.clone();
        if let Some((_, existing)) = out.params.iter_mut().find(|(key, _)| key == name) {
            *existing = value;
        } else {
            out.params.push((name.to_string(), value));
        }
        out
    }

    pub fn param(&self, name: &str) -> Option<f64> {
        self.params
            .iter()
            .find_map(|(key, value)| (key == name).then_some(*value))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimulationResult {
    pub states: Vec<(String, Vec<f64>)>,
    pub rate_hz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BifurcationSweep {
    pub param_name: String,
    pub param_values: Vec<f64>,
    pub attractors: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityEntry {
    pub param: String,
    pub sensitivity: f64,
    pub base_rate: f64,
    pub rate_minus: Option<f64>,
    pub rate_plus: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityReport {
    pub base_rate: f64,
    pub sensitivities: Vec<SensitivityEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NullclineReport {
    pub var_names: [String; 2],
    pub nullcline_0: Vec<[f64; 2]>,
    pub nullcline_1: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Heatmap2D {
    pub param_x: String,
    pub x_values: Vec<f64>,
    pub param_y: String,
    pub y_values: Vec<f64>,
    pub rates: Vec<Vec<f64>>,
    pub rate_min: f64,
    pub rate_max: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpikeTriggeredAverage {
    pub time_ms: Vec<f64>,
    pub average: Vec<f64>,
    pub n_spikes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyResponse {
    pub frequencies_hz: Vec<f64>,
    pub rates: Vec<f64>,
    pub amplitude: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionError {
    pub variable: String,
    pub max_error: f64,
    pub mean_error: f64,
    pub rms_error: f64,
    pub trace: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionComparison {
    pub float_trace: Vec<f64>,
    pub fixed_trace: Vec<f64>,
    pub error: PrecisionError,
    pub quantized_params: Vec<(String, f64)>,
}

pub fn bifurcation_sweep<F>(
    simulate_fn: F,
    base_config: &AnalysisConfig,
    param_name: &str,
    param_min: f64,
    param_max: f64,
    n_values: usize,
) -> Result<BifurcationSweep, String>
where
    F: Fn(&AnalysisConfig) -> Result<SimulationResult, String>,
{
    validate_range(param_min, param_max, n_values, "bifurcation range")?;
    if param_name.trim().is_empty() {
        return Err("param_name must not be empty".to_string());
    }
    let param_values = linspace(param_min, param_max, n_values);
    let mut attractors = Vec::with_capacity(param_values.len());
    for value in &param_values {
        let cfg = base_config.with_param(param_name, *value);
        let result = simulate_fn(&cfg)?;
        let voltage = first_state(&result)?;
        let half = &voltage[voltage.len() / 2..];
        attractors.push(extract_attractor_extrema(half));
    }
    Ok(BifurcationSweep {
        param_name: param_name.to_string(),
        param_values,
        attractors,
    })
}

pub fn sensitivity_analysis<F>(
    simulate_fn: F,
    base_config: &AnalysisConfig,
    param_names: &[&str],
    perturbation: f64,
) -> Result<SensitivityReport, String>
where
    F: Fn(&AnalysisConfig) -> Result<SimulationResult, String> + Copy,
{
    if !perturbation.is_finite() || perturbation <= 0.0 {
        return Err("perturbation must be finite and positive".to_string());
    }
    let base_rate = simulate_fn(base_config)?.rate_hz;
    let mut sensitivities = Vec::with_capacity(param_names.len());
    for name in param_names {
        let base_val = base_config.param(name).unwrap_or(0.0);
        if base_val == 0.0 {
            sensitivities.push(SensitivityEntry {
                param: (*name).to_string(),
                sensitivity: 0.0,
                base_rate,
                rate_minus: None,
                rate_plus: None,
            });
            continue;
        }
        let delta = base_val.abs() * perturbation;
        let minus_rate = simulate_fn(&base_config.with_param(name, base_val - delta))?.rate_hz;
        let plus_rate = simulate_fn(&base_config.with_param(name, base_val + delta))?.rate_hz;
        let deriv = (plus_rate - minus_rate) / (2.0 * delta);
        let scale = base_rate.max(0.1);
        sensitivities.push(SensitivityEntry {
            param: (*name).to_string(),
            sensitivity: round4(deriv.abs() * base_val.abs() / scale),
            base_rate,
            rate_minus: Some(minus_rate),
            rate_plus: Some(plus_rate),
        });
    }
    sensitivities.sort_by(|left, right| {
        right
            .sensitivity
            .partial_cmp(&left.sensitivity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(SensitivityReport {
        base_rate,
        sensitivities,
    })
}

pub fn nullclines_2d<F0, F1>(
    derivative_0: F0,
    derivative_1: F1,
    var_names: (&str, &str),
    ranges: ((f64, f64), (f64, f64)),
    grid_size: usize,
) -> Result<NullclineReport, String>
where
    F0: Fn(f64, f64) -> f64,
    F1: Fn(f64, f64) -> f64,
{
    validate_range(ranges.0 .0, ranges.0 .1, grid_size, "x nullcline range")?;
    validate_range(ranges.1 .0, ranges.1 .1, grid_size, "y nullcline range")?;
    let x = linspace(ranges.0 .0, ranges.0 .1, grid_size);
    let y = linspace(ranges.1 .0, ranges.1 .1, grid_size);
    let mut field_0 = vec![vec![0.0; grid_size]; grid_size];
    let mut field_1 = vec![vec![0.0; grid_size]; grid_size];
    for row in 0..grid_size {
        for col in 0..grid_size {
            field_0[row][col] = derivative_0(x[col], y[row]);
            field_1[row][col] = derivative_1(x[col], y[row]);
        }
    }
    Ok(NullclineReport {
        var_names: [var_names.0.to_string(), var_names.1.to_string()],
        nullcline_0: contour_points(&field_0, &x, &y, 0.0)?,
        nullcline_1: contour_points(&field_1, &x, &y, 0.0)?,
    })
}

pub fn heatmap_2d<F>(
    simulate_fn: F,
    base_config: &AnalysisConfig,
    param_x: &str,
    x_range: (f64, f64, usize),
    param_y: &str,
    y_range: (f64, f64, usize),
) -> Result<Heatmap2D, String>
where
    F: Fn(&AnalysisConfig) -> Result<SimulationResult, String> + Copy,
{
    validate_range(x_range.0, x_range.1, x_range.2, "x heatmap range")?;
    validate_range(y_range.0, y_range.1, y_range.2, "y heatmap range")?;
    let x_values = linspace(x_range.0, x_range.1, x_range.2);
    let y_values = linspace(y_range.0, y_range.1, y_range.2);
    let mut rates = vec![vec![0.0; x_values.len()]; y_values.len()];
    for (row, y_val) in y_values.iter().enumerate() {
        for (col, x_val) in x_values.iter().enumerate() {
            let cfg = base_config
                .with_param(param_x, *x_val)
                .with_param(param_y, *y_val);
            rates[row][col] = simulate_fn(&cfg)?.rate_hz;
        }
    }
    let (rate_min, rate_max) = min_max_2d(&rates)?;
    Ok(Heatmap2D {
        param_x: param_x.to_string(),
        x_values,
        param_y: param_y.to_string(),
        y_values,
        rates,
        rate_min,
        rate_max,
    })
}

pub fn spike_triggered_average(
    time: &[f64],
    voltage: &[f64],
    spikes: &[usize],
    dt: f64,
    window_ms: f64,
) -> Result<SpikeTriggeredAverage, String> {
    if time.len() != voltage.len() {
        return Err("time and voltage lengths must match".to_string());
    }
    if !dt.is_finite() || dt <= 0.0 || !window_ms.is_finite() || window_ms <= 0.0 {
        return Err("dt and window_ms must be finite and positive".to_string());
    }
    if spikes.len() < 2 {
        return Ok(SpikeTriggeredAverage {
            time_ms: Vec::new(),
            average: Vec::new(),
            n_spikes: spikes.len(),
        });
    }
    let half_win = ((window_ms / dt / 2.0) as usize).max(1);
    let width = half_win * 2;
    let mut snippets = Vec::new();
    for spike in spikes {
        if *spike >= half_win && spike + half_win < voltage.len() {
            snippets.push(voltage[*spike - half_win..*spike + half_win].to_vec());
        }
    }
    if snippets.is_empty() {
        return Ok(SpikeTriggeredAverage {
            time_ms: Vec::new(),
            average: Vec::new(),
            n_spikes: 0,
        });
    }
    let mut average = vec![0.0; width];
    for snippet in &snippets {
        for (idx, value) in snippet.iter().enumerate() {
            average[idx] += *value;
        }
    }
    for value in &mut average {
        *value /= snippets.len() as f64;
    }
    let time_ms = (0..width)
        .map(|idx| (idx as isize - half_win as isize) as f64 * dt)
        .collect();
    Ok(SpikeTriggeredAverage {
        time_ms,
        average,
        n_spikes: snippets.len(),
    })
}

pub fn frequency_response<F>(
    simulate_fn: F,
    base_config: &AnalysisConfig,
    freq_min: f64,
    freq_max: f64,
    n_freqs: usize,
    amplitude: f64,
) -> Result<FrequencyResponse, String>
where
    F: Fn(&AnalysisConfig) -> Result<SimulationResult, String> + Copy,
{
    validate_range(freq_min, freq_max, n_freqs, "frequency range")?;
    if freq_min <= 0.0 || !amplitude.is_finite() {
        return Err("freq_min must be positive and amplitude finite".to_string());
    }
    let frequencies_hz = logspace(freq_min.log10(), freq_max.log10(), n_freqs);
    let mut rates = Vec::with_capacity(frequencies_hz.len());
    for freq in &frequencies_hz {
        let mut cfg = base_config.clone();
        cfg.current = amplitude;
        cfg.protocol = Some("sine".to_string());
        cfg.frequency_hz = Some(*freq);
        rates.push(simulate_fn(&cfg)?.rate_hz);
    }
    Ok(FrequencyResponse {
        frequencies_hz,
        rates,
        amplitude,
    })
}

pub fn precision_compare(
    float_trace: &[f64],
    fixed_trace: &[f64],
    variable: &str,
    params: &[(String, f64)],
) -> Result<PrecisionComparison, String> {
    let n = float_trace.len().min(fixed_trace.len());
    if n == 0 {
        return Err("precision comparison requires non-empty traces".to_string());
    }
    let trace = (0..n)
        .map(|idx| (float_trace[idx] - fixed_trace[idx]).abs())
        .collect::<Vec<_>>();
    let max_error = trace.iter().copied().fold(0.0, f64::max);
    let mean_error = trace.iter().sum::<f64>() / trace.len() as f64;
    let rms_error = (trace.iter().map(|v| v * v).sum::<f64>() / trace.len() as f64).sqrt();
    Ok(PrecisionComparison {
        float_trace: float_trace[..n].to_vec(),
        fixed_trace: fixed_trace[..n].to_vec(),
        error: PrecisionError {
            variable: variable.to_string(),
            max_error: round6(max_error),
            mean_error: round6(mean_error),
            rms_error: round6(rms_error),
            trace,
        },
        quantized_params: params
            .iter()
            .map(|(name, value)| (name.clone(), q88(*value)))
            .collect(),
    })
}

pub fn contour_points(
    z: &[Vec<f64>],
    x_values: &[f64],
    y_values: &[f64],
    threshold: f64,
) -> Result<Vec<[f64; 2]>, String> {
    let rows = z.len();
    if rows < 2 || x_values.len() < 2 || y_values.len() < 2 {
        return Err("contour grid must be at least 2x2".to_string());
    }
    let cols = z[0].len();
    if cols < 2 || z.iter().any(|row| row.len() != cols) {
        return Err("contour grid must be rectangular".to_string());
    }
    if rows != y_values.len() || cols != x_values.len() {
        return Err("contour axes must match grid shape".to_string());
    }
    let mut points = Vec::new();
    for row in 0..rows - 1 {
        for col in 0..cols - 1 {
            let vals = [
                z[row][col],
                z[row + 1][col],
                z[row][col + 1],
                z[row + 1][col + 1],
            ];
            let min_val = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if min_val <= threshold && threshold <= max_val {
                points.push([x_values[col], y_values[row]]);
            }
        }
    }
    Ok(points)
}

pub fn q88(val: f64) -> f64 {
    if !val.is_finite() {
        return 0.0;
    }
    ((val * 256.0).round() / 256.0).clamp(-128.0, 127.996)
}

fn first_state(result: &SimulationResult) -> Result<&[f64], String> {
    result
        .states
        .first()
        .map(|(_, values)| values.as_slice())
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "simulation result must contain a non-empty state trace".to_string())
}

fn extract_attractor_extrema(values: &[f64]) -> Vec<f64> {
    if values.len() < 10 {
        return Vec::new();
    }
    let mut extrema = Vec::new();
    for idx in 1..values.len() - 1 {
        let left = values[idx] - values[idx - 1];
        let right = values[idx + 1] - values[idx];
        if left > 0.0 && right < 0.0 || left < 0.0 && right > 0.0 {
            extrema.push(round2(values[idx]));
        }
    }
    if extrema.is_empty() {
        extrema.push(round2(values.iter().sum::<f64>() / values.len() as f64));
    }
    extrema.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    extrema.dedup_by(|left, right| (*left - *right).abs() < 1e-12);
    extrema
}

fn validate_range(min: f64, max: f64, n: usize, label: &str) -> Result<(), String> {
    if !min.is_finite() || !max.is_finite() || n < 2 || min > max {
        Err(format!(
            "{label} must be finite, ordered, and have at least two points"
        ))
    } else {
        Ok(())
    }
}

fn linspace(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![min];
    }
    let step = (max - min) / (n - 1) as f64;
    (0..n).map(|idx| min + step * idx as f64).collect()
}

fn logspace(min_log10: f64, max_log10: f64, n: usize) -> Vec<f64> {
    linspace(min_log10, max_log10, n)
        .into_iter()
        .map(|value| 10.0_f64.powf(value))
        .collect()
}

fn min_max_2d(values: &[Vec<f64>]) -> Result<(f64, f64), String> {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for row in values {
        for value in row {
            if !value.is_finite() {
                return Err("heatmap values must be finite".to_string());
            }
            min_value = min_value.min(*value);
            max_value = max_value.max(*value);
        }
    }
    Ok((min_value, max_value))
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AnalysisConfig {
        AnalysisConfig {
            params: vec![("gain".to_string(), 2.0), ("bias".to_string(), 1.0)],
            dt: 0.1,
            duration: 10.0,
            current: 0.0,
            protocol: None,
            frequency_hz: None,
        }
    }

    fn simulator(cfg: &AnalysisConfig) -> Result<SimulationResult, String> {
        let gain = cfg.param("gain").unwrap_or(1.0);
        let bias = cfg.param("bias").unwrap_or(0.0);
        let freq = cfg.frequency_hz.unwrap_or(1.0);
        let states = (0..64)
            .map(|idx| {
                let t = idx as f64 * cfg.dt;
                gain * (freq * t).sin() + bias
            })
            .collect::<Vec<_>>();
        Ok(SimulationResult {
            states: vec![("v".to_string(), states)],
            rate_hz: gain * 10.0 + bias + cfg.current * 0.1 + freq,
        })
    }

    #[test]
    fn test_q88_quantizes_and_saturates() {
        assert_eq!(q88(1.234), 1.234375);
        assert_eq!(q88(200.0), 127.996);
        assert_eq!(q88(-200.0), -128.0);
    }

    #[test]
    fn test_bifurcation_and_sensitivity_use_simulation_outputs() {
        let sweep = bifurcation_sweep(simulator, &config(), "gain", 1.0, 3.0, 3).unwrap();
        assert_eq!(sweep.param_values, vec![1.0, 2.0, 3.0]);
        assert_eq!(sweep.attractors.len(), 3);
        assert!(sweep.attractors.iter().all(|values| !values.is_empty()));

        let report = sensitivity_analysis(simulator, &config(), &["gain", "missing"], 0.1).unwrap();
        assert_eq!(report.sensitivities[0].param, "gain");
        assert!(report.sensitivities[0].sensitivity > 0.0);
        assert_eq!(report.sensitivities[1].sensitivity, 0.0);
    }

    #[test]
    fn test_heatmap_frequency_and_nullclines() {
        let heatmap = heatmap_2d(
            simulator,
            &config(),
            "gain",
            (1.0, 2.0, 2),
            "bias",
            (0.0, 1.0, 2),
        )
        .unwrap();
        assert_eq!(heatmap.rates.len(), 2);
        assert_eq!(heatmap.rates[0].len(), 2);
        assert!(heatmap.rate_max > heatmap.rate_min);

        let response = frequency_response(simulator, &config(), 1.0, 100.0, 3, 10.0).unwrap();
        assert_eq!(response.frequencies_hz.len(), 3);
        assert_eq!(response.amplitude, 10.0);

        let nullclines = nullclines_2d(
            |x, _y| x,
            |_x, y| y,
            ("x", "y"),
            ((-1.0, 1.0), (-1.0, 1.0)),
            5,
        )
        .unwrap();
        assert!(!nullclines.nullcline_0.is_empty());
        assert!(!nullclines.nullcline_1.is_empty());
    }

    #[test]
    fn test_spike_triggered_average_and_precision_compare() {
        let time = (0..10).map(|idx| idx as f64).collect::<Vec<_>>();
        let voltage = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sta = spike_triggered_average(&time, &voltage, &[3, 6], 1.0, 4.0).unwrap();
        assert_eq!(sta.n_spikes, 2);
        assert_eq!(sta.average, vec![2.5, 3.5, 4.5, 5.5]);

        let comparison = precision_compare(
            &[0.0, 1.0, 2.0],
            &[0.0, 1.25, 1.5],
            "v",
            &[("tau".to_string(), 1.234)],
        )
        .unwrap();
        assert_eq!(comparison.error.max_error, 0.5);
        assert_eq!(comparison.quantized_params[0].1, 1.234375);
    }

    #[test]
    fn test_contour_points_detect_sign_crossings() {
        let z = vec![vec![-1.0, 1.0], vec![-0.5, 0.5]];
        let points = contour_points(&z, &[0.0, 1.0], &[0.0, 1.0], 0.0).unwrap();
        assert_eq!(points, vec![[0.0, 0.0]]);
    }
}
