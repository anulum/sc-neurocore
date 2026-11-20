// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Symmetric Daubechies-4 background denoising

// db4 analysis low-pass filter in convolution order. Analysis high-pass comes
// from the alternating reversed filter, and the synthesis filters are reversed.
const LOW: [f32; 8] = [
    -0.010_597_402,
    0.032_883_01,
    0.030_841_382,
    -0.187_034_82,
    -0.027_983_77,
    0.630_880_8,
    0.714_846_55,
    0.230_377_81,
];

const THRESHOLD: f32 = 3.0;

/// Run the symmetric multilevel db4 transform on each channel and hard-threshold details.
pub(super) fn denoise(values: &mut [f32], rows: usize, channels: usize) {
    let levels = (rows / (LOW.len() - 1)).checked_ilog2().unwrap_or(0);
    if levels == 0 {
        return;
    }
    let high =
        std::array::from_fn(|tap| LOW[LOW.len() - 1 - tap] * if tap % 2 == 0 { -1.0 } else { 1.0 });
    for channel in 0..channels {
        let samples = values
            .chunks_exact(channels)
            .map(|row| row[channel])
            .collect();
        let restored = denoise_channel(samples, levels, &high);
        for (row, value) in values.chunks_exact_mut(channels).zip(restored) {
            row[channel] = value;
        }
    }
}

fn denoise_channel(mut samples: Vec<f32>, levels: u32, high: &[f32; 8]) -> Vec<f32> {
    let mut details = Vec::with_capacity(levels as usize);
    for _ in 0..levels {
        let mut low_band = Vec::new();
        let mut high_band = Vec::new();
        for center in (1..samples.len() + LOW.len() - 1).step_by(2) {
            let (low, detail) = analyze(&samples, center, high);
            low_band.push(low);
            high_band.push(if detail.abs() < THRESHOLD {
                0.0
            } else {
                detail
            });
        }
        details.push((samples.len(), high_band));
        samples = low_band;
    }
    for (length, detail) in details.into_iter().rev() {
        samples = (0..length)
            .map(|index| {
                let position = index + LOW.len() - 2;
                let first = position % 2;
                let low = synthesize(position, first, &samples, &LOW);
                let high = synthesize(position, first, &detail, high);
                low + high
            })
            .collect();
    }
    samples
}

fn analyze(samples: &[f32], center: usize, high: &[f32; 8]) -> (f32, f32) {
    let mut low_value = 0.0;
    let mut high_value = 0.0;
    // Accumulate right-edge reflected taps from the boundary outwards, as
    // PyWavelets does. The float32 sum is order dependent.
    let overhang = (center + 1).saturating_sub(samples.len());
    for tap in (0..overhang).rev().chain(overhang..LOW.len()) {
        let index = reflection(center as i64 - tap as i64, samples.len());
        let sample = samples[index];
        low_value += LOW[tap] * sample;
        high_value += high[tap] * sample;
    }
    (low_value, high_value)
}

fn reflection(index: i64, samples: usize) -> usize {
    let period = 2 * samples as i64;
    let reflected = index.rem_euclid(period);
    reflected.min(period - 1 - reflected) as usize
}

fn synthesize(position: usize, first: usize, band: &[f32], filter: &[f32; 8]) -> f32 {
    let mut value = 0.0;
    for tap in (first..filter.len()).step_by(2) {
        value += filter[filter.len() - 1 - tap] * band[(position - tap) / 2];
    }
    value
}
