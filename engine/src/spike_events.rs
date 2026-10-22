// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-event layer: threshold crossings and spike-band power for narrow links

//! Spike-event layer for narrow links such as Bluetooth.
//!
//! Instead of the waveform, the encoder sends what BCI decoders use: threshold-crossing events
//! in short bins (1 ms by default) and, optionally, spike-band power. Everything is integer
//! arithmetic, so the Python reference produces identical bytes and the method fits implant
//! firmware:
//!
//! 1. Each channel passes a second-order Butterworth high-pass (default 250 Hz) with Q14
//!    coefficients.
//! 2. The noise level is a streaming estimate of the median of `|y|` (Q4 fixed point,
//!    multiplicative steps shrinking from 1/32 to 1/4096 during the warm-up, 1/4096 after),
//!    and the threshold is `threshold_sigma` times `median / 0.6745`, at least one unit.
//! 3. An event is a downward crossing of the threshold after the warm-up, at least
//!    `refractory_samples` after the channel's previous event (so at most one per bin).
//! 4. Spike-band power is the mean of `|z|` over `power_bins` bins, where `z` is the
//!    high-passed signal through a second-order low-pass (default 1000 Hz), shifted right by
//!    `power_shift` bits.
//!
//! Each packet carries, per channel group, the bins completed by the window: one adaptive
//! binary decision per channel and bin (a 16-bit probability per channel), the sample
//! offset of each event inside its bin when `sample_timing` is set, and the band-power values
//! completed by the window as zigzagged differences. A bin left incomplete at the end of a
//! window is sent with the next packet.

use rayon::prelude::*;

use super::waveform_stream::{
    group_bounds, unzigzag, zigzag, RangeDecoder, RangeEncoder, ResidualModel, PROB_INIT,
};

/// Fraction bits of the filter coefficients.
pub const COEFFICIENT_BITS: u32 = 14;
/// Fraction bits of the noise estimate.
pub const NOISE_BITS: u32 = 4;
/// Fraction bits of the threshold factor.
pub const THRESHOLD_BITS: u32 = 12;
/// Step shift of the noise estimate at the start of the warm-up.
pub const WARMUP_STEP_SHIFT: u32 = 5;
/// Step shift of the noise estimate at the end of the warm-up and after it.
pub const NOISE_STEP_SHIFT: u32 = 12;
/// Longest warm-up in samples.
pub const MAX_WARMUP_SAMPLES: u64 = 1 << 40;
/// Largest bin length in samples.
pub const MAX_BIN_SAMPLES: usize = 1024;
/// Adaptation shift of the per-channel 16-bit event probability.
pub const EVENT_ADAPT_SHIFT: u32 = 4;
/// Initial event probability state (one half in 1/65536).
const EVENT_PROB_INIT: u16 = 1 << 15;
/// Filter outputs are clamped to this magnitude (keeps all arithmetic far from overflow).
pub const SIGNAL_LIMIT: i64 = 1 << 24;
/// Median of `|x|` divided by this is the noise standard deviation of Gaussian noise.
pub const MAD_TO_SIGMA: f64 = 0.6745;

const PACKET: &str = "corrupted spike-event packet";

/// A biquad's Q14 coefficients (`a0` is one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Biquad {
    /// Numerator coefficient of the current input.
    pub b0: i64,
    /// Numerator coefficient of the previous input.
    pub b1: i64,
    /// Numerator coefficient of the input two samples back.
    pub b2: i64,
    /// Denominator coefficient of the previous output.
    pub a1: i64,
    /// Denominator coefficient of the output two samples back.
    pub a2: i64,
}

fn q14(value: f64) -> i64 {
    (value * f64::from(1_u32 << COEFFICIENT_BITS) + 0.5).floor() as i64
}

fn butterworth(cutoff_hz: f64, rate_hz: f64, highpass: bool) -> Result<Biquad, String> {
    if !(cutoff_hz.is_finite() && rate_hz.is_finite())
        || cutoff_hz < rate_hz / 1000.0
        || cutoff_hz >= 0.45 * rate_hz
    {
        return Err(format!(
            "filter cutoff {cutoff_hz} Hz must lie in [rate/1000, 0.45 * rate) for rate {rate_hz} Hz"
        ));
    }
    let k = (std::f64::consts::PI * cutoff_hz / rate_hz).tan();
    let norm = 1.0 / (1.0 + std::f64::consts::SQRT_2 * k + k * k);
    let (b0, b1, b2) = if highpass {
        (norm, -2.0 * norm, norm)
    } else {
        let b0 = k * k * norm;
        (b0, 2.0 * b0, b0)
    };
    let coefficients = Biquad {
        b0: q14(b0),
        b1: q14(b1),
        b2: q14(b2),
        a1: q14(2.0 * (k * k - 1.0) * norm),
        a2: q14((1.0 - std::f64::consts::SQRT_2 * k + k * k) * norm),
    };
    // The quantised filter must keep its gain and its poles inside the stability triangle
    // (|a2| < 1, |a1| < 1 + a2).
    let one = 1_i64 << COEFFICIENT_BITS;
    if coefficients.b0 == 0
        || coefficients.a2.abs() >= one
        || coefficients.a1.abs() >= one + coefficients.a2
    {
        return Err(format!(
            "filter cutoff {cutoff_hz} Hz at {rate_hz} Hz cannot be represented in Q14"
        ));
    }
    Ok(coefficients)
}

/// Q14 coefficients of the second-order Butterworth high-pass.
pub fn highpass(cutoff_hz: f64, rate_hz: f64) -> Result<Biquad, String> {
    butterworth(cutoff_hz, rate_hz, true)
}

/// Q14 coefficients of the second-order Butterworth low-pass.
pub fn lowpass(cutoff_hz: f64, rate_hz: f64) -> Result<Biquad, String> {
    butterworth(cutoff_hz, rate_hz, false)
}

/// Threshold factor `threshold_sigma / 0.6745` in Q12.
pub fn threshold_factor(threshold_sigma: f64) -> i64 {
    (threshold_sigma / MAD_TO_SIGMA * f64::from(1_u32 << THRESHOLD_BITS) + 0.5).floor() as i64
}

#[derive(Clone, Default)]
struct BiquadState {
    x1: i64,
    x2: i64,
    y1: i64,
    y2: i64,
}

impl BiquadState {
    #[inline(always)]
    fn step(&mut self, c: &Biquad, x: i64) -> i64 {
        let acc = c.b0 * x + c.b1 * self.x1 + c.b2 * self.x2 - c.a1 * self.y1 - c.a2 * self.y2;
        let y = ((acc + (1 << (COEFFICIENT_BITS - 1))) >> COEFFICIENT_BITS)
            .clamp(-SIGNAL_LIMIT, SIGNAL_LIMIT);
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Encoder settings (the decoder needs only `channels`, `bin_samples`, `sample_timing`,
/// `power_bins` and `groups`).
#[derive(Clone, Debug)]
pub struct SpikeEventConfig {
    /// Channels per window.
    pub channels: usize,
    /// Sampling rate of the raw stream.
    pub sample_rate_hz: f64,
    /// Samples per event bin (1..=1024).
    pub bin_samples: usize,
    /// Minimum distance between two events of a channel (`bin_samples`..=2^20).
    pub refractory_samples: usize,
    /// Threshold in noise standard deviations, in (0, 100].
    pub threshold_sigma: f64,
    /// Cutoff of the detection high-pass.
    pub highpass_hz: f64,
    /// Upper cutoff of the spike band for the band power.
    pub lowpass_hz: f64,
    /// Samples before the first event may be reported (at most 2^40).
    pub warmup_samples: u64,
    /// Send each event's sample offset inside its bin.
    pub sample_timing: bool,
    /// Bins per band-power value (0 disables band power, at most 65535).
    pub power_bins: usize,
    /// Right shift of the band-power values (0..=15).
    pub power_shift: u32,
    /// Contiguous channel groups, each with its own range coder (1..=min(channels, 65535)).
    pub groups: usize,
}

fn validate_layout(
    channels: usize,
    bin_samples: usize,
    power_bins: usize,
    groups: usize,
) -> Result<(), String> {
    if channels == 0 {
        return Err("a spike-event stream needs at least one channel".into());
    }
    if !(1..=MAX_BIN_SAMPLES).contains(&bin_samples) {
        return Err(format!("bin_samples must be 1..{MAX_BIN_SAMPLES}"));
    }
    if power_bins > 65535 {
        return Err("power_bins must be 0..65535".into());
    }
    if groups == 0 || groups > channels || groups > 65535 {
        return Err("groups must be 1..min(channels, 65535)".into());
    }
    Ok(())
}

/// Bits of a sample offset inside a bin.
fn offset_bits(bin_samples: usize) -> u32 {
    usize::BITS - (bin_samples - 1).leading_zeros()
}

/// Per-channel coding state shared by encoder and decoder.
#[derive(Clone)]
struct EventModel {
    prob: u16,
    previous_power: i64,
}

impl EventModel {
    fn new() -> Self {
        EventModel {
            prob: EVENT_PROB_INIT,
            previous_power: 0,
        }
    }
}

/// Per-group coding state shared by encoder and decoder.
#[derive(Clone)]
struct GroupModel {
    channels: Vec<EventModel>,
    offsets: Vec<u16>,
    power: ResidualModel,
}

impl GroupModel {
    fn new(channels: usize, bin_samples: usize) -> Self {
        GroupModel {
            channels: vec![EventModel::new(); channels],
            offsets: vec![PROB_INIT; 1 << offset_bits(bin_samples)],
            power: ResidualModel::new(),
        }
    }
}

/// Per-channel detector state (encoder only).
#[derive(Clone)]
struct Detector {
    highpass: BiquadState,
    lowpass: BiquadState,
    noise: i64,
    below: bool,
    since_event: u64,
    power_sum: i64,
    /// An event in the bin the previous window left incomplete.
    carried: Option<u64>,
}

/// Stateful spike-event encoder.
pub struct SpikeEventEncoder {
    config: SpikeEventConfig,
    highpass: Biquad,
    lowpass: Biquad,
    factor: i64,
    groups: Vec<(usize, usize, GroupModel, Vec<Detector>)>,
    samples_seen: u64,
}

impl SpikeEventEncoder {
    /// New encoder; refuses settings outside the documented ranges.
    pub fn new(config: SpikeEventConfig) -> Result<Self, String> {
        validate_layout(
            config.channels,
            config.bin_samples,
            config.power_bins,
            config.groups,
        )?;
        if !(config.sample_rate_hz.is_finite() && config.sample_rate_hz > 0.0) {
            return Err("sample_rate_hz must be positive".into());
        }
        if config.refractory_samples < config.bin_samples || config.refractory_samples > 1 << 20 {
            return Err("refractory_samples must be bin_samples..2^20".into());
        }
        if !(config.threshold_sigma.is_finite()
            && config.threshold_sigma > 0.0
            && config.threshold_sigma <= 100.0)
        {
            return Err("threshold_sigma must be in (0, 100]".into());
        }
        if config.warmup_samples > MAX_WARMUP_SAMPLES {
            return Err("warmup_samples must be 0..2^40".into());
        }
        if config.power_shift > 15 {
            return Err("power_shift must be 0..15".into());
        }
        let highpass = highpass(config.highpass_hz, config.sample_rate_hz)?;
        let lowpass = lowpass(config.lowpass_hz, config.sample_rate_hz)?;
        let detector = Detector {
            highpass: BiquadState::default(),
            lowpass: BiquadState::default(),
            noise: 0,
            below: false,
            since_event: config.refractory_samples as u64,
            power_sum: 0,
            carried: None,
        };
        let groups = group_bounds(config.channels, config.groups)
            .into_iter()
            .map(|(start, end)| {
                (
                    start,
                    end,
                    GroupModel::new(end - start, config.bin_samples),
                    vec![detector.clone(); end - start],
                )
            })
            .collect();
        Ok(SpikeEventEncoder {
            factor: threshold_factor(config.threshold_sigma),
            config,
            highpass,
            lowpass,
            groups,
            samples_seen: 0,
        })
    }

    /// Encode one row-major `(samples, channels)` int16 window into a packet.
    ///
    /// The packet holds, per group, a big-endian u32 length and the range-coder bytes.
    pub fn encode(&mut self, window: &[i16], samples: usize) -> Result<Vec<u8>, String> {
        let channels = self.config.channels;
        if samples.checked_mul(channels) != Some(window.len()) {
            return Err("window size does not match samples x channels".into());
        }
        let config = &self.config;
        let (highpass, lowpass, factor) = (&self.highpass, &self.lowpass, self.factor);
        let start = self.samples_seen;
        let bin = config.bin_samples as u64;
        let period = (config.power_bins as u64) * bin;
        let first_bin = start / bin;
        let end_bin = (start + samples as u64) / bin;
        let bodies: Vec<Vec<u8>> = self
            .groups
            .par_iter_mut()
            .map(|(first, _end, model, detectors)| {
                let mut encoder = RangeEncoder::new();
                let mut events = Vec::new();
                let mut powers = Vec::new();
                for (offset, (detector, channel)) in detectors
                    .iter_mut()
                    .zip(model.channels.iter_mut())
                    .enumerate()
                {
                    events.clear();
                    powers.clear();
                    events.extend(detector.carried.take());
                    for row in 0..samples {
                        let t = start + row as u64;
                        let x = i64::from(window[row * channels + *first + offset]);
                        let y = detector.highpass.step(highpass, x);
                        let threshold =
                            ((detector.noise * factor) >> THRESHOLD_BITS).max(1 << NOISE_BITS);
                        let below = (y << NOISE_BITS) < -threshold;
                        if below
                            && !detector.below
                            && detector.since_event >= config.refractory_samples as u64
                            && t >= config.warmup_samples
                        {
                            events.push(t);
                            detector.since_event = 0;
                        }
                        detector.below = below;
                        detector.since_event = detector.since_event.saturating_add(1);
                        let magnitude = y.abs() << NOISE_BITS;
                        let shift = noise_step_shift(t, config.warmup_samples);
                        let step = (detector.noise >> shift).max(1);
                        if magnitude > detector.noise {
                            detector.noise += step;
                        } else if magnitude < detector.noise {
                            detector.noise -= step;
                        }
                        if period > 0 {
                            detector.power_sum += detector.lowpass.step(lowpass, y).abs();
                            if (t + 1).is_multiple_of(period) {
                                powers.push(
                                    (detector.power_sum / period as i64) >> config.power_shift,
                                );
                                detector.power_sum = 0;
                            }
                        }
                    }
                    let mut pending = events.iter().copied().peekable();
                    for index in first_bin..end_bin {
                        let event = match pending.peek() {
                            Some(&t) if t / bin == index => {
                                pending.next();
                                Some(t)
                            }
                            _ => None,
                        };
                        encoder.bit_precise(&mut channel.prob, event.is_some(), EVENT_ADAPT_SHIFT);
                        if let (Some(t), true) = (event, config.sample_timing) {
                            put_offset(
                                &mut encoder,
                                &mut model.offsets,
                                t % bin,
                                config.bin_samples,
                            );
                        }
                    }
                    detector.carried = pending.next();
                    for &value in &powers {
                        model
                            .power
                            .put(&mut encoder, zigzag(value - channel.previous_power));
                        channel.previous_power = value;
                    }
                }
                encoder.finish()
            })
            .collect();
        self.samples_seen += samples as u64;
        let mut packet = Vec::with_capacity(bodies.iter().map(|b| b.len() + 4).sum());
        for body in bodies {
            packet.extend_from_slice(&(body.len() as u32).to_be_bytes());
            packet.extend_from_slice(&body);
        }
        Ok(packet)
    }
}

/// The step shrinks from 1/32 to 1/4096 across the warm-up, then stays at 1/4096.
fn noise_step_shift(t: u64, warmup: u64) -> u32 {
    if t >= warmup {
        return NOISE_STEP_SHIFT;
    }
    let ramp = (t * u64::from(NOISE_STEP_SHIFT - WARMUP_STEP_SHIFT + 1)) / warmup;
    (WARMUP_STEP_SHIFT + ramp as u32).min(NOISE_STEP_SHIFT)
}

fn put_offset(encoder: &mut RangeEncoder, tree: &mut [u16], offset: u64, bin_samples: usize) {
    let mut node = 1_usize;
    for i in (0..offset_bits(bin_samples)).rev() {
        let bit = (offset >> i) & 1 == 1;
        encoder.bit(&mut tree[node], bit);
        node = node * 2 + usize::from(bit);
    }
}

/// Events and band power decoded from one packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpikeEventWindow {
    /// Absolute index of the first bin in `events`.
    pub first_bin: u64,
    /// Row-major `(bins, channels)`: 1 where the channel has an event in the bin.
    pub events: Vec<u8>,
    /// Row-major `(bins, channels)` sample offsets inside the bin, -1 without an event
    /// (empty unless `sample_timing`).
    pub offsets: Vec<i16>,
    /// Absolute index of the first band-power period in `power`.
    pub first_period: u64,
    /// Row-major `(periods, channels)` band-power values.
    pub power: Vec<i32>,
}

/// Stateful spike-event decoder.
pub struct SpikeEventDecoder {
    channels: usize,
    bin_samples: usize,
    sample_timing: bool,
    power_bins: usize,
    groups: Vec<(usize, usize, GroupModel)>,
    samples_seen: u64,
    broken: bool,
}

struct GroupResult {
    events: Vec<u8>,
    offsets: Vec<i16>,
    power: Vec<i32>,
}

impl SpikeEventDecoder {
    /// New decoder with the encoder's layout settings.
    pub fn new(
        channels: usize,
        bin_samples: usize,
        sample_timing: bool,
        power_bins: usize,
        groups: usize,
    ) -> Result<Self, String> {
        validate_layout(channels, bin_samples, power_bins, groups)?;
        let groups = group_bounds(channels, groups)
            .into_iter()
            .map(|(start, end)| (start, end, GroupModel::new(end - start, bin_samples)))
            .collect();
        Ok(SpikeEventDecoder {
            channels,
            bin_samples,
            sample_timing,
            power_bins,
            groups,
            samples_seen: 0,
            broken: false,
        })
    }

    /// Decode the packet of a window of `samples` samples.
    ///
    /// Framing errors leave the decoder unchanged; content found invalid while decoding
    /// breaks it, and every later packet is refused (start a new decoder).
    pub fn decode(&mut self, packet: &[u8], samples: usize) -> Result<SpikeEventWindow, String> {
        if self.broken {
            return Err(
                "spike-event decoder is broken by an earlier corrupted packet; start a new decoder"
                    .into(),
            );
        }
        let mut slices = Vec::with_capacity(self.groups.len());
        let mut pos = 0;
        for _ in 0..self.groups.len() {
            let header = packet
                .get(pos..pos + 4)
                .ok_or_else(|| format!("{PACKET}: missing group length"))?;
            let length = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
            let body = packet
                .get(pos + 4..pos + 4 + length)
                .ok_or_else(|| format!("{PACKET}: group longer than packet"))?;
            if body.len() < 5 || body[0] != 0 {
                return Err(format!("{PACKET}: bad range-coder start"));
            }
            slices.push(body);
            pos += 4 + length;
        }
        if pos != packet.len() {
            return Err(format!("{PACKET}: trailing bytes"));
        }
        let bin = self.bin_samples as u64;
        let period = (self.power_bins as u64) * bin;
        let too_many = || format!("{PACKET}: more samples requested than the packet can hold");
        let end = self
            .samples_seen
            .checked_add(samples as u64)
            .ok_or_else(too_many)?;
        let first_bin = self.samples_seen / bin;
        let bins = (end / bin - first_bin) as usize;
        let (first_period, periods) = match self.samples_seen.checked_div(period) {
            Some(first) => (first, (end / period - first) as usize),
            None => (0, 0),
        };
        // An event decision costs at least 0.00033 bits (probability 65521/65536) and a
        // band-power value at least 0.134 bits, so a group body of n bytes holds fewer than
        // 32768 * (n + 8) bin decisions and 64 * (n + 8) values. Larger requests are refused
        // before any allocation or state change.
        for ((start, end, _), body) in self.groups.iter().zip(&slices) {
            let (width, room) = ((end - start) as u128, body.len() as u128 + 8);
            if bins as u128 * width > 32768 * room || periods as u128 * width > 64 * room {
                return Err(too_many());
            }
        }
        let (bin_samples, sample_timing) = (self.bin_samples, self.sample_timing);
        let results: Vec<Result<GroupResult, String>> = self
            .groups
            .par_iter_mut()
            .zip(slices.par_iter())
            .map(|((start, end, model), body)| {
                decode_group(
                    model,
                    body,
                    *end - *start,
                    bins,
                    periods,
                    bin_samples,
                    sample_timing,
                )
            })
            .collect();
        let channels = self.channels;
        let mut window = SpikeEventWindow {
            first_bin,
            events: vec![0; bins * channels],
            offsets: if sample_timing {
                vec![-1; bins * channels]
            } else {
                Vec::new()
            },
            first_period,
            power: vec![0; periods * channels],
        };
        for ((start, end, _), result) in self.groups.iter().zip(results) {
            let part = match result {
                Ok(part) => part,
                Err(message) => {
                    self.broken = true;
                    return Err(message);
                }
            };
            let width = end - start;
            for row in 0..bins {
                let target = row * channels + start..row * channels + end;
                window.events[target.clone()]
                    .copy_from_slice(&part.events[row * width..(row + 1) * width]);
                if sample_timing {
                    window.offsets[target]
                        .copy_from_slice(&part.offsets[row * width..(row + 1) * width]);
                }
            }
            for row in 0..periods {
                window.power[row * channels + start..row * channels + end]
                    .copy_from_slice(&part.power[row * width..(row + 1) * width]);
            }
        }
        self.samples_seen = end;
        Ok(window)
    }
}

/// Decode one group's channels; an overrun takes precedence over any other error, as in the
/// reference, which stops at the first byte read past the end.
fn decode_group(
    model: &mut GroupModel,
    body: &[u8],
    width: usize,
    bins: usize,
    periods: usize,
    bin_samples: usize,
    sample_timing: bool,
) -> Result<GroupResult, String> {
    let mut decoder = RangeDecoder::new(body);
    let overrun = || format!("{PACKET}: range coder ran past its bytes");
    let mut part = GroupResult {
        events: vec![0; bins * width],
        offsets: if sample_timing {
            vec![-1; bins * width]
        } else {
            Vec::new()
        },
        power: vec![0; periods * width],
    };
    let bits = offset_bits(bin_samples);
    for (offset, channel) in model.channels.iter_mut().enumerate() {
        for row in 0..bins {
            let event = decoder.bit_precise(&mut channel.prob, EVENT_ADAPT_SHIFT);
            part.events[row * width + offset] = u8::from(event);
            if event && sample_timing {
                let mut node = 1_usize;
                for _ in 0..bits {
                    node = node * 2 + usize::from(decoder.bit(&mut model.offsets[node]));
                }
                let position = node - (1 << bits);
                if decoder.overrun {
                    return Err(overrun());
                }
                if position >= bin_samples {
                    return Err(format!("{PACKET}: event offset beyond its bin"));
                }
                part.offsets[row * width + offset] = position as i16;
            }
        }
        for row in 0..periods {
            let item = model.power.get(&mut decoder);
            if decoder.overrun {
                return Err(overrun());
            }
            let value = match item {
                Ok(Some(u)) => channel.previous_power + unzigzag(u),
                Ok(None) => return Err(format!("{PACKET}: unexpected marker in band power")),
                Err(_) => return Err(format!("{PACKET}: escape too long")),
            };
            if !(0..=SIGNAL_LIMIT).contains(&value) {
                return Err(format!("{PACKET}: band power out of range"));
            }
            channel.previous_power = value;
            part.power[row * width + offset] = value as i32;
        }
        if decoder.overrun {
            return Err(overrun());
        }
    }
    if !decoder.consumed_exactly() {
        return Err(format!("{PACKET}: trailing range-coder bytes"));
    }
    Ok(part)
}
