// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Streaming lossless waveform coder (ADC levels, sign-sign LMS, binary range coder)

//! Stateful lossless coder for multi-channel int16 electrode streams.
//!
//! Each sample is mapped to its ADC level `round((x + 32768) * (2^bits - 1) / 65535)`; the
//! exact 16-bit value of a level is sent once, as a deviation from the level's nominal value,
//! the first time the level occurs in a channel group.
//!
//! Levels are numbered densely: each group counts how often every level occurs, and a level
//! far rarer than both neighbours (an ADC missing code) is taken out of the numbering, so a
//! signal crossing it moves by one rank instead of two. Both sides apply the same rule to the
//! same counts; a sample that does hit an excluded level is sent as a reserved escape plus the
//! level in `bits` plain bits.
//!
//! The first difference of the level ranks is predicted by a sign-sign LMS filter of order 16
//! on earlier differences (32-bit weights clamped to ±2^15, history clamped to ±2^11, so the
//! filter vectorises), and the zigzagged prediction error is coded with an adaptive binary
//! tree (values below 63) plus an adaptive Elias-gamma escape, through an LZMA-style binary
//! range coder with 12-bit probabilities.
//!
//! Every operation is integer arithmetic, so the Python reference produces identical bytes.
//! State persists across windows: a window (packet) costs only its range-coder flush, so a
//! 1 ms window compresses as well as a whole recording. Channels split into contiguous groups;
//! each group has its own range coder and level table and can be coded in parallel.

use rayon::prelude::*;

/// Probability precision of the binary models (bits).
pub const PROB_BITS: u32 = 12;
/// Adaptation shift of the binary models.
pub const ADAPT_SHIFT: u32 = 6;
/// Order of the sign-sign LMS predictor.
pub const LMS_ORDER: usize = 16;
/// Fixed-point shift of the LMS weights.
pub const LMS_SHIFT: u32 = 12;
/// LMS weight step.
pub const LMS_STEP: i32 = 2;
/// LMS weights are clamped to this range.
pub const WEIGHT_MIN: i32 = -32768;
/// LMS weights are clamped to this range.
pub const WEIGHT_MAX: i32 = 32767;
/// Differences enter the LMS history clamped to this range (prediction only; coding is exact).
pub const HISTORY_MIN: i64 = -2048;
/// Differences enter the LMS history clamped to this range (prediction only; coding is exact).
pub const HISTORY_MAX: i64 = 2047;
/// Residuals at or above this value use the escape code.
pub const TREE_ESCAPE: u64 = 63;
/// Longest escape length the decoder accepts.
pub const MAX_ESCAPE_BITS: usize = 48;
/// A level is excluded once `count * EXCLUDE_RATIO` is below the rarer neighbour's count…
pub const EXCLUDE_RATIO: u64 = 64;
/// …and that neighbour has occurred at least this often.
pub const EXCLUDE_MIN_COUNT: u32 = 8;
/// An excluded level returns once `count * INCLUDE_RATIO` reaches the rarer neighbour's count.
pub const INCLUDE_RATIO: u64 = 16;
/// When a level count reaches this value, every count of the group is halved (the numbering
/// then follows slow changes of the signal).
pub const COUNT_LIMIT: u32 = 1 << 16;

const PROB_INIT: u16 = 1 << (PROB_BITS - 1);
const TOP: u32 = 1 << 24;
const OVERRUN: &str = "corrupted lossless packet: range coder ran past its bytes";

// Probabilities start at 2048 and `adapt` keeps them in [63, 4033], so after coding a bit (or
// halving the range for a plain bit) the range is at least 2^24 * 63 / 4096 > 2^16 and one
// 8-bit renormalisation step restores `range >= TOP`.

/// Adapt a probability towards the coded bit (`mask` is all ones for a 1, zero for a 0).
#[inline(always)]
fn adapt(prob: u32, mask: u32) -> u16 {
    let up = ((1_u32 << PROB_BITS) - prob) >> ADAPT_SHIFT;
    let down = prob >> ADAPT_SHIFT;
    (prob + (up & !mask) - (down & mask)) as u16
}

struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: u32::MAX,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low as u32) >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = u64::from((self.low as u32) << 8);
    }

    #[inline(always)]
    fn normalize(&mut self) {
        if self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
        debug_assert!(self.range >= TOP);
    }

    /// Code one adaptive bit without data-dependent branches (masks instead of jumps).
    #[inline(always)]
    fn bit(&mut self, prob: &mut u16, bit: bool) {
        let p = u32::from(*prob);
        let bound = (self.range >> PROB_BITS) * p;
        let mask = u32::from(bit).wrapping_neg();
        self.low += u64::from(bound & mask);
        self.range = (bound & !mask) | ((self.range - bound) & mask);
        *prob = adapt(p, mask);
        self.normalize();
    }

    fn direct(&mut self, value: u64, count: u32) {
        for i in (0..count).rev() {
            self.range >>= 1;
            if (value >> i) & 1 == 1 {
                self.low += u64::from(self.range);
            }
            self.normalize();
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

/// Binary range decoder. Reading never fails: past the end of its bytes it reads zeros and sets
/// `overrun`, which the caller checks after every sample and before reporting any other error,
/// so the hot path carries no error values (the reported error equals the reference's).
struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
    overrun: bool,
}

impl<'a> RangeDecoder<'a> {
    /// `data` has passed the packet framing checks (at least 5 bytes, first byte zero).
    fn new(data: &'a [u8]) -> Self {
        let mut decoder = RangeDecoder {
            data,
            pos: 0,
            range: u32::MAX,
            code: 0,
            overrun: false,
        };
        for _ in 0..5 {
            decoder.code = (decoder.code << 8) | u32::from(decoder.next());
        }
        decoder
    }

    #[inline(always)]
    fn next(&mut self) -> u8 {
        let byte = match self.data.get(self.pos) {
            Some(&byte) => byte,
            None => {
                self.overrun = true;
                0
            }
        };
        self.pos += 1;
        byte
    }

    #[inline(always)]
    fn normalize(&mut self) {
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next());
        }
        debug_assert!(self.range >= TOP);
    }

    /// Decode one adaptive bit without data-dependent branches.
    #[inline(always)]
    fn bit(&mut self, prob: &mut u16) -> bool {
        let p = u32::from(*prob);
        let bound = (self.range >> PROB_BITS) * p;
        let bit = self.code >= bound;
        let mask = u32::from(bit).wrapping_neg();
        self.code -= bound & mask;
        self.range = (bound & !mask) | ((self.range - bound) & mask);
        *prob = adapt(p, mask);
        self.normalize();
        bit
    }

    fn direct(&mut self, count: u32) -> u64 {
        let mut value = 0_u64;
        for _ in 0..count {
            self.range >>= 1;
            let bit = self.code >= self.range;
            if bit {
                self.code -= self.range;
            }
            value = (value << 1) | u64::from(bit);
            self.normalize();
        }
        value
    }

    /// The error to report: an overrun takes precedence, because the reference stops at the
    /// first byte read past the end.
    fn error(&self, message: String) -> String {
        if self.overrun {
            OVERRUN.to_string()
        } else {
            message
        }
    }

    fn finish(self) -> Result<(), String> {
        if self.overrun {
            return Err(OVERRUN.into());
        }
        if self.pos != self.data.len() {
            return Err("corrupted lossless packet: trailing range-coder bytes".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
struct ResidualModel {
    tree: [u16; 64],
    escape: [u16; MAX_ESCAPE_BITS],
}

impl ResidualModel {
    fn new() -> Self {
        ResidualModel {
            tree: [PROB_INIT; 64],
            escape: [PROB_INIT; MAX_ESCAPE_BITS],
        }
    }

    /// Code `u`: values below 63 through the tree, larger ones as the escape symbol plus the
    /// Elias-gamma code of `u - 61` (escape value 1 is reserved, see [`Self::put_marker`]).
    #[inline(always)]
    fn put(&mut self, encoder: &mut RangeEncoder, u: u64) {
        let symbol = u.min(TREE_ESCAPE);
        self.put_symbol(encoder, symbol);
        if symbol == TREE_ESCAPE {
            self.put_gamma(encoder, u - TREE_ESCAPE + 2);
        }
    }

    /// Code the reserved escape value 1 (the next item is not a residual).
    fn put_marker(&mut self, encoder: &mut RangeEncoder) {
        self.put_symbol(encoder, TREE_ESCAPE);
        self.put_gamma(encoder, 1);
    }

    #[inline(always)]
    fn put_symbol(&mut self, encoder: &mut RangeEncoder, symbol: u64) {
        let mut node = 1_usize;
        for i in (0..6).rev() {
            let bit = (symbol >> i) & 1 == 1;
            encoder.bit(&mut self.tree[node], bit);
            node = node * 2 + usize::from(bit);
        }
    }

    fn put_gamma(&mut self, encoder: &mut RangeEncoder, value: u64) {
        let length = (64 - value.leading_zeros()) as usize;
        for i in 0..length - 1 {
            encoder.bit(&mut self.escape[i], true);
        }
        encoder.bit(&mut self.escape[length - 1], false);
        encoder.direct(value, (length - 1) as u32);
    }

    /// Decode one item: `Some(u)` for a residual, `None` for the reserved marker.
    #[inline(always)]
    fn get(&mut self, decoder: &mut RangeDecoder<'_>) -> Result<Option<u64>, String> {
        let mut node = 1_usize;
        for _ in 0..6 {
            node = node * 2 + usize::from(decoder.bit(&mut self.tree[node]));
        }
        let symbol = (node - 64) as u64;
        if symbol < TREE_ESCAPE {
            return Ok(Some(symbol));
        }
        let mut length = 1;
        while decoder.bit(&mut self.escape[length - 1]) {
            length += 1;
            if length > MAX_ESCAPE_BITS {
                return Err("corrupted lossless packet: escape too long".into());
            }
        }
        let value = (1_u64 << (length - 1)) | decoder.direct((length - 1) as u32);
        Ok((value > 1).then(|| value + TREE_ESCAPE - 2))
    }

    /// Decode a residual; the reserved marker is refused (the deviation model never sends it).
    fn get_residual(&mut self, decoder: &mut RangeDecoder<'_>) -> Result<u64, String> {
        self.get(decoder)?
            .ok_or_else(|| "corrupted lossless packet: unexpected level marker".to_string())
    }
}

fn zigzag(value: i64) -> u64 {
    if value >= 0 {
        (value as u64) << 1
    } else {
        ((value.unsigned_abs()) << 1) - 1
    }
}

fn unzigzag(value: u64) -> i64 {
    if value & 1 == 0 {
        (value >> 1) as i64
    } else {
        -(((value >> 1) + 1) as i64)
    }
}

/// ADC level of an int16 sample for an ADC of `bits` bits scaled to the int16 range.
pub fn level_of(sample: i64, bits: u32) -> i64 {
    let numerator = (sample + 32768) * ((1_i64 << bits) - 1);
    (2 * numerator + 65535) / 131_070
}

/// Nominal int16 value of an ADC level (inverse scaling, rounded half up).
pub fn nominal_value(level: i64, bits: u32) -> i64 {
    let scale = (1_i64 << bits) - 1;
    (2 * level * 65535 + scale) / (2 * scale) - 32768
}

/// Smallest ADC width in `1..=16` whose level mapping is one-to-one on `samples`.
///
/// Width 16 always qualifies (the mapping is `x + 32768`).
pub fn detect_adc_bits(samples: &[i16]) -> u32 {
    for bits in 1..16 {
        let mut table: Vec<Option<i16>> = vec![None; 1 << bits];
        let consistent = samples.iter().all(|&x| {
            let level = level_of(i64::from(x), bits) as usize;
            match table[level] {
                None => {
                    table[level] = Some(x);
                    true
                }
                Some(known) => known == x,
            }
        });
        if consistent {
            return bits;
        }
    }
    16
}

#[derive(Clone)]
struct ChannelState {
    previous: i64,
    /// Twice-written ring of past differences clamped to the history range:
    /// `ring[head..head + LMS_ORDER]` is newest first. Weights and history fit 16 bits, so
    /// the dot product maps to 16-bit multiply-add instructions.
    ring: [i16; 2 * LMS_ORDER],
    /// `LMS_STEP * sign` of the ring entries (stored once; the update then needs no multiply).
    steps: [i16; 2 * LMS_ORDER],
    head: usize,
    weights: [i16; LMS_ORDER],
    model: ResidualModel,
}

impl ChannelState {
    fn new(bits: u32) -> Self {
        ChannelState {
            previous: 1_i64 << (bits - 1),
            ring: [0; 2 * LMS_ORDER],
            steps: [0; 2 * LMS_ORDER],
            head: 0,
            weights: [0; LMS_ORDER],
            model: ResidualModel::new(),
        }
    }

    #[inline(always)]
    fn window(ring: &[i16; 2 * LMS_ORDER], head: usize) -> [i16; LMS_ORDER] {
        let mut out = [0_i16; LMS_ORDER];
        out.copy_from_slice(&ring[head..head + LMS_ORDER]);
        out
    }

    #[inline(always)]
    fn predict(&self) -> i64 {
        let history = Self::window(&self.ring, self.head);
        // |weight| <= 2^15 and |history| <= 2^11, so 16 products stay below 2^31.
        let mut acc = 0_i32;
        for i in 0..LMS_ORDER {
            acc += i32::from(self.weights[i]) * i32::from(history[i]);
        }
        i64::from(acc >> LMS_SHIFT)
    }

    #[inline(always)]
    fn update(&mut self, difference: i64, error: i64) {
        if error != 0 {
            // Add +steps for a positive error, -steps for a negative one; weights saturate
            // at the i16 range, which is exactly [WEIGHT_MIN, WEIGHT_MAX].
            let negate: i16 = if error > 0 { 0 } else { -1 };
            let steps = Self::window(&self.steps, self.head);
            for i in 0..LMS_ORDER {
                self.weights[i] =
                    self.weights[i].saturating_add((steps[i] ^ negate).wrapping_sub(negate));
            }
        }
        // Newest first: step the head back by one and write the value into both halves.
        self.head = if self.head == 0 {
            LMS_ORDER - 1
        } else {
            self.head - 1
        };
        let value = difference.clamp(HISTORY_MIN, HISTORY_MAX) as i16;
        let step = LMS_STEP as i16 * value.signum();
        self.ring[self.head] = value;
        self.ring[self.head + LMS_ORDER] = value;
        self.steps[self.head] = step;
        self.steps[self.head + LMS_ORDER] = step;
    }
}

/// Marks an ADC level whose exact value has not been sent yet.
const UNSET: i32 = i32::MIN;

/// Dense numbering of a group's ADC levels with adaptive exclusion of missing codes.
#[derive(Clone)]
struct LevelRanks {
    /// Occurrences of every level in the group (halved together at `COUNT_LIMIT`).
    count: Vec<u32>,
    excluded: Vec<bool>,
    /// Number of included levels below every level (an excluded level shares the rank of
    /// the next included one).
    rank: Vec<i32>,
    /// The included levels in ascending order (the inverse of `rank`).
    level_at: Vec<u32>,
}

impl LevelRanks {
    fn new(bits: u32) -> Self {
        let levels = 1_usize << bits;
        LevelRanks {
            count: vec![0; levels],
            excluded: vec![false; levels],
            rank: (0..levels as i32).collect(),
            level_at: (0..levels as u32).collect(),
        }
    }

    /// Whether `level` should be excluded, given whether it is now (hysteresis). The lowest
    /// and highest levels are never excluded, so every rank maps to an included level.
    fn exclusion(&self, level: usize, now: bool) -> bool {
        if level == 0 || level + 1 >= self.count.len() {
            return false;
        }
        let rarer = u64::from(self.count[level - 1].min(self.count[level + 1]));
        let own = u64::from(self.count[level]);
        if now {
            own * INCLUDE_RATIO < rarer
        } else {
            rarer >= u64::from(EXCLUDE_MIN_COUNT) && own * EXCLUDE_RATIO < rarer
        }
    }

    /// Count one occurrence of `level` and update the numbering (identically on both sides).
    ///
    /// A level's status depends only on its own and its neighbours' counts, so after `level`
    /// grows only two changes are possible: `level` itself may return (if excluded), and an
    /// included neighbour may leave (which needs `neighbour * EXCLUDE_RATIO < count[level]`).
    /// Checking just these gives the same numbering as re-evaluating all three levels.
    #[inline]
    fn observe(&mut self, level: usize) {
        self.count[level] += 1;
        let own = self.count[level];
        let mut changed = false;
        if own >= COUNT_LIMIT {
            for count in &mut self.count {
                *count >>= 1;
            }
            for candidate in 0..self.count.len() {
                let now = self.excluded[candidate];
                let next = self.exclusion(candidate, now);
                changed |= next != now;
                self.excluded[candidate] = next;
            }
        } else {
            if self.excluded[level] && !self.exclusion(level, true) {
                self.excluded[level] = false;
                changed = true;
            }
            let limit = u64::from(own);
            let lower = level.wrapping_sub(1);
            for neighbour in [lower, level + 1] {
                if neighbour < self.count.len()
                    && !self.excluded[neighbour]
                    && u64::from(self.count[neighbour]) * EXCLUDE_RATIO < limit
                    && self.exclusion(neighbour, false)
                {
                    self.excluded[neighbour] = true;
                    changed = true;
                }
            }
        }
        if changed {
            self.level_at.clear();
            for (level, excluded) in self.excluded.iter().enumerate() {
                self.rank[level] = self.level_at.len() as i32;
                if !excluded {
                    self.level_at.push(level as u32);
                }
            }
        }
    }
}

#[derive(Clone)]
struct GroupState {
    channels: Vec<ChannelState>,
    /// Exact int16 value of every ADC level seen so far, or `UNSET`.
    table: Vec<i32>,
    /// Scratch for the encoder's validation pass (always all `UNSET` between windows).
    pending: Vec<i32>,
    deviation: ResidualModel,
    ranks: LevelRanks,
}

impl GroupState {
    fn new(channels: usize, bits: u32) -> Self {
        GroupState {
            channels: vec![ChannelState::new(bits); channels],
            table: vec![UNSET; 1 << bits],
            pending: vec![UNSET; 1 << bits],
            deviation: ResidualModel::new(),
            ranks: LevelRanks::new(bits),
        }
    }
}

fn group_bounds(channels: usize, groups: usize) -> Vec<(usize, usize)> {
    (0..groups)
        .map(|g| (g * channels / groups, (g + 1) * channels / groups))
        .collect()
}

fn validate_config(channels: usize, bits: u32, groups: usize) -> Result<(), String> {
    if channels == 0 {
        return Err("a lossless stream needs at least one channel".into());
    }
    if !(1..=16).contains(&bits) {
        return Err("adc_bits must be 1..16".into());
    }
    if groups == 0 || groups > channels || groups > 65535 {
        return Err("groups must be 1..min(channels, 65535)".into());
    }
    Ok(())
}

/// Stateful encoder of one lossless stream.
pub struct StreamEncoder {
    channels: usize,
    bits: u32,
    /// ADC level of every int16 value, indexed by `value + 32768`.
    levels: Vec<u16>,
    groups: Vec<(usize, usize, GroupState)>,
}

impl StreamEncoder {
    /// New encoder for `channels` channels of a `bits`-bit ADC, split into `groups` groups.
    pub fn new(channels: usize, bits: u32, groups: usize) -> Result<Self, String> {
        validate_config(channels, bits, groups)?;
        let groups = group_bounds(channels, groups)
            .into_iter()
            .map(|(start, end)| (start, end, GroupState::new(end - start, bits)))
            .collect();
        let levels = (-32768_i64..=32767)
            .map(|value| level_of(value, bits) as u16)
            .collect();
        Ok(StreamEncoder {
            channels,
            bits,
            levels,
            groups,
        })
    }

    /// Encode one row-major `(samples, channels)` window into a packet.
    ///
    /// Within a group the samples are coded channel by channel (all samples of a channel in
    /// window order), which keeps one channel's state in cache. The packet holds, per group, a
    /// big-endian u32 length and the range-coder bytes. A sample that contradicts the group's
    /// level table (the input is not a `bits`-bit ADC stream) is found in a validation pass
    /// before any state changes; the encoder then returns an error and stays unchanged.
    pub fn encode(&mut self, window: &[i16], samples: usize) -> Result<Vec<u8>, String> {
        if window.len() != samples * self.channels {
            return Err("window size does not match samples x channels".into());
        }
        let (bits, channels) = (self.bits, self.channels);
        let levels = &self.levels;
        let level_index = |sample: i16| levels[(i32::from(sample) + 32768) as usize] as usize;
        let conflicts: Vec<Option<String>> = self
            .groups
            .par_iter_mut()
            .map(|(start, end, state)| {
                let mut touched = Vec::new();
                let mut conflict = None;
                'scan: for channel in *start..*end {
                    for row in 0..samples {
                        let sample = window[row * channels + channel];
                        let level = level_index(sample);
                        let known = if state.table[level] == UNSET { state.pending[level] } else { state.table[level] };
                        if known == UNSET {
                            state.pending[level] = i32::from(sample);
                            touched.push(level);
                        } else if known != i32::from(sample) {
                            conflict = Some(format!(
                                "sample {sample} in channel {channel} contradicts {known} at {bits}-bit ADC level {level}"
                            ));
                            break 'scan;
                        }
                    }
                }
                for level in touched {
                    state.pending[level] = UNSET;
                }
                conflict
            })
            .collect();
        if let Some(message) = conflicts.into_iter().flatten().next() {
            return Err(message);
        }
        let bodies: Vec<Vec<u8>> = self
            .groups
            .par_iter_mut()
            .map(|(start, _end, state)| {
                let mut encoder = RangeEncoder::new();
                let GroupState {
                    channels: states,
                    table,
                    deviation,
                    ranks,
                    ..
                } = state;
                for (offset, channel) in states.iter_mut().enumerate() {
                    for row in 0..samples {
                        let sample = window[row * channels + *start + offset];
                        let value = i64::from(sample);
                        let level = level_index(sample);
                        let difference =
                            i64::from(ranks.rank[level] - ranks.rank[channel.previous as usize]);
                        let error = difference - channel.predict();
                        if ranks.excluded[level] {
                            channel.model.put_marker(&mut encoder);
                            encoder.direct(level as u64, bits);
                        } else {
                            channel.model.put(&mut encoder, zigzag(error));
                        }
                        channel.update(difference, error);
                        channel.previous = level as i64;
                        if table[level] == UNSET {
                            deviation.put(
                                &mut encoder,
                                zigzag(value - nominal_value(level as i64, bits)),
                            );
                            table[level] = i32::from(sample);
                        }
                        ranks.observe(level);
                    }
                }
                encoder.finish()
            })
            .collect();
        let mut packet = Vec::with_capacity(bodies.iter().map(|b| b.len() + 4).sum());
        for body in bodies {
            packet.extend_from_slice(&(body.len() as u32).to_be_bytes());
            packet.extend_from_slice(&body);
        }
        Ok(packet)
    }
}

/// Decode one sample of `channel` (the decoder's overrun flag is checked by the caller).
#[inline(always)]
fn decode_sample(
    channel: &mut ChannelState,
    ranks: &mut LevelRanks,
    table: &mut [i32],
    deviation: &mut ResidualModel,
    decoder: &mut RangeDecoder<'_>,
    bits: u32,
) -> Result<i16, String> {
    let previous_rank = i64::from(ranks.rank[channel.previous as usize]);
    let prediction = channel.predict();
    let (level, difference, error) = match channel.model.get(decoder)? {
        Some(u) => {
            let error = unzigzag(u);
            let difference = prediction + error;
            let level = usize::try_from(previous_rank + difference)
                .ok()
                .and_then(|rank| ranks.level_at.get(rank))
                .ok_or_else(|| "corrupted lossless packet: ADC level out of range".to_string())?;
            (*level as usize, difference, error)
        }
        None => {
            // `bits` plain bits are always a valid level index.
            let level = decoder.direct(bits) as usize;
            if !ranks.excluded[level] {
                return Err("corrupted lossless packet: escaped level is not excluded".into());
            }
            let difference = i64::from(ranks.rank[level]) - previous_rank;
            (level, difference, difference - prediction)
        }
    };
    channel.update(difference, error);
    channel.previous = level as i64;
    let known = table[level];
    let value = if known != UNSET {
        known
    } else {
        let value = nominal_value(level as i64, bits) + unzigzag(deviation.get_residual(decoder)?);
        if !(-32768..=32767).contains(&value) {
            return Err("corrupted lossless packet: sample outside int16".into());
        }
        table[level] = value as i32;
        value as i32
    };
    ranks.observe(level);
    Ok(value as i16)
}

/// Stateful decoder of one lossless stream.
pub struct StreamDecoder {
    channels: usize,
    bits: u32,
    groups: Vec<(usize, usize, GroupState)>,
    broken: bool,
}

impl StreamDecoder {
    /// New decoder with the encoder's configuration.
    pub fn new(channels: usize, bits: u32, groups: usize) -> Result<Self, String> {
        validate_config(channels, bits, groups)?;
        let groups = group_bounds(channels, groups)
            .into_iter()
            .map(|(start, end)| (start, end, GroupState::new(end - start, bits)))
            .collect();
        Ok(StreamDecoder {
            channels,
            bits,
            groups,
            broken: false,
        })
    }

    /// Decode a packet of `samples` rows into a row-major `(samples, channels)` window.
    ///
    /// Packet framing (group lengths, trailing bytes) is checked before any state changes,
    /// and such a packet is refused with the decoder unchanged. A packet whose coded content
    /// turns out invalid while decoding leaves the stream state inconsistent: the decoder
    /// refuses it and every later packet, and a new decoder must be started (as for any
    /// stateful stream codec after a lost packet).
    pub fn decode(&mut self, packet: &[u8], samples: usize) -> Result<Vec<i16>, String> {
        if self.broken {
            return Err("lossless stream decoder is broken by an earlier corrupted packet; start a new decoder".into());
        }
        let mut slices = Vec::with_capacity(self.groups.len());
        let mut pos = 0;
        for _ in 0..self.groups.len() {
            let header = packet
                .get(pos..pos + 4)
                .ok_or_else(|| "corrupted lossless packet: missing group length".to_string())?;
            let length = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
            let body = packet
                .get(pos + 4..pos + 4 + length)
                .ok_or_else(|| "corrupted lossless packet: group longer than packet".to_string())?;
            if body.len() < 5 || body[0] != 0 {
                return Err("corrupted lossless packet: bad range-coder start".into());
            }
            slices.push(body);
            pos += 4 + length;
        }
        if pos != packet.len() {
            return Err("corrupted lossless packet: trailing bytes".into());
        }
        let (bits, channels) = (self.bits, self.channels);
        let mut window = vec![0_i16; samples * channels];
        let results: Vec<Result<Vec<i16>, String>> = self
            .groups
            .par_iter_mut()
            .zip(slices.par_iter())
            .map(|((start, end, state), body)| {
                let width = *end - *start;
                let mut values = vec![0_i16; samples * width];
                let mut decoder = RangeDecoder::new(body);
                let GroupState {
                    channels: states,
                    table,
                    deviation,
                    ranks,
                    ..
                } = state;
                for (offset, channel) in states.iter_mut().enumerate() {
                    for row in 0..samples {
                        let value =
                            decode_sample(channel, ranks, table, deviation, &mut decoder, bits)
                                .map_err(|message| decoder.error(message))?;
                        if decoder.overrun {
                            return Err(OVERRUN.into());
                        }
                        values[row * width + offset] = value;
                    }
                }
                decoder.finish()?;
                Ok(values)
            })
            .collect();
        for ((start, end, _), result) in self.groups.iter().zip(results) {
            let values = match result {
                Ok(values) => values,
                Err(message) => {
                    self.broken = true;
                    return Err(message);
                }
            };
            let width = end - start;
            for row in 0..samples {
                window[row * channels + start..row * channels + end]
                    .copy_from_slice(&values[row * width..(row + 1) * width]);
            }
        }
        Ok(window)
    }
}
