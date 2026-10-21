// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — WaveformCodec kernels (spike detection, templates, background, lossless)

//! Native kernels of `sc_neurocore.spike_codec.waveform_codec` (stream format `WFC2`).
//!
//! Every kernel reproduces the Python reference bit for bit, so both backends
//! write identical streams: float32 operations are the same IEEE operations in
//! the same order (no fused multiply-add), rounding is half to even, dot
//! products are correctly rounded sums of exact float64 products (CPython's
//! `math.fsum` algorithm), and the lossless coder makes the same integer
//! choices with the same tie-breaking. The byte compressors (zstd or zlib) and
//! the optional wavelet step stay in Python.

/// Largest predictor order of the lossless coder.
pub const LOSSLESS_MAX_ORDER: usize = 3;
/// Residuals per Rice block.
pub const LOSSLESS_BLOCK: usize = 256;
/// Largest Rice parameter.
pub const LOSSLESS_MAX_RICE_K: u32 = 24;
/// Unary length that marks an escaped (verbatim 32-bit) residual.
pub const LOSSLESS_ESCAPE: u64 = 32;

/// Background downsampling factor of the `full` mode.
pub const BACKGROUND_DOWNSAMPLE: usize = 16;

/// Per-channel negative spike thresholds from the median absolute sample.
///
/// For each channel of row-major `(t, n)` data: `sigma = max(median(|x|) /
/// 0.6745, 1e-6)` and `threshold = negative_sigma_factor * sigma`, all in
/// float32. An even count takes the float32 mean of the two middle values,
/// as NumPy's median does.
pub fn noise_thresholds(
    waveform: &[f32],
    t: usize,
    n: usize,
    negative_sigma_factor: f32,
) -> Vec<f32> {
    let mut column = vec![0.0_f32; t];
    (0..n)
        .map(|ch| {
            for (time, value) in column.iter_mut().enumerate() {
                *value = waveform[time * n + ch].abs();
            }
            let middle = t / 2;
            let (_, upper, _) = column.select_nth_unstable_by(middle, f32::total_cmp);
            let upper = *upper;
            let median = if t % 2 == 1 {
                upper
            } else {
                let lower = column[..middle]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                (lower + upper) / 2.0
            };
            let sigma = (median / 0.6745_f32).max(1e-6_f32);
            negative_sigma_factor * sigma
        })
        .collect()
}

/// Zero every spike window and average the background over blocks of 16 rows.
///
/// A window covers `[max(0, time - half), min(t, time + half))`. With at
/// least 16 rows the result has `t / 16` rows, each the float32 sum of its 16
/// rows in order divided by 16; shorter inputs are returned whole. Returns
/// the row count and the row-major `(rows, n)` values.
pub fn extract_background(
    waveform: &[f32],
    t: usize,
    n: usize,
    times_per_channel: &[Vec<usize>],
    half: usize,
) -> (usize, Vec<f32>) {
    let mut background = waveform.to_vec();
    for (ch, times) in times_per_channel.iter().enumerate() {
        for &time in times {
            let start = time.saturating_sub(half);
            let end = t.min(time + half);
            for row in start..end {
                background[row * n + ch] = 0.0;
            }
        }
    }
    let ds = BACKGROUND_DOWNSAMPLE;
    if t < ds {
        return (t, background);
    }
    let rows = t / ds;
    let mut averaged = vec![0.0_f32; rows * n];
    for row in 0..rows {
        let out = &mut averaged[row * n..(row + 1) * n];
        out.copy_from_slice(&background[row * ds * n..(row * ds + 1) * n]);
        for k in 1..ds {
            let source = &background[(row * ds + k) * n..(row * ds + k + 1) * n];
            for (acc, value) in out.iter_mut().zip(source) {
                *acc += value;
            }
        }
        for value in out.iter_mut() {
            *value /= ds as f32;
        }
    }
    (rows, averaged)
}

/// Detect negative threshold crossings with a refractory period.
///
/// `waveform` is row-major `(t, n)`. A spike at `(time, ch)` needs
/// `waveform[time, ch] < thresholds[ch]`, a falling sample, `time >= 1` and more
/// than `refractory` samples since the channel's previous spike.
///
/// Returns the spike times of every channel in increasing order.
pub fn detect_spikes(
    waveform: &[f32],
    t: usize,
    n: usize,
    thresholds: &[f32],
    refractory: usize,
) -> Vec<Vec<usize>> {
    let mut times = vec![Vec::new(); n];
    for (ch, channel_times) in times.iter_mut().enumerate() {
        let mut last: i64 = -(refractory as i64) - 1;
        for time in 1..t {
            let value = waveform[time * n + ch];
            if value < thresholds[ch]
                && value < waveform[(time - 1) * n + ch]
                && (time as i64 - last) > refractory as i64
            {
                channel_times.push(time);
                last = time as i64;
            }
        }
    }
    times
}

/// Correctly rounded sum of `values` (CPython `math.fsum`, finite inputs).
pub fn exact_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for value in values {
        let mut x = value;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let mut n = partials.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

/// Correctly rounded dot product of two float32 vectors (exact float64 products).
pub fn exact_dot(a: &[f32], b: &[f32]) -> f64 {
    exact_sum(a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)))
}

/// Template library built by [`template_match`].
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateMatch {
    /// Index of the snippet that became each template, in library order.
    pub template_sources: Vec<usize>,
    /// Template used for each snippet.
    pub template_ids: Vec<usize>,
    /// Template scale of each snippet (1 unless matched above the threshold).
    pub scales: Vec<f64>,
    /// Row-major `(snippets, samples)` residuals, float32.
    pub residuals: Vec<f32>,
}

/// Build the template library and code every snippet as `scale * template + residual`.
///
/// `snippets` is row-major `(m, samples)`. The first snippet is template 0. A
/// later snippet takes the best-correlated template when the correlation
/// reaches `threshold` (scale = dot / |template|²), otherwise becomes a new
/// template while fewer than `max_templates` exist, otherwise uses the best
/// template (or template 0 if none correlates) with scale 1.
pub fn template_match(
    snippets: &[f32],
    m: usize,
    samples: usize,
    max_templates: usize,
    threshold: f64,
) -> TemplateMatch {
    let row = |i: usize| &snippets[i * samples..(i + 1) * samples];
    let mut result = TemplateMatch {
        template_sources: Vec::new(),
        template_ids: Vec::with_capacity(m),
        scales: Vec::with_capacity(m),
        residuals: vec![0.0; m * samples],
    };
    if m == 0 {
        return result;
    }
    result.template_sources.push(0);
    result.template_ids.push(0);
    result.scales.push(1.0);
    for i in 1..m {
        let s = row(i);
        let norm_s = exact_dot(s, s).sqrt();
        let mut best_corr = -1.0_f64;
        let mut best_idx: i64 = -1;
        for (j, &source) in result.template_sources.iter().enumerate() {
            let tmpl = row(source);
            let norm_t = exact_dot(tmpl, tmpl).sqrt();
            if norm_s > 1e-6 && norm_t > 1e-6 {
                let corr = exact_dot(s, tmpl) / (norm_s * norm_t);
                if corr > best_corr {
                    best_corr = corr;
                    best_idx = j as i64;
                }
            }
        }
        let residual = &mut result.residuals[i * samples..(i + 1) * samples];
        if best_corr >= threshold {
            let best = best_idx as usize;
            let tmpl = row(result.template_sources[best]);
            let scale = exact_dot(s, tmpl) / exact_dot(tmpl, tmpl).max(1e-10);
            let scale32 = scale as f32;
            for k in 0..samples {
                residual[k] = s[k] - scale32 * tmpl[k];
            }
            result.template_ids.push(best);
            result.scales.push(scale);
        } else if result.template_sources.len() < max_templates {
            result.template_sources.push(i);
            result.template_ids.push(result.template_sources.len() - 1);
            result.scales.push(1.0);
        } else {
            let fallback = best_idx.max(0) as usize;
            let tmpl = row(result.template_sources[fallback]);
            for k in 0..samples {
                residual[k] = s[k] - tmpl[k];
            }
            result.template_ids.push(fallback);
            result.scales.push(1.0);
        }
    }
    result
}

/// Closed-loop delta quantisation of a row-major `(rows, cols)` background.
///
/// Row 0 is sent separately (its codes are 0). Each later row is quantised
/// against the running reconstruction: `q = clip(round_half_even((x - recon) /
/// step), -half, half - 1)`, `recon += q * step`, all in float32.
pub fn quantize_background(
    background: &[f32],
    rows: usize,
    cols: usize,
    step: f32,
    half: i32,
) -> Vec<i8> {
    let mut codes = vec![0_i8; rows * cols];
    if rows == 0 {
        return codes;
    }
    let mut recon: Vec<f32> = background[..cols].to_vec();
    let low = -(half as f32);
    let high = (half - 1) as f32;
    for t in 1..rows {
        for c in 0..cols {
            let ratio = (background[t * cols + c] - recon[c]) / step;
            let q = ratio.round_ties_even().clamp(low, high);
            codes[t * cols + c] = q as i8;
            recon[c] += q * step;
        }
    }
    codes
}

/// Rebuild the background rows from their first row and closed-loop codes.
pub fn reconstruct_background(
    codes: &[i8],
    rows: usize,
    cols: usize,
    first: &[f32],
    step: f32,
) -> Vec<f32> {
    let mut recon = vec![0.0_f32; rows * cols];
    if rows == 0 {
        return recon;
    }
    recon[..cols].copy_from_slice(first);
    for t in 1..rows {
        for c in 0..cols {
            recon[t * cols + c] = recon[(t - 1) * cols + c] + f32::from(codes[t * cols + c]) * step;
        }
    }
    recon
}

fn zigzag(value: i64) -> u64 {
    if value >= 0 {
        (value as u64) << 1
    } else {
        ((-value as u64) << 1) - 1
    }
}

fn unzigzag(value: u64) -> i64 {
    if value.is_multiple_of(2) {
        (value / 2) as i64
    } else {
        -(value.div_ceil(2) as i64)
    }
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value & 0x7F) as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut value = 0_u64;
    let mut shift = 0_u32;
    while pos < data.len() {
        let byte = data[pos];
        pos += 1;
        if shift < 64 {
            value |= u64::from(byte & 0x7F) << shift;
        }
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (value, pos)
}

fn diff(values: &[i64]) -> Vec<i64> {
    values.windows(2).map(|w| w[1] - w[0]).collect()
}

fn diff_n(values: &[i64], order: usize) -> Vec<i64> {
    let mut current = values.to_vec();
    for _ in 0..order {
        current = diff(&current);
    }
    current
}

fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0_u8; bits.len().div_ceil(8)];
    for (i, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            out[i / 8] |= 0x80 >> (i % 8);
        }
    }
    out
}

fn unpack_bits(bytes: &[u8], count: usize) -> Vec<u8> {
    (0..count)
        .map(|i| (bytes[i / 8] >> (7 - i % 8)) & 1)
        .collect()
}

/// Block-adaptive Rice coding of non-negative integers (three bit streams).
pub fn rice_encode(u: &[u64]) -> Vec<u8> {
    let n = u.len();
    let n_blocks = n.div_ceil(LOSSLESS_BLOCK);
    let mut ks = Vec::with_capacity(n_blocks);
    for block in u.chunks(LOSSLESS_BLOCK) {
        let mut best_k = 0_u32;
        let mut best_cost = u64::MAX;
        for k in 0..=LOSSLESS_MAX_RICE_K {
            let cost: u64 = block
                .iter()
                .map(|&value| {
                    let q = value >> k;
                    if q < LOSSLESS_ESCAPE {
                        q + 1 + u64::from(k)
                    } else {
                        LOSSLESS_ESCAPE + 1 + 32
                    }
                })
                .sum();
            if cost < best_cost {
                best_cost = cost;
                best_k = k;
            }
        }
        ks.push(best_k as u8);
    }
    let mut unary = Vec::new();
    let mut remainder = Vec::new();
    let mut escapes = Vec::new();
    for (i, &value) in u.iter().enumerate() {
        let k = u32::from(ks[i / LOSSLESS_BLOCK]);
        let q = value >> k;
        if q < LOSSLESS_ESCAPE {
            unary.extend(std::iter::repeat_n(1_u8, q as usize));
            unary.push(0);
            for bit in (0..k).rev() {
                remainder.push(((value >> bit) & 1) as u8);
            }
        } else {
            unary.extend(std::iter::repeat_n(1_u8, LOSSLESS_ESCAPE as usize));
            unary.push(0);
            escapes.push(value as u32);
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(unary.len() as u32).to_be_bytes());
    out.extend_from_slice(&(remainder.len() as u32).to_be_bytes());
    out.extend_from_slice(&(escapes.len() as u32).to_be_bytes());
    out.extend_from_slice(&ks);
    out.extend_from_slice(&pack_bits(&unary));
    out.extend_from_slice(&pack_bits(&remainder));
    for value in escapes {
        out.extend_from_slice(&value.to_be_bytes());
    }
    out
}

fn be_u32(data: &[u8], pos: usize) -> Result<u32, String> {
    data.get(pos..pos + 4)
        .map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .ok_or_else(|| "malformed WaveformCodec stream: section too short".to_string())
}

/// Invert [`rice_encode`] for `n` values, refusing inconsistent sections.
pub fn rice_decode(data: &[u8], n: usize) -> Result<Vec<u64>, String> {
    let n_unary = be_u32(data, 0)? as usize;
    let n_rem = be_u32(data, 4)? as usize;
    let n_esc = be_u32(data, 8)? as usize;
    let n_blocks = n.div_ceil(LOSSLESS_BLOCK);
    let unary_bytes = n_unary.div_ceil(8);
    let rem_bytes = n_rem.div_ceil(8);
    let ks_at = 12;
    let unary_at = ks_at + n_blocks;
    let rem_at = unary_at + unary_bytes;
    let esc_at = rem_at + rem_bytes;
    if esc_at + 4 * n_esc != data.len() {
        return Err("lossless Rice section has the wrong length".into());
    }
    let ks = &data[ks_at..unary_at];
    if ks.iter().any(|&k| u32::from(k) > LOSSLESS_MAX_RICE_K) {
        return Err("lossless Rice parameter out of range".into());
    }
    let unary = unpack_bits(&data[unary_at..rem_at], n_unary);
    let mut quotients = Vec::with_capacity(n);
    let mut run = 0_u64;
    for bit in unary {
        if bit == 0 {
            quotients.push(run);
            run = 0;
        } else {
            run += 1;
        }
    }
    if quotients.len() != n {
        return Err("lossless Rice section has the wrong number of values".into());
    }
    let widths: Vec<u32> = quotients
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            if q < LOSSLESS_ESCAPE {
                u32::from(ks[i / LOSSLESS_BLOCK])
            } else {
                0
            }
        })
        .collect();
    let escaped = quotients.iter().filter(|&&q| q >= LOSSLESS_ESCAPE).count();
    if widths.iter().map(|&w| w as usize).sum::<usize>() != n_rem || escaped != n_esc {
        return Err("lossless Rice section is inconsistent".into());
    }
    let rem_bits = unpack_bits(&data[rem_at..esc_at], n_rem);
    let mut bit_pos = 0;
    let mut esc_pos = esc_at;
    let mut values = Vec::with_capacity(n);
    for (i, &q) in quotients.iter().enumerate() {
        if q < LOSSLESS_ESCAPE {
            let k = widths[i];
            let mut remainder = 0_u64;
            for _ in 0..k {
                remainder = (remainder << 1) | u64::from(rem_bits[bit_pos]);
                bit_pos += 1;
            }
            values.push((q << k) | remainder);
        } else {
            values.push(u64::from(be_u32(data, esc_pos)?));
            esc_pos += 4;
        }
    }
    Ok(values)
}

fn encode_levels(levels: &[i64]) -> Vec<u8> {
    let max_order = LOSSLESS_MAX_ORDER.min(levels.len() - 1);
    let mut order = 0;
    let mut best = i64::MAX;
    for k in 0..=max_order {
        let total: i64 = diff_n(levels, k).iter().map(|v| v.abs()).sum();
        if total < best {
            best = total;
            order = k;
        }
    }
    let mut out = vec![order as u8];
    for j in 0..order {
        push_varint(&mut out, zigzag(diff_n(levels, j)[0]));
    }
    let residuals: Vec<u64> = diff_n(levels, order).into_iter().map(zigzag).collect();
    out.extend(rice_encode(&residuals));
    out
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Encode one channel of integer samples exactly (shorter of the affine and table mappings).
pub fn lossless_encode_channel(x: &[i64]) -> Vec<u8> {
    let offset = *x.iter().min().expect("a channel has at least one sample");
    let step = x.iter().fold(0, |g, &v| gcd(g, v - offset)).max(1);
    let mut affine = vec![0_u8];
    affine.extend_from_slice(&(offset as i32).to_be_bytes());
    affine.extend_from_slice(&(step as u32).to_be_bytes());
    let affine_levels: Vec<i64> = x.iter().map(|&v| (v - offset) / step).collect();
    affine.extend(encode_levels(&affine_levels));

    let mut table: Vec<i64> = x.to_vec();
    table.sort_unstable();
    table.dedup();
    let mut indexed = vec![1_u8];
    indexed.extend_from_slice(&(table.len() as u32).to_be_bytes());
    let mut previous = 0;
    for &value in &table {
        push_varint(&mut indexed, zigzag(value - previous));
        previous = value;
    }
    let table_levels: Vec<i64> = x
        .iter()
        .map(|v| {
            table
                .binary_search(v)
                .expect("every sample is in its table") as i64
        })
        .collect();
    indexed.extend(encode_levels(&table_levels));
    if indexed.len() < affine.len() {
        indexed
    } else {
        affine
    }
}

/// Decode one channel written by [`lossless_encode_channel`] into `t` samples.
pub fn lossless_decode_channel(section: &[u8], t: usize) -> Result<Vec<i64>, String> {
    let kind = *section
        .first()
        .ok_or_else(|| "malformed WaveformCodec stream: empty lossless channel".to_string())?;
    let mut table = Vec::new();
    let (mut offset, mut step) = (0_i64, 1_i64);
    let mut pos;
    match kind {
        1 => {
            let n_table = be_u32(section, 1)? as usize;
            pos = 5;
            if n_table < 1 || n_table > section.len() - pos {
                return Err("lossless value table length is impossible for its section".into());
            }
            let mut value = 0_i64;
            for _ in 0..n_table {
                let (code, next) = read_varint(section, pos);
                pos = next;
                value += unzigzag(code);
                table.push(value);
            }
        }
        0 => {
            offset = i64::from(be_u32(section, 1)? as i32);
            step = i64::from(be_u32(section, 5)?);
            pos = 9;
        }
        _ => return Err("invalid lossless level mapping".into()),
    }
    let order = *section
        .get(pos)
        .ok_or_else(|| "malformed WaveformCodec stream: missing predictor order".to_string())?
        as usize;
    pos += 1;
    if order > LOSSLESS_MAX_ORDER || order >= t.max(1) {
        return Err("invalid lossless predictor order".into());
    }
    let mut firsts = Vec::with_capacity(order);
    for _ in 0..order {
        let (code, next) = read_varint(section, pos);
        pos = next;
        firsts.push(unzigzag(code));
    }
    let mut current: Vec<i64> = rice_decode(&section[pos..], t - order)?
        .into_iter()
        .map(unzigzag)
        .collect();
    for &first in firsts.iter().rev() {
        let mut next = Vec::with_capacity(current.len() + 1);
        next.push(first);
        let mut acc = first;
        for value in current {
            acc += value;
            next.push(acc);
        }
        current = next;
    }
    let samples: Vec<i64> = if kind == 1 {
        if current
            .iter()
            .any(|&level| level < 0 || level as usize >= table.len())
        {
            return Err("lossless level outside the value table".into());
        }
        current.iter().map(|&level| table[level as usize]).collect()
    } else {
        current.iter().map(|&level| level * step + offset).collect()
    };
    if samples.iter().any(|&v| !(-32768..=32767).contains(&v)) {
        return Err("lossless samples outside the int16 range".into());
    }
    Ok(samples)
}
