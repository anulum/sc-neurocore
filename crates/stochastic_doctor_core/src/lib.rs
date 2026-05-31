// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic Doctor Core (Rust FFI)
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # Stochastic Doctor Core
//!
//! High-performance bitstream-level correlation analysis for the
//! SC-NeuroCore stochastic doctor module.
//!
//! Provides C-FFI surface for:
//! - Single-pair SCC computation (`compute_scc_f64`)
//! - Batch N×N pairwise SCC matrix (`compute_scc_batch`)
//! - Precision estimation (`compute_precision_f64`)
//! - Activity histogram (`histogram_u64`)
//!
//! ## References
//! - Alaghi & Hayes, "Stochastic Circuits for Real-Time Image-Processing Applications",
//!   DAC 2013. (SCC definition)

use std::slice;

#[inline]
fn validated_len_matches(len: usize, validated_len: usize) -> bool {
    len == validated_len
}

// ---------------------------------------------------------------------------
// Chunked popcount
// ---------------------------------------------------------------------------

/// Popcount over a byte slice using u64-aligned chunks for throughput.
#[inline]
fn popcount_bytes(data: &[u8]) -> usize {
    let (prefix, chunks, suffix) = unsafe { data.align_to::<u64>() };
    let mut count = 0usize;
    for &x in prefix {
        count += x.count_ones() as usize;
    }
    for &x in chunks {
        count += x.count_ones() as usize;
    }
    for &x in suffix {
        count += x.count_ones() as usize;
    }
    count
}

/// Popcount over a u64 slice.
#[inline]
fn popcount_u64(data: &[u64]) -> u64 {
    data.iter().map(|&w| w.count_ones() as u64).sum()
}

// ---------------------------------------------------------------------------
// SCC computation (byte-level streams: each byte is 0 or 1)
// ---------------------------------------------------------------------------

/// Compute the Stochastic Cross-Correlation between two byte-level bitstreams.
///
/// Each element in `a` and `b` is 0 or 1 (one bit per byte, unpacked format).
///
/// # Safety
/// `a_ptr` and `b_ptr` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn compute_scc_f64(
    a_ptr: *const u8,
    b_ptr: *const u8,
    len: usize,
    len_validated: usize,
) -> f64 {
    if !validated_len_matches(len, len_validated) {
        return f64::NAN;
    }
    if len == 0 {
        return 0.0;
    }
    if a_ptr.is_null() || b_ptr.is_null() {
        return f64::NAN;
    }
    let a = unsafe { slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { slice::from_raw_parts(b_ptr, len) };
    scc_bytes(a, b)
}

/// Pure Rust SCC computation on byte-level (0/1) streams.
pub fn scc_bytes(a: &[u8], b: &[u8]) -> f64 {
    let len = a.len();
    assert_eq!(len, b.len(), "stream length mismatch");
    if len == 0 {
        return 0.0;
    }
    let n = len as f64;

    let pa_count = popcount_bytes(a);
    let pb_count = popcount_bytes(b);

    // AND in unpacked format: a[i] & b[i]
    let and_count: usize = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & y) as usize)
        .sum();

    let pa = pa_count as f64 / n;
    let pb = pb_count as f64 / n;
    let p_and = and_count as f64 / n;

    let numerator = p_and - (pa * pb);
    if numerator.abs() < 1e-12 {
        return 0.0;
    }

    let denominator = if numerator > 0.0 {
        pa.min(pb) - (pa * pb)
    } else {
        (pa * pb) - (pa + pb - 1.0).max(0.0)
    };

    if denominator.abs() < 1e-12 {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// SCC computation (packed u64 streams)
// ---------------------------------------------------------------------------

/// SCC on packed u64 bitstreams (matching core_engine Bitstream layout).
pub fn scc_packed(a: &[u64], b: &[u64], bit_length: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "word count mismatch");
    if bit_length == 0 {
        return 0.0;
    }
    let n = bit_length as f64;

    let pa = popcount_u64(a) as f64 / n;
    let pb = popcount_u64(b) as f64 / n;

    let and_count: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & y).count_ones() as u64)
        .sum();
    let p_and = and_count as f64 / n;

    let numerator = p_and - (pa * pb);
    if numerator.abs() < 1e-12 {
        return 0.0;
    }

    let denominator = if numerator > 0.0 {
        pa.min(pb) - (pa * pb)
    } else {
        (pa * pb) - (pa + pb - 1.0).max(0.0)
    };

    if denominator.abs() < 1e-12 {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

/// C-FFI: SCC on packed u64 bitstreams.
///
/// # Safety
/// `a_ptr` and `b_ptr` must be valid for `word_count` u64 elements.
#[no_mangle]
pub unsafe extern "C" fn compute_scc_packed(
    a_ptr: *const u64,
    b_ptr: *const u64,
    word_count: usize,
    word_count_validated: usize,
    bit_length: usize,
) -> f64 {
    if !validated_len_matches(word_count, word_count_validated) {
        return f64::NAN;
    }
    if word_count == 0 || bit_length == 0 {
        return 0.0;
    }
    if a_ptr.is_null() || b_ptr.is_null() {
        return f64::NAN;
    }
    let a = unsafe { slice::from_raw_parts(a_ptr, word_count) };
    let b = unsafe { slice::from_raw_parts(b_ptr, word_count) };
    scc_packed(a, b, bit_length)
}

// ---------------------------------------------------------------------------
// Batch SCC (N×N pairwise matrix)
// ---------------------------------------------------------------------------

/// Compute the full N×N pairwise SCC matrix for a set of byte-level bitstreams.
///
/// Input: row-major N × stream_len matrix of 0/1 bytes.
/// Output: row-major N × N SCC matrix (f64). Diagonal is 1.0 by definition.
///
/// # Safety
/// `streams_ptr` must be valid for `n * stream_len` bytes.
/// `out_ptr` must be valid for `n * n` f64 elements.
#[no_mangle]
pub unsafe extern "C" fn compute_scc_batch(
    streams_ptr: *const u8,
    n: usize,
    stream_len: usize,
    streams_len_validated: usize,
    out_ptr: *mut f64,
    out_len_validated: usize,
) -> bool {
    let Some(streams_len) = n.checked_mul(stream_len) else {
        return false;
    };
    let Some(out_len) = n.checked_mul(n) else {
        return false;
    };
    if !validated_len_matches(streams_len, streams_len_validated)
        || !validated_len_matches(out_len, out_len_validated)
    {
        return false;
    }
    if streams_len == 0 {
        return true;
    }
    if streams_ptr.is_null() || out_ptr.is_null() {
        return false;
    }
    let streams = unsafe { slice::from_raw_parts(streams_ptr, streams_len) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_len) };

    scc_batch_impl(streams, n, stream_len, out);
    true
}

/// Pure Rust batch SCC computation.
pub fn scc_batch_impl(streams: &[u8], n: usize, stream_len: usize, out: &mut [f64]) {
    assert_eq!(streams.len(), n * stream_len);
    assert_eq!(out.len(), n * n);

    for i in 0..n {
        out[i * n + i] = 1.0; // diagonal
        let a = &streams[i * stream_len..(i + 1) * stream_len];
        for j in (i + 1)..n {
            let b = &streams[j * stream_len..(j + 1) * stream_len];
            let scc_val = scc_bytes(a, b);
            out[i * n + j] = scc_val;
            out[j * n + i] = scc_val; // symmetric
        }
    }
}

// ---------------------------------------------------------------------------
// Precision estimator
// ---------------------------------------------------------------------------

/// Estimate the encoding precision of a byte-level (0/1) bitstream.
///
/// Returns the estimated probability value P = popcount / length.
/// For a well-encoded bitstream, the variance is bounded by 1/(4*N)
/// where N is the stream length.
///
/// # Safety
/// `data_ptr` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn compute_precision_f64(
    data_ptr: *const u8,
    len: usize,
    len_validated: usize,
) -> f64 {
    if !validated_len_matches(len, len_validated) {
        return f64::NAN;
    }
    if len == 0 {
        return 0.0;
    }
    if data_ptr.is_null() {
        return f64::NAN;
    }
    let data = unsafe { slice::from_raw_parts(data_ptr, len) };
    popcount_bytes(data) as f64 / len as f64
}

/// Estimate precision and variance for a packed u64 bitstream.
///
/// Returns `(probability, variance_bound)`.
pub fn precision_packed(data: &[u64], bit_length: usize) -> (f64, f64) {
    if bit_length == 0 {
        return (0.0, 0.0);
    }
    let p = popcount_u64(data) as f64 / bit_length as f64;
    let variance = p * (1.0 - p) / bit_length as f64;
    (p, variance)
}

/// C-FFI: precision estimation on packed u64 bitstream.
///
/// # Safety
/// `data_ptr` must be valid for `word_count` u64 elements.
/// `out_prob` and `out_variance` must be valid f64 pointers.
#[no_mangle]
pub unsafe extern "C" fn compute_precision_packed(
    data_ptr: *const u64,
    word_count: usize,
    word_count_validated: usize,
    bit_length: usize,
    out_prob: *mut f64,
    out_variance: *mut f64,
) -> bool {
    if !validated_len_matches(word_count, word_count_validated) {
        return false;
    }
    if out_prob.is_null() || out_variance.is_null() {
        return false;
    }
    if word_count == 0 || bit_length == 0 {
        unsafe {
            *out_prob = 0.0;
            *out_variance = 0.0;
        }
        return true;
    }
    if data_ptr.is_null() {
        return false;
    }
    let data = unsafe { slice::from_raw_parts(data_ptr, word_count) };
    let (p, v) = precision_packed(data, bit_length);
    unsafe {
        *out_prob = p;
        *out_variance = v;
    }
    true
}

// ---------------------------------------------------------------------------
// Activity histogram
// ---------------------------------------------------------------------------

/// Compute a histogram of popcount values across consecutive u64 words.
///
/// Each word's popcount (0..=64) is tallied, producing a 65-element histogram.
/// This reveals activity distribution: uniform → healthy, bimodal → saturated.
///
/// # Safety
/// `data_ptr` must be valid for `word_count` u64 elements.
/// `out_ptr` must be valid for 65 u64 elements.
#[no_mangle]
pub unsafe extern "C" fn histogram_u64(
    data_ptr: *const u64,
    word_count: usize,
    word_count_validated: usize,
    out_ptr: *mut u64,
    out_len_validated: usize,
) -> bool {
    if !validated_len_matches(word_count, word_count_validated)
        || !validated_len_matches(65, out_len_validated)
    {
        return false;
    }
    if out_ptr.is_null() {
        return false;
    }
    if word_count == 0 {
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, 65) };
        for bin in out.iter_mut() {
            *bin = 0;
        }
        return true;
    }
    if data_ptr.is_null() {
        return false;
    }
    let data = unsafe { slice::from_raw_parts(data_ptr, word_count) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, 65) };

    for bin in out.iter_mut() {
        *bin = 0;
    }
    for &word in data {
        let pc = word.count_ones() as usize;
        out[pc] += 1;
    }
    true
}

/// Pure Rust histogram computation returning a Vec.
pub fn histogram_u64_vec(data: &[u64]) -> Vec<u64> {
    let mut hist = vec![0u64; 65];
    for &word in data {
        let pc = word.count_ones() as usize;
        hist[pc] += 1;
    }
    hist
}

// ---------------------------------------------------------------------------
// Drift detector (EMA-based)
// ---------------------------------------------------------------------------

/// Exponential moving average drift detector for SCC monitoring.
///
/// Tracks the running EMA of SCC values over time. When the absolute EMA
/// exceeds the alert threshold, the detector flags a drift event.
pub struct DriftDetector {
    pub alpha: f64,
    pub threshold: f64,
    pub ema: f64,
    pub active: bool,
}

impl DriftDetector {
    /// Create a new detector.
    ///
    /// * `alpha` — EMA smoothing factor (0.0–1.0; lower = smoother)
    /// * `threshold` — absolute SCC value above which to flag drift
    pub fn new(alpha: f64, threshold: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            threshold: threshold.clamp(0.0, 1.0),
            ema: 0.0,
            active: false,
        }
    }

    /// Feed a new SCC observation. Returns `true` if drift is detected.
    pub fn observe(&mut self, scc_value: f64) -> bool {
        self.ema = self.alpha * scc_value + (1.0 - self.alpha) * self.ema;
        self.active = self.ema.abs() > self.threshold;
        self.active
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.ema = 0.0;
        self.active = false;
    }
}

// ---------------------------------------------------------------------------
// Hamming(7,4) single-error-correcting code
// ---------------------------------------------------------------------------

/// Encode a 4-bit word (`data & 0x0F`) into a 7-bit Hamming(7,4) codeword.
///
/// Layout (MSB..LSB): `p1 p2 d1 p3 d2 d3 d4` where
/// `p1 = d1 ^ d2 ^ d4`, `p2 = d1 ^ d3 ^ d4`, `p3 = d2 ^ d3 ^ d4`.
/// Matches the Python reference in `sc_neurocore.debug.sc_doctor.ScDoctor.encode_ecc`.
#[inline]
pub fn hamming74_encode(data: u32) -> u32 {
    let d1 = (data >> 3) & 1;
    let d2 = (data >> 2) & 1;
    let d3 = (data >> 1) & 1;
    let d4 = data & 1;

    let p1 = d1 ^ d2 ^ d4;
    let p2 = d1 ^ d3 ^ d4;
    let p3 = d2 ^ d3 ^ d4;

    (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4
}

/// Decode a 7-bit Hamming(7,4) codeword back to the 4-bit data word, correcting
/// a single bit error if the syndrome indicates one.
#[inline]
pub fn hamming74_decode(encoded: u32) -> u32 {
    let p1 = (encoded >> 6) & 1;
    let p2 = (encoded >> 5) & 1;
    let d1 = (encoded >> 4) & 1;
    let p3 = (encoded >> 3) & 1;
    let d2 = (encoded >> 2) & 1;
    let d3 = (encoded >> 1) & 1;
    let d4 = encoded & 1;

    let s1 = p1 ^ d1 ^ d2 ^ d4;
    let s2 = p2 ^ d1 ^ d3 ^ d4;
    let s3 = p3 ^ d2 ^ d3 ^ d4;

    let syndrome = (s3 << 2) | (s2 << 1) | s1;

    // Map the syndrome to a bit position inside the 7-bit codeword (same mapping
    // as the Python reference). `syndrome == 0` means no correction needed.
    let corrected = match syndrome {
        1 => encoded ^ (1 << 6),
        2 => encoded ^ (1 << 5),
        3 => encoded ^ (1 << 4),
        4 => encoded ^ (1 << 3),
        5 => encoded ^ (1 << 2),
        6 => encoded ^ (1 << 1),
        7 => encoded ^ 1,
        _ => encoded,
    };

    let cd1 = (corrected >> 4) & 1;
    let cd2 = (corrected >> 2) & 1;
    let cd3 = (corrected >> 1) & 1;
    let cd4 = corrected & 1;

    (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4
}

#[no_mangle]
/// # Safety
/// `data` is treated as a 4-bit unsigned value; the upper bits are ignored.
pub unsafe extern "C" fn hamming74_encode_ffi(data: u32) -> u32 {
    hamming74_encode(data)
}

#[no_mangle]
/// # Safety
/// `encoded` is treated as a 7-bit unsigned codeword.
pub unsafe extern "C" fn hamming74_decode_ffi(encoded: u32) -> u32 {
    hamming74_decode(encoded)
}

// ---------------------------------------------------------------------------
// Adaptive-length control law (ScDoctor.adapt)
// ---------------------------------------------------------------------------

/// Output of one adapt step: new bitstream length and ECC-enable flag.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptResult {
    pub new_length: u32,
    pub ecc_enabled: bool,
}

/// Adaptive SC bitstream length controller, matching
/// `sc_neurocore.debug.sc_doctor.ScDoctor.adapt` exactly.
///
/// * `current_length` — current bitstream length.
/// * `current_ecc` — current ECC state.
/// * `correlation` — latest SCC observation.
///
/// Hysteresis rules:
///   * SCC > 0.15: double length; enable ECC if the new length exceeds 2048.
///   * SCC < 0.05 and length > 256: halve length; disable ECC.
///   * otherwise: unchanged.
#[inline]
pub fn sc_doctor_adapt(current_length: u32, current_ecc: bool, correlation: f64) -> AdaptResult {
    if correlation > 0.15 {
        let new_length = current_length.saturating_mul(2);
        AdaptResult {
            new_length,
            ecc_enabled: current_ecc || new_length > 2048,
        }
    } else if correlation < 0.05 && current_length > 256 {
        AdaptResult {
            new_length: current_length / 2,
            ecc_enabled: false,
        }
    } else {
        AdaptResult {
            new_length: current_length,
            ecc_enabled: current_ecc,
        }
    }
}

#[no_mangle]
/// # Safety
/// Pure integer + float arithmetic on values; no pointer dereference.
pub unsafe extern "C" fn sc_doctor_adapt_ffi(
    current_length: u32,
    current_ecc: u32,
    correlation: f64,
    out_new_length: *mut u32,
    out_ecc_enabled: *mut u32,
) {
    let result = sc_doctor_adapt(current_length, current_ecc != 0, correlation);
    if !out_new_length.is_null() {
        *out_new_length = result.new_length;
    }
    if !out_ecc_enabled.is_null() {
        *out_ecc_enabled = u32::from(result.ecc_enabled);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SCC (byte-level) ─────────────────────────────────────────────

    #[test]
    fn scc_identical_streams() {
        let a: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 0];
        let scc = scc_bytes(&a, &a);
        assert!(
            (scc - 1.0).abs() < 1e-6,
            "identical streams should have SCC=1.0, got {scc}"
        );
    }

    #[test]
    fn scc_anticorrelated_streams() {
        let a: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let b: Vec<u8> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let scc = scc_bytes(&a, &b);
        assert!(
            (scc - (-1.0)).abs() < 1e-6,
            "anticorrelated should have SCC=-1.0, got {scc}"
        );
    }

    #[test]
    fn scc_uncorrelated() {
        let a: Vec<u8> = vec![0; 256];
        let b: Vec<u8> = vec![0; 256];
        let scc = scc_bytes(&a, &b);
        assert!(
            scc.abs() < 1e-6,
            "all-zero streams should have SCC=0, got {scc}"
        );
    }

    #[test]
    fn scc_empty_stream() {
        let scc = scc_bytes(&[], &[]);
        assert_eq!(scc, 0.0);
    }

    // ── SCC (packed u64) ─────────────────────────────────────────────

    #[test]
    fn scc_packed_identical() {
        let a = vec![0xAAAA_AAAA_AAAA_AAAAu64; 4];
        let scc = scc_packed(&a, &a, 256);
        assert!(
            (scc - 1.0).abs() < 1e-6,
            "packed identical SCC should be 1.0, got {scc}"
        );
    }

    #[test]
    fn scc_packed_anticorrelated() {
        let a = vec![0xAAAA_AAAA_AAAA_AAAAu64; 4];
        let b = vec![0x5555_5555_5555_5555u64; 4];
        let scc = scc_packed(&a, &b, 256);
        assert!(
            (scc - (-1.0)).abs() < 1e-6,
            "packed anticorrelated SCC should be -1.0, got {scc}"
        );
    }

    // ── Batch SCC ────────────────────────────────────────────────────

    #[test]
    fn scc_batch_diagonal_is_one() {
        let n = 3;
        let stream_len = 100;
        let streams: Vec<u8> = (0..n * stream_len)
            .map(|i| ((i * 7 + 3) % 2) as u8)
            .collect();
        let mut out = vec![0.0f64; n * n];
        scc_batch_impl(&streams, n, stream_len, &mut out);

        for i in 0..n {
            assert!(
                (out[i * n + i] - 1.0).abs() < 1e-6,
                "diagonal should be 1.0"
            );
        }
    }

    #[test]
    fn scc_batch_symmetric() {
        let n = 4;
        let stream_len = 200;
        let streams: Vec<u8> = (0..n * stream_len)
            .map(|i| ((i * 13 + 5) % 2) as u8)
            .collect();
        let mut out = vec![0.0f64; n * n];
        scc_batch_impl(&streams, n, stream_len, &mut out);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (out[i * n + j] - out[j * n + i]).abs() < 1e-10,
                    "SCC matrix must be symmetric"
                );
            }
        }
    }

    // ── Precision ────────────────────────────────────────────────────

    #[test]
    fn precision_half_density() {
        let data = vec![0xAAAA_AAAA_AAAA_AAAAu64; 4]; // alternating bits
        let (p, v) = precision_packed(&data, 256);
        assert!(
            (p - 0.5).abs() < 0.01,
            "alternating bits should give p≈0.5, got {p}"
        );
        assert!(v > 0.0, "variance should be positive");
        // Theoretical variance for p=0.5, N=256: 0.5*0.5/256 ≈ 0.000977
        assert!((v - 0.000977).abs() < 0.001, "variance mismatch: {v}");
    }

    #[test]
    fn precision_all_zeros() {
        let data = vec![0u64; 4];
        let (p, v) = precision_packed(&data, 256);
        assert_eq!(p, 0.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn precision_all_ones() {
        let data = vec![u64::MAX; 4];
        let (p, v) = precision_packed(&data, 256);
        assert!((p - 1.0).abs() < 1e-6);
        assert!(v.abs() < 1e-6);
    }

    // ── Histogram ────────────────────────────────────────────────────

    #[test]
    fn histogram_all_zeros() {
        let data = vec![0u64; 10];
        let hist = histogram_u64_vec(&data);
        assert_eq!(hist[0], 10, "all-zero words should count in bin 0");
        assert_eq!(hist[1..].iter().sum::<u64>(), 0);
    }

    #[test]
    fn histogram_all_ones() {
        let data = vec![u64::MAX; 5];
        let hist = histogram_u64_vec(&data);
        assert_eq!(hist[64], 5, "all-ones words should count in bin 64");
        assert_eq!(hist[..64].iter().sum::<u64>(), 0);
    }

    #[test]
    fn histogram_mixed() {
        let data = vec![0u64, u64::MAX, 0x0F0F_0F0F_0F0F_0F0Fu64];
        let hist = histogram_u64_vec(&data);
        assert_eq!(hist[0], 1);
        assert_eq!(hist[64], 1);
        assert_eq!(hist[32], 1); // 0x0F0F... has 32 bits set
    }

    // ── Drift Detector ───────────────────────────────────────────────

    #[test]
    fn drift_detector_stable() {
        let mut dd = DriftDetector::new(0.1, 0.5);
        for _ in 0..100 {
            assert!(!dd.observe(0.0), "no drift on zero observations");
        }
    }

    #[test]
    fn drift_detector_detects_correlation() {
        let mut dd = DriftDetector::new(0.1, 0.3);
        // Feed sustained high-correlation observations
        for _ in 0..50 {
            dd.observe(0.9);
        }
        assert!(dd.active, "should detect drift after sustained high SCC");
        assert!(dd.ema > 0.3);
    }

    #[test]
    fn drift_detector_reset() {
        let mut dd = DriftDetector::new(0.1, 0.3);
        for _ in 0..50 {
            dd.observe(0.9);
        }
        dd.reset();
        assert_eq!(dd.ema, 0.0);
        assert!(!dd.active);
    }

    #[test]
    fn drift_detector_negative_correlation() {
        let mut dd = DriftDetector::new(0.1, 0.3);
        for _ in 0..50 {
            dd.observe(-0.8);
        }
        assert!(dd.active, "should detect drift on sustained negative SCC");
    }

    // ── C-FFI ────────────────────────────────────────────────────────

    #[test]
    fn ffi_scc_f64() {
        let a: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let b: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let scc = unsafe { compute_scc_f64(a.as_ptr(), b.as_ptr(), a.len(), a.len()) };
        assert!((scc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ffi_precision() {
        let data: Vec<u8> = vec![1, 1, 1, 0, 0, 0, 0, 0, 1, 1]; // 5/10 = 0.5
        let p = unsafe { compute_precision_f64(data.as_ptr(), data.len(), data.len()) };
        assert!((p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ffi_histogram() {
        let data = vec![0u64, u64::MAX];
        let mut out = vec![0u64; 65];
        let ok = unsafe {
            histogram_u64(
                data.as_ptr(),
                data.len(),
                data.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert!(ok);
        assert_eq!(out[0], 1);
        assert_eq!(out[64], 1);
    }

    #[test]
    fn ffi_scc_batch() {
        let n = 2;
        let stream_len = 4;
        let streams: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let mut out = vec![0.0f64; 4];
        let ok = unsafe {
            compute_scc_batch(
                streams.as_ptr(),
                n,
                stream_len,
                streams.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert!(ok);
        assert!((out[0] - 1.0).abs() < 1e-6, "diag(0,0) should be 1.0");
        assert!((out[3] - 1.0).abs() < 1e-6, "diag(1,1) should be 1.0");
        assert!((out[1] - out[2]).abs() < 1e-10, "should be symmetric");
    }

    #[test]
    fn ffi_precision_packed() {
        let data = vec![0xAAAA_AAAA_AAAA_AAAAu64; 2];
        let mut prob = 0.0f64;
        let mut var = 0.0f64;
        let ok = unsafe { compute_precision_packed(data.as_ptr(), 2, 2, 128, &mut prob, &mut var) };
        assert!(ok);
        assert!((prob - 0.5).abs() < 0.01);
        assert!(var > 0.0);
    }

    #[test]
    fn ffi_scc_f64_len_mismatch_returns_nan() {
        let a: Vec<u8> = vec![1, 0, 1, 0];
        let b: Vec<u8> = vec![1, 0, 1, 0];
        let scc = unsafe { compute_scc_f64(a.as_ptr(), b.as_ptr(), a.len(), a.len() + 1) };
        assert!(scc.is_nan());
    }

    #[test]
    fn ffi_precision_f64_len_mismatch_returns_nan() {
        let data: Vec<u8> = vec![1, 0, 1, 0];
        let p = unsafe { compute_precision_f64(data.as_ptr(), data.len(), data.len() + 1) };
        assert!(p.is_nan());
    }

    #[test]
    fn ffi_batch_len_mismatch_returns_false() {
        let n = 2;
        let stream_len = 4;
        let streams: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let mut out = vec![0.0f64; 4];
        let ok = unsafe {
            compute_scc_batch(
                streams.as_ptr(),
                n,
                stream_len,
                streams.len() + 1,
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert!(!ok);
    }

    // ── Hamming(7,4) ──────────────────────────────────────────────────

    #[test]
    fn hamming_roundtrip_all_16_words() {
        for word in 0u32..16 {
            let encoded = hamming74_encode(word);
            let decoded = hamming74_decode(encoded);
            assert_eq!(decoded, word, "round-trip failed for {word}");
        }
    }

    #[test]
    fn hamming_corrects_single_bit_flip() {
        for word in 0u32..16 {
            let encoded = hamming74_encode(word);
            for flip in 0..7 {
                let corrupted = encoded ^ (1 << flip);
                let decoded = hamming74_decode(corrupted);
                assert_eq!(
                    decoded, word,
                    "single-bit error at position {flip} not corrected for word {word}",
                );
            }
        }
    }

    #[test]
    fn hamming_ffi_matches_safe_fn() {
        for word in 0u32..16 {
            let safe_enc = hamming74_encode(word);
            let ffi_enc = unsafe { hamming74_encode_ffi(word) };
            assert_eq!(safe_enc, ffi_enc);
            let safe_dec = hamming74_decode(safe_enc);
            let ffi_dec = unsafe { hamming74_decode_ffi(safe_enc) };
            assert_eq!(safe_dec, ffi_dec);
        }
    }

    // ── ScDoctor.adapt ────────────────────────────────────────────────

    #[test]
    fn adapt_doubles_on_high_scc() {
        let r = sc_doctor_adapt(512, false, 0.2);
        assert_eq!(r.new_length, 1024);
        assert!(!r.ecc_enabled, "length 1024 stays under 2048 → ECC off");
    }

    #[test]
    fn adapt_enables_ecc_when_length_crosses_2048() {
        let r = sc_doctor_adapt(2048, false, 0.2);
        assert_eq!(r.new_length, 4096);
        assert!(r.ecc_enabled, "new length 4096 > 2048 → ECC on");
    }

    #[test]
    fn adapt_halves_on_low_scc() {
        let r = sc_doctor_adapt(1024, true, 0.02);
        assert_eq!(r.new_length, 512);
        assert!(!r.ecc_enabled, "halving disables ECC");
    }

    #[test]
    fn adapt_floor_at_256() {
        let r = sc_doctor_adapt(256, false, 0.02);
        assert_eq!(r.new_length, 256, "must not halve below 256 floor");
    }

    #[test]
    fn adapt_hysteresis_band_unchanged() {
        for corr in [0.05, 0.10, 0.149] {
            let r = sc_doctor_adapt(1024, false, corr);
            assert_eq!(
                r.new_length, 1024,
                "length must not change inside hysteresis band ({corr})",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "pyo3_bindings")]
mod python {
    use super::*;
    use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::prelude::*;

    /// Compute SCC between two 1D uint8 bitstreams.
    #[pyfunction]
    fn py_scc_bytes<'py>(
        a: PyReadonlyArray1<'py, u8>,
        b: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        if a_slice.len() != b_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stream length mismatch",
            ));
        }
        Ok(scc_bytes(a_slice, b_slice))
    }

    /// Compute SCC between two packed u64 bitstreams.
    #[pyfunction]
    fn py_scc_packed<'py>(
        a: PyReadonlyArray1<'py, u64>,
        b: PyReadonlyArray1<'py, u64>,
        bit_length: usize,
    ) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        if a_slice.len() != b_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "word count mismatch",
            ));
        }
        Ok(scc_packed(a_slice, b_slice, bit_length))
    }

    /// Compute full N*N pairwise SCC matrix.
    /// Input shape: (num_streams, stream_length), uint8.
    /// Returns flat (N*N,) f64 array.
    #[pyfunction]
    fn py_scc_batch<'py>(
        py: Python<'py>,
        streams: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = streams.shape();
        let n = shape[0];
        let stream_len = shape[1];
        let flat = streams.as_slice()?;
        let mut out = vec![0.0f64; n * n];
        scc_batch_impl(flat, n, stream_len, &mut out);
        Ok(PyArray1::from_vec(py, out))
    }

    /// Estimate precision (probability, variance) for a uint8 bitstream.
    #[pyfunction]
    fn py_precision_bytes<'py>(data: PyReadonlyArray1<'py, u8>) -> PyResult<(f64, f64)> {
        let slice = data.as_slice()?;
        let n = slice.len();
        if n == 0 {
            return Ok((0.0, 0.0));
        }
        let p = popcount_bytes(slice) as f64 / n as f64;
        let variance = p * (1.0 - p) / n as f64;
        Ok((p, variance))
    }

    /// Compute per-word popcount histogram for a uint8 bitstream.
    #[pyfunction]
    fn py_histogram<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, u8>,
        word_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let slice = data.as_slice()?;
        if word_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "word_size must be > 0",
            ));
        }
        let mut hist = vec![0i64; word_size + 1];
        for chunk in slice.chunks(word_size) {
            let pc: usize = chunk.iter().map(|&x| x as usize).sum();
            if pc <= word_size {
                hist[pc] += 1;
            }
        }
        Ok(PyArray1::from_vec(py, hist))
    }

    /// Encode a 4-bit data word via Hamming(7,4).
    #[pyfunction]
    fn py_hamming74_encode(data: u32) -> u32 {
        hamming74_encode(data & 0x0F)
    }

    /// Decode a 7-bit codeword with single-error correction.
    #[pyfunction]
    fn py_hamming74_decode(encoded: u32) -> u32 {
        hamming74_decode(encoded & 0x7F)
    }

    /// Adaptive bitstream length controller (hysteresis rules from ScDoctor).
    #[pyfunction]
    fn py_sc_doctor_adapt(current_length: u32, current_ecc: bool, correlation: f64) -> (u32, bool) {
        let result = sc_doctor_adapt(current_length, current_ecc, correlation);
        (result.new_length, result.ecc_enabled)
    }

    /// EMA-based drift detector (stateful).
    #[pyclass]
    struct PyDriftDetector {
        inner: DriftDetector,
        history: Vec<f64>,
    }

    #[pymethods]
    impl PyDriftDetector {
        #[new]
        fn new(alpha: f64, threshold: f64) -> Self {
            Self {
                inner: DriftDetector::new(alpha, threshold),
                history: Vec::new(),
            }
        }

        fn observe(&mut self, scc_value: f64) -> bool {
            let result = self.inner.observe(scc_value);
            self.history.push(self.inner.ema);
            result
        }

        fn reset(&mut self) {
            self.inner.reset();
            self.history.clear();
        }

        #[getter]
        fn ema(&self) -> f64 {
            self.inner.ema
        }

        #[getter]
        fn active(&self) -> bool {
            self.inner.active
        }

        #[getter]
        fn get_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            PyArray1::from_vec(py, self.history.clone())
        }
    }

    /// Python module registration.
    #[pymodule]
    fn stochastic_doctor_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_scc_bytes, m)?)?;
        m.add_function(wrap_pyfunction!(py_scc_packed, m)?)?;
        m.add_function(wrap_pyfunction!(py_scc_batch, m)?)?;
        m.add_function(wrap_pyfunction!(py_precision_bytes, m)?)?;
        m.add_function(wrap_pyfunction!(py_histogram, m)?)?;
        m.add_function(wrap_pyfunction!(py_hamming74_encode, m)?)?;
        m.add_function(wrap_pyfunction!(py_hamming74_decode, m)?)?;
        m.add_function(wrap_pyfunction!(py_sc_doctor_adapt, m)?)?;
        m.add_class::<PyDriftDetector>()?;
        Ok(())
    }
}
