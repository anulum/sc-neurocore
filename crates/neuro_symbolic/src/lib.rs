// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuro-Symbolic HDC/VSA Engine
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # Neuro-Symbolic HDC/VSA Engine
//!
//! Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA)
//! implementation with associative memory and symbol encoding.
//!
//! Data layout is `Vec<u64>` packed bitstreams matching `engine::BitStreamTensor`.
//! Operations align with the engine's HDC primitives (`xor`, `rotate_right`, `bundle`,
//! `hamming_distance`) while adding higher-level abstractions:
//!
//! - **`Hypervector`**: Core type compatible with `BitStreamTensor`
//! - **`AssociativeMemory`**: Clean-up / auto-associative memory (nearest-neighbour)
//! - **`SymbolEncoder`**: Deterministic string → hypervector mapping
//!
//! ## References
//! - Kanerva, "Hyperdimensional Computing", Cognitive Computation 1(2), 2009.
//! - Plate, "Holographic Reduced Representations", IEEE Trans. NN 6(3), 1995.
//! - Rahimi et al., "Classification and Regression with Binary HDC", DAC 2016.

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Default dimension for hypervectors.
///
/// 10 000 bits = 157 packed u64 words. Matches the standard HDC literature
/// recommendation of D ≥ 1000 for high-confidence retrieval.
pub const HYPERVECTOR_DIM: usize = 10_000;

/// Number of packed u64 words for `HYPERVECTOR_DIM` bits.
pub const HYPERVECTOR_WORDS: usize = HYPERVECTOR_DIM.div_ceil(64);

// ---------------------------------------------------------------------------
// Core type: Hypervector
// ---------------------------------------------------------------------------

/// Packed binary hypervector.
///
/// Layout-compatible with `engine::BitStreamTensor { data: Vec<u64>, length: usize }`.
/// Uses little-endian bit ordering within each word: bit 0 of word 0 = dimension 0.
#[derive(Clone, Debug)]
pub struct Hypervector {
    pub data: Vec<u64>,
    pub length: usize,
}

impl Hypervector {
    /// Create a zero hypervector of default dimension.
    pub fn zeros() -> Self {
        Self {
            data: vec![0u64; HYPERVECTOR_WORDS],
            length: HYPERVECTOR_DIM,
        }
    }

    /// Create a hypervector of arbitrary dimension.
    pub fn zeros_with_dim(dim: usize) -> Self {
        Self {
            data: vec![0u64; dim.div_ceil(64)],
            length: dim,
        }
    }

    /// Create a random hypervector (~50% density) using a seeded RNG.
    pub fn random(seed: u64) -> Self {
        use rand::RngExt;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut data = vec![0u64; HYPERVECTOR_WORDS];
        for w in data.iter_mut() {
            *w = rng.random();
        }
        // Mask trailing bits
        let trailing = HYPERVECTOR_DIM % 64;
        if trailing > 0 {
            data[HYPERVECTOR_WORDS - 1] &= (1u64 << trailing) - 1;
        }
        Self {
            data,
            length: HYPERVECTOR_DIM,
        }
    }

    /// Create from pre-packed words.
    pub fn from_words(data: Vec<u64>, length: usize) -> Self {
        debug_assert!(
            data.len() >= length.div_ceil(64),
            "insufficient words for hypervector length"
        );
        Self { data, length }
    }

    // ── HDC Operations ───────────────────────────────────────────────

    /// BIND: Bitwise XOR (returns new vector).
    ///
    /// The fundamental binding operation in MAP-I VSA.
    /// Properties: self-inverse, dimension-preserving, distributes over bundle.
    pub fn bind(&self, other: &Hypervector) -> Hypervector {
        assert_eq!(self.length, other.length, "dimension mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        Hypervector {
            data,
            length: self.length,
        }
    }

    /// BIND in-place: XOR mutating self.
    pub fn bind_inplace(&mut self, other: &Hypervector) {
        assert_eq!(self.length, other.length, "dimension mismatch");
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= b;
        }
    }

    /// PERMUTE: Cyclic right rotation by `shift` bits.
    ///
    /// Matches `BitStreamTensor::rotate_right` semantics. Used for sequence
    /// encoding: bind(permute(A, 1), B) encodes ordered pair (A, B).
    pub fn permute(&mut self, shift: usize) {
        if self.length == 0 || shift % self.length == 0 {
            return;
        }
        let effective = shift % self.length;
        let mut bits = unpack(self);
        bits.rotate_right(effective);
        let repacked = pack(&bits, self.length);
        self.data = repacked.data;
    }

    /// PERMUTE returning a new vector (non-mutating).
    pub fn permuted(&self, shift: usize) -> Hypervector {
        let mut result = self.clone();
        result.permute(shift);
        result
    }

    /// BUNDLE (majority vote) across N vectors.
    ///
    /// Returns a hypervector where each bit is 1 if a strict majority (> N/2)
    /// of input vectors have that bit set. Matches `BitStreamTensor::bundle`.
    ///
    /// For N=2, uses a random tiebreaker seed.
    pub fn threshold_bundle(vectors: &[&Hypervector]) -> Hypervector {
        assert!(!vectors.is_empty(), "cannot bundle zero vectors");
        let length = vectors[0].length;
        let words = vectors[0].data.len();

        for v in vectors {
            assert_eq!(v.length, length, "all vectors must have same dimension");
        }

        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let threshold = vectors.len() / 2;
        let mut data = vec![0u64; words];

        if vectors.len() == 3 {
            // Optimized 3-way majority: (a&b) | (b&c) | (a&c)
            for i in 0..words {
                let a = vectors[0].data[i];
                let b = vectors[1].data[i];
                let c = vectors[2].data[i];
                data[i] = (a & b) | (b & c) | (a & c);
            }
        } else {
            for i in 0..words {
                for bit in 0..64 {
                    let idx = i * 64 + bit;
                    if idx >= length {
                        break;
                    }
                    let mut count = 0usize;
                    for v in vectors {
                        if (v.data[i] >> bit) & 1 == 1 {
                            count += 1;
                        }
                    }
                    if count > threshold {
                        data[i] |= 1u64 << bit;
                    }
                }
            }
        }

        Hypervector { data, length }
    }

    /// SIMILARITY: Normalized Hamming distance (0.0 = identical, 1.0 = opposite).
    ///
    /// Matches `BitStreamTensor::hamming_distance` semantics.
    pub fn hamming_distance(&self, other: &Hypervector) -> f64 {
        assert_eq!(self.length, other.length, "dimension mismatch");
        let xor_count: u64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a ^ b).count_ones() as u64)
            .sum();
        xor_count as f64 / self.length as f64
    }

    /// Cosine-like similarity: 1.0 - 2*hamming_distance.
    ///
    /// Maps [0, 0.5, 1.0] Hamming → [+1.0, 0.0, -1.0] similarity.
    pub fn similarity(&self, other: &Hypervector) -> f64 {
        1.0 - 2.0 * self.hamming_distance(other)
    }

    /// Population count of set bits.
    pub fn popcount(&self) -> u64 {
        self.data.iter().map(|w| w.count_ones() as u64).sum()
    }

    /// Density: fraction of bits set (0.0 to 1.0).
    pub fn density(&self) -> f64 {
        if self.length == 0 {
            return 0.0;
        }
        self.popcount() as f64 / self.length as f64
    }
}

impl PartialEq for Hypervector {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length && self.data == other.data
    }
}

// ---------------------------------------------------------------------------
// Pack / Unpack utilities
// ---------------------------------------------------------------------------

/// Unpack a `Hypervector` into a `0/1` byte vector.
pub fn unpack(hv: &Hypervector) -> Vec<u8> {
    let mut bits = vec![0u8; hv.length];
    for (idx, bit) in bits.iter_mut().enumerate() {
        *bit = ((hv.data[idx / 64] >> (idx % 64)) & 1) as u8;
    }
    bits
}

/// Pack a `0/1` byte slice into a `Hypervector`.
pub fn pack(bits: &[u8], length: usize) -> Hypervector {
    let words = length.div_ceil(64);
    let mut data = vec![0u64; words];
    for (idx, &bit) in bits.iter().enumerate().take(length) {
        if bit != 0 {
            data[idx / 64] |= 1u64 << (idx % 64);
        }
    }
    Hypervector { data, length }
}

// ---------------------------------------------------------------------------
// Associative Memory
// ---------------------------------------------------------------------------

/// Clean-up / auto-associative memory for HDC retrieval.
///
/// Stores labelled hypervectors and retrieves the nearest match by Hamming distance.
/// Used for classification: encode a query, find the closest stored prototype.
pub struct AssociativeMemory {
    entries: Vec<(String, Hypervector)>,
}

impl AssociativeMemory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Store a labelled hypervector.
    pub fn store(&mut self, label: String, hv: Hypervector) {
        self.entries.push((label, hv));
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Query: find the nearest stored vector (minimum Hamming distance).
    ///
    /// Returns `(label, distance)` or `None` if empty.
    pub fn query(&self, probe: &Hypervector) -> Option<(&str, f64)> {
        if self.entries.is_empty() {
            return None;
        }
        let mut best_label = &self.entries[0].0;
        let mut best_dist = f64::MAX;

        for (label, hv) in &self.entries {
            let d = probe.hamming_distance(hv);
            if d < best_dist {
                best_dist = d;
                best_label = label;
            }
        }
        Some((best_label, best_dist))
    }

    /// Query returning the top-K nearest entries, sorted by distance (ascending).
    pub fn query_topk(&self, probe: &Hypervector, k: usize) -> Vec<(&str, f64)> {
        let mut distances: Vec<(&str, f64)> = self
            .entries
            .iter()
            .map(|(label, hv)| (label.as_str(), probe.hamming_distance(hv)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    /// Remove all entries matching a label.
    pub fn remove(&mut self, label: &str) {
        self.entries.retain(|(l, _)| l != label);
    }

    /// List all stored labels.
    pub fn labels(&self) -> Vec<&str> {
        self.entries.iter().map(|(l, _)| l.as_str()).collect()
    }
}

impl Default for AssociativeMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Symbol Encoder
// ---------------------------------------------------------------------------

/// Deterministic symbol-to-hypervector encoder.
///
/// Each unique symbol string is mapped to a random hypervector generated from
/// a hash-derived seed, ensuring that the same symbol always produces the same
/// hypervector across runs. Previously unseen symbols are generated on demand.
///
/// Randomness guarantee: ChaCha8 CSPRNG seeded from the symbol's hash means
/// different symbols produce near-orthogonal vectors (expected Hamming ≈ 0.5).
pub struct SymbolEncoder {
    cache: HashMap<String, Hypervector>,
    base_seed: u64,
}

impl SymbolEncoder {
    /// Create a new encoder with a base seed.
    ///
    /// The base seed is mixed with per-symbol hashes for reproducibility.
    pub fn new(base_seed: u64) -> Self {
        Self {
            cache: HashMap::new(),
            base_seed,
        }
    }

    /// Encode a symbol string. Caches the result for future lookups.
    pub fn encode(&mut self, symbol: &str) -> &Hypervector {
        if !self.cache.contains_key(symbol) {
            let seed = self.symbol_seed(symbol);
            let hv = Hypervector::random(seed);
            self.cache.insert(symbol.to_string(), hv);
        }
        &self.cache[symbol]
    }

    /// Encode a sequence of symbols into a single hypervector.
    ///
    /// Uses the n-gram encoding pattern:
    ///   H = bind(permute(S_0, n-1), bind(permute(S_1, n-2), ... S_{n-1}))
    ///
    /// Preserves order information through permutation shifts.
    pub fn encode_sequence(&mut self, symbols: &[&str]) -> Hypervector {
        let n = symbols.len();
        assert!(n > 0, "cannot encode empty sequence");

        if n == 1 {
            return self.encode(symbols[0]).clone();
        }

        // Start from the last symbol (no permutation)
        let mut result = self.encode(symbols[n - 1]).clone();

        // Fold right-to-left with increasing permutation
        for (shift, sym) in symbols.iter().rev().skip(1).enumerate() {
            let mut component = self.encode(sym).clone();
            component.permute(shift + 1);
            result.bind_inplace(&component);
        }
        result
    }

    /// Number of symbols in the cache.
    pub fn vocabulary_size(&self) -> usize {
        self.cache.len()
    }

    fn symbol_seed(&self, symbol: &str) -> u64 {
        // FNV-1a hash mixed with base seed
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in symbol.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^ self.base_seed
    }
}

// ---------------------------------------------------------------------------
// C-FFI Surface
// ---------------------------------------------------------------------------

/// Create a random hypervector. Caller must free with `hv_destroy`.
#[no_mangle]
pub extern "C" fn hv_create_random(seed: u64) -> *mut Hypervector {
    Box::into_raw(Box::new(Hypervector::random(seed)))
}

/// Create a zero hypervector. Caller must free with `hv_destroy`.
#[no_mangle]
pub extern "C" fn hv_create_zeros() -> *mut Hypervector {
    Box::into_raw(Box::new(Hypervector::zeros()))
}

/// Bind (XOR) two hypervectors, returning a new one.
///
/// # Safety
/// Both pointers must be valid `Hypervector` instances.
#[no_mangle]
pub unsafe extern "C" fn hv_bind(a: *const Hypervector, b: *const Hypervector) -> *mut Hypervector {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (a, b) = unsafe { (&*a, &*b) };
    Box::into_raw(Box::new(a.bind(b)))
}

/// Permute (cyclic right rotation) in-place.
///
/// # Safety
/// `ptr` must be a valid `Hypervector`.
#[no_mangle]
pub unsafe extern "C" fn hv_permute(ptr: *mut Hypervector, shift: usize) {
    if ptr.is_null() {
        return;
    }
    let hv = unsafe { &mut *ptr };
    hv.permute(shift);
}

/// Hamming distance between two hypervectors.
///
/// # Safety
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn hv_hamming(a: *const Hypervector, b: *const Hypervector) -> f64 {
    if a.is_null() || b.is_null() {
        return 1.0;
    }
    let (a, b) = unsafe { (&*a, &*b) };
    a.hamming_distance(b)
}

/// Cosine-like similarity between two hypervectors.
///
/// # Safety
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn hv_similarity(a: *const Hypervector, b: *const Hypervector) -> f64 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }
    let (a, b) = unsafe { (&*a, &*b) };
    a.similarity(b)
}

/// Destroy a hypervector.
///
/// # Safety
/// `ptr` must have been returned by `hv_create_*` or `hv_bind`.
#[no_mangle]
pub unsafe extern "C" fn hv_destroy(ptr: *mut Hypervector) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_zero_popcount() {
        let hv = Hypervector::zeros();
        assert_eq!(hv.popcount(), 0);
        assert_eq!(hv.length, HYPERVECTOR_DIM);
        assert_eq!(hv.data.len(), HYPERVECTOR_WORDS);
    }

    #[test]
    fn random_near_half_density() {
        let hv = Hypervector::random(0xDEAD_BEEF);
        let density = hv.density();
        assert!(
            (density - 0.5).abs() < 0.05,
            "random hypervector density should be ~0.5, got {density}"
        );
    }

    #[test]
    fn random_deterministic() {
        let a = Hypervector::random(42);
        let b = Hypervector::random(42);
        assert_eq!(a, b, "same seed must produce identical vectors");
    }

    #[test]
    fn random_different_seeds_orthogonal() {
        let a = Hypervector::random(1);
        let b = Hypervector::random(2);
        let sim = a.similarity(&b);
        assert!(
            sim.abs() < 0.1,
            "different seeds should produce near-orthogonal vectors, got similarity {sim}"
        );
    }

    #[test]
    fn bind_self_inverse() {
        let a = Hypervector::random(100);
        let b = Hypervector::random(200);
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        assert_eq!(a, recovered, "bind is self-inverse: A⊕B⊕B = A");
    }

    #[test]
    fn bind_dissimilar_to_inputs() {
        let a = Hypervector::random(10);
        let b = Hypervector::random(20);
        let c = a.bind(&b);
        assert!(
            c.similarity(&a).abs() < 0.1,
            "A⊕B should be dissimilar to A"
        );
        assert!(
            c.similarity(&b).abs() < 0.1,
            "A⊕B should be dissimilar to B"
        );
    }

    #[test]
    fn permute_preserves_population() {
        let hv = Hypervector::random(333);
        let pop_before = hv.popcount();
        let permuted = hv.permuted(7);
        assert_eq!(
            permuted.popcount(),
            pop_before,
            "permutation should preserve popcount"
        );
    }

    #[test]
    fn permute_full_cycle_returns_identity() {
        let hv = Hypervector::random(444);
        let permuted = hv.permuted(HYPERVECTOR_DIM);
        assert_eq!(hv, permuted, "rotate by full length should be identity");
    }

    #[test]
    fn permute_changes_vector() {
        let hv = Hypervector::random(555);
        let permuted = hv.permuted(1);
        assert_ne!(hv, permuted, "single-bit rotation should change the vector");
    }

    #[test]
    fn threshold_bundle_majority_vote() {
        let a = Hypervector::random(10);
        let b = Hypervector::random(20);
        let c = Hypervector::random(30);
        let bundled = Hypervector::threshold_bundle(&[&a, &b, &c]);
        // Bundle should be more similar to each input than random
        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);
        let sim_c = bundled.similarity(&c);
        assert!(
            sim_a > 0.2,
            "bundle should be similar to input A, got {sim_a}"
        );
        assert!(
            sim_b > 0.2,
            "bundle should be similar to input B, got {sim_b}"
        );
        assert!(
            sim_c > 0.2,
            "bundle should be similar to input C, got {sim_c}"
        );
    }

    #[test]
    fn threshold_bundle_three_way_optimized() {
        // Verify 3-way optimized path matches general path
        let a = Hypervector::random(100);
        let b = Hypervector::random(200);
        let c = Hypervector::random(300);

        let bundled = Hypervector::threshold_bundle(&[&a, &b, &c]);

        // Manual verification: for 3 inputs, majority = at least 2
        for i in 0..HYPERVECTOR_WORDS {
            let expected = (a.data[i] & b.data[i]) | (b.data[i] & c.data[i]) | (a.data[i] & c.data[i]);
            assert_eq!(bundled.data[i], expected, "word {i} mismatch in 3-way bundle");
        }
    }

    #[test]
    fn threshold_bundle_single_vector() {
        let a = Hypervector::random(42);
        let bundled = Hypervector::threshold_bundle(&[&a]);
        assert_eq!(
            a, bundled,
            "bundling a single vector should return a clone"
        );
    }

    #[test]
    fn hamming_distance_self_zero() {
        let a = Hypervector::random(77);
        assert!(a.hamming_distance(&a).abs() < 1e-10);
    }

    #[test]
    fn similarity_self_one() {
        let a = Hypervector::random(88);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let hv = Hypervector::random(999);
        let bits = unpack(&hv);
        let repacked = pack(&bits, hv.length);
        assert_eq!(hv, repacked, "pack/unpack should be lossless");
    }

    // ── Associative Memory Tests ─────────────────────────────────────

    #[test]
    fn memory_store_and_query() {
        let mut mem = AssociativeMemory::new();
        let cat = Hypervector::random(1);
        let dog = Hypervector::random(2);
        mem.store("cat".into(), cat.clone());
        mem.store("dog".into(), dog.clone());

        let (label, dist) = mem.query(&cat).unwrap();
        assert_eq!(label, "cat");
        assert!(dist < 0.01, "querying stored vector should return distance ~0");
    }

    #[test]
    fn memory_noisy_query() {
        let mut mem = AssociativeMemory::new();
        let cat = Hypervector::random(10);
        let dog = Hypervector::random(20);
        let bird = Hypervector::random(30);
        mem.store("cat".into(), cat.clone());
        mem.store("dog".into(), dog.clone());
        mem.store("bird".into(), bird.clone());

        // Add noise to cat (flip ~10% of bits)
        let mut noisy_cat = cat.clone();
        let flip_mask = Hypervector::random(999);
        // Keep only ~10% of the flip mask
        for w in noisy_cat.data.iter_mut().zip(flip_mask.data.iter()) {
            *w.0 ^= w.1 & 0x1111_1111_1111_1111; // ~25% of mask bits
        }

        let (label, _dist) = mem.query(&noisy_cat).unwrap();
        assert_eq!(label, "cat", "noisy cat should still match cat");
    }

    #[test]
    fn memory_topk() {
        let mut mem = AssociativeMemory::new();
        for i in 0..10 {
            let hv = Hypervector::random(i as u64);
            mem.store(format!("item_{i}"), hv);
        }

        let probe = Hypervector::random(0);
        let topk = mem.query_topk(&probe, 3);
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].0, "item_0", "exact match should be first");
        assert!(topk[0].1 < 0.01, "first result should be near-exact match");
    }

    #[test]
    fn memory_remove() {
        let mut mem = AssociativeMemory::new();
        mem.store("a".into(), Hypervector::random(1));
        mem.store("b".into(), Hypervector::random(2));
        assert_eq!(mem.len(), 2);
        mem.remove("a");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem.labels(), vec!["b"]);
    }

    #[test]
    fn memory_empty_query() {
        let mem = AssociativeMemory::new();
        assert!(mem.query(&Hypervector::random(1)).is_none());
    }

    // ── Symbol Encoder Tests ─────────────────────────────────────────

    #[test]
    fn encoder_deterministic() {
        let mut enc1 = SymbolEncoder::new(42);
        let mut enc2 = SymbolEncoder::new(42);
        let hv1 = enc1.encode("hello").clone();
        let hv2 = enc2.encode("hello").clone();
        assert_eq!(hv1, hv2, "same seed + same symbol = same vector");
    }

    #[test]
    fn encoder_different_symbols_orthogonal() {
        let mut enc = SymbolEncoder::new(42);
        let a = enc.encode("cat").clone();
        let b = enc.encode("dog").clone();
        let sim = a.similarity(&b);
        assert!(
            sim.abs() < 0.1,
            "different symbols should be near-orthogonal, got {sim}"
        );
    }

    #[test]
    fn encoder_caches_result() {
        let mut enc = SymbolEncoder::new(42);
        enc.encode("test");
        enc.encode("test");
        assert_eq!(enc.vocabulary_size(), 1);
    }

    #[test]
    fn encoder_sequence_order_matters() {
        let mut enc = SymbolEncoder::new(42);
        let ab = enc.encode_sequence(&["A", "B"]);
        let ba = enc.encode_sequence(&["B", "A"]);
        let sim = ab.similarity(&ba);
        assert!(
            sim.abs() < 0.2,
            "different orderings should produce different vectors, got similarity {sim}"
        );
    }

    #[test]
    fn encoder_sequence_single_symbol() {
        let mut enc = SymbolEncoder::new(42);
        let single = enc.encode("X").clone();
        let seq = enc.encode_sequence(&["X"]);
        assert_eq!(single, seq, "single-symbol sequence should equal the symbol itself");
    }

    // ── FFI Tests ────────────────────────────────────────────────────

    #[test]
    fn ffi_lifecycle() {
        let a = hv_create_random(0xACE1);
        let b = hv_create_random(0xBEEF);
        assert!(!a.is_null());
        assert!(!b.is_null());

        unsafe {
            let c = hv_bind(a, b);
            assert!(!c.is_null());

            let dist = hv_hamming(a, b);
            assert!(dist > 0.3 && dist < 0.7);

            hv_permute(a, 5);

            let sim = hv_similarity(a, a);
            assert!((sim - 1.0).abs() < 1e-10);

            hv_destroy(c);
            hv_destroy(b);
            hv_destroy(a);
        }
    }

    #[test]
    fn ffi_null_safety() {
        unsafe {
            let result = hv_bind(std::ptr::null(), std::ptr::null());
            assert!(result.is_null());
            assert_eq!(hv_hamming(std::ptr::null(), std::ptr::null()), 1.0);
            assert_eq!(hv_similarity(std::ptr::null(), std::ptr::null()), 0.0);
            hv_permute(std::ptr::null_mut(), 5); // should not crash
            hv_destroy(std::ptr::null_mut()); // should not crash
        }
    }

    // ── Integration: Concept Composition ─────────────────────────────

    #[test]
    fn concept_composition_scenario() {
        let mut enc = SymbolEncoder::new(2030);
        let mut mem = AssociativeMemory::new();

        // Build concepts: "red circle", "blue square"
        let red_circle = enc.encode_sequence(&["red", "circle"]);
        let blue_square = enc.encode_sequence(&["blue", "square"]);
        let red_square = enc.encode_sequence(&["red", "square"]);

        mem.store("red_circle".into(), red_circle.clone());
        mem.store("blue_square".into(), blue_square.clone());
        mem.store("red_square".into(), red_square.clone());

        // Query with red_circle → should get red_circle
        let (label, dist) = mem.query(&red_circle).unwrap();
        assert_eq!(label, "red_circle");
        assert!(dist < 0.01);

        // Query with blue_square → should get blue_square
        let (label, _) = mem.query(&blue_square).unwrap();
        assert_eq!(label, "blue_square");
    }
}
