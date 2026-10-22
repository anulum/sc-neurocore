// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU streaming lossless encoder (one channel per invocation, one range coder each)
//
// Bit-exact port of engine/src/waveform_stream.rs with groups == channels. All arithmetic is
// 32-bit: the range coder's 33-bit `low` is kept as (low_lo, low_hi), residuals stay below 2^21,
// and nominal level values come from a host-computed table. The `table` buffer holds, per
// channel and level, three adjacent words: the exact level value, the level count (bit 31 =
// excluded missing code) and the dense level rank, at `((channel << bits) + level) * 3`, so a
// level and its neighbours share cache lines.

struct Params {
    channels: u32,
    samples: u32,
    bits: u32,
    max_words: u32,
}

const PROB_BITS: u32 = 12u;
const ADAPT_SHIFT: u32 = 6u;
const LMS_ORDER: u32 = 16u;
const LMS_SHIFT: u32 = 12u;
const TREE_ESCAPE: u32 = 63u;
const MAX_ESCAPE_BITS: u32 = 48u;
const TOP: u32 = 16777216u;
const UNSET: i32 = -2147483647 - 1;
const EXCLUDE_RATIO: u32 = 64u;
const EXCLUDE_MIN_COUNT: u32 = 8u;
const INCLUDE_RATIO: u32 = 16u;
const COUNT_LIMIT: u32 = 65536u;
const EXCLUDED_BIT: u32 = 0x80000000u;

// Per-channel state words.
const S_PREV: u32 = 0u;
const S_HEAD: u32 = 1u;
const S_RING: u32 = 2u;      // 32 words
const S_STEPS: u32 = 34u;    // 32 words
const S_WEIGHTS: u32 = 66u;  // 16 words
const S_TREE: u32 = 82u;     // 64 words
const S_ESC: u32 = 146u;     // 48 words
const S_DEV_TREE: u32 = 194u; // 64 words (deviation model, one per channel because groups == channels)
const S_DEV_ESC: u32 = 258u;  // 48 words
const STATE_WORDS: u32 = 306u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<i32>;
@group(0) @binding(2) var<storage, read> levels: array<u32>;
@group(0) @binding(3) var<storage, read> nominal: array<i32>;
@group(0) @binding(4) var<storage, read_write> state: array<i32>;
@group(0) @binding(5) var<storage, read_write> table: array<i32>;
@group(0) @binding(6) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_len: array<u32>;
@group(0) @binding(8) var<storage, read_write> conflict: array<atomic<u32>>;

var<private> low_lo: u32;
var<private> low_hi: u32;
var<private> range_: u32;
var<private> cache: u32;
var<private> cache_size: u32;
var<private> out_pos: u32;
var<private> out_base: u32;
var<private> ch: u32;
var<private> n_ch: u32;

fn si(word: u32) -> u32 { return word * n_ch + ch; }

fn put_byte(b: u32) {
    let w = out_base + (out_pos >> 2u);
    let sh = (out_pos & 3u) * 8u;
    if (sh == 0u) {
        out_words[w] = b & 255u;
    } else {
        out_words[w] = out_words[w] | ((b & 255u) << sh);
    }
    out_pos = out_pos + 1u;
}

fn shift_low() {
    if (low_lo < 0xFF000000u || low_hi != 0u) {
        let carry = low_hi;
        var t = cache;
        loop {
            put_byte(t + carry);
            t = 255u;
            cache_size = cache_size - 1u;
            if (cache_size == 0u) { break; }
        }
        cache = low_lo >> 24u;
    }
    cache_size = cache_size + 1u;
    low_lo = low_lo << 8u;
    low_hi = 0u;
}

fn add_low(v: u32) {
    let s = low_lo + v;
    if (s < low_lo) { low_hi = low_hi + 1u; }
    low_lo = s;
}

fn normalize() {
    loop {
        if (range_ >= TOP) { break; }
        range_ = range_ << 8u;
        shift_low();
    }
}

fn code_bit(word: u32, bit: bool) {
    let index = si(word);
    let p = u32(state[index]);
    let bound = (range_ >> PROB_BITS) * p;
    if (bit) {
        add_low(bound);
        range_ = range_ - bound;
        state[index] = i32(p - (p >> ADAPT_SHIFT));
    } else {
        range_ = bound;
        state[index] = i32(p + (((1u << PROB_BITS) - p) >> ADAPT_SHIFT));
    }
    normalize();
}

fn code_direct(v: u32, n: u32) {
    var i = n;
    loop {
        if (i == 0u) { break; }
        i = i - 1u;
        range_ = range_ >> 1u;
        if (((v >> i) & 1u) == 1u) { add_low(range_); }
        normalize();
    }
}

fn put_symbol(tree: u32, symbol: u32) {
    var node = 1u;
    var i = 6u;
    loop {
        if (i == 0u) { break; }
        i = i - 1u;
        let bit = ((symbol >> i) & 1u) == 1u;
        code_bit(tree + node, bit);
        node = node * 2u + select(0u, 1u, bit);
    }
}

fn put_gamma(esc: u32, value: u32) {
    let length = 32u - countLeadingZeros(value);
    for (var k = 0u; k + 1u < length; k = k + 1u) { code_bit(esc + k, true); }
    code_bit(esc + length - 1u, false);
    code_direct(value, length - 1u);
}

// Values of 63 and more escape with the gamma code of u - 61 (escape value 1 is the marker).
fn put_residual(tree: u32, esc: u32, u: u32) {
    let symbol = min(u, TREE_ESCAPE);
    put_symbol(tree, symbol);
    if (symbol == TREE_ESCAPE) { put_gamma(esc, u - TREE_ESCAPE + 2u); }
}

// Dense level numbering (sections of `table`, see the header comment).
var<private> n_levels: u32;

fn value_index(level: u32) -> u32 { return ((ch << params.bits) + level) * 3u; }
fn count_index(level: u32) -> u32 { return value_index(level) + 1u; }
fn rank_index(level: u32) -> u32 { return value_index(level) + 2u; }
fn level_count(level: u32) -> u32 { return u32(table[count_index(level)]) & ~EXCLUDED_BIT; }
fn is_excluded(level: u32) -> bool { return (u32(table[count_index(level)]) & EXCLUDED_BIT) != 0u; }

fn exclusion(level: u32, now: bool) -> bool {
    if (level == 0u || level + 1u >= n_levels) { return false; }
    let rarer = min(level_count(level - 1u), level_count(level + 1u));
    let own = level_count(level);
    if (now) { return own * INCLUDE_RATIO < rarer; }
    return rarer >= EXCLUDE_MIN_COUNT && own * EXCLUDE_RATIO < rarer;
}

fn set_excluded(level: u32, excluded: bool) {
    table[count_index(level)] = i32(level_count(level) | select(0u, EXCLUDED_BIT, excluded));
}

// After `level` grows, only two changes are possible (as in the CPU coder): `level` returns if
// it was excluded, and an included neighbour leaves (needs neighbour * EXCLUDE_RATIO < count).
fn observe(level: u32) {
    let word = u32(table[count_index(level)]) + 1u;
    table[count_index(level)] = i32(word);
    let own = word & ~EXCLUDED_BIT;
    var changed = false;
    if (own >= COUNT_LIMIT) {
        for (var l = 0u; l < n_levels; l = l + 1u) {
            let w = u32(table[count_index(l)]);
            table[count_index(l)] = i32(((w & ~EXCLUDED_BIT) >> 1u) | (w & EXCLUDED_BIT));
        }
        for (var l = 0u; l < n_levels; l = l + 1u) {
            let now = is_excluded(l);
            let next = exclusion(l, now);
            if (next != now) { set_excluded(l, next); changed = true; }
        }
    } else {
        if ((word & EXCLUDED_BIT) != 0u && !exclusion(level, true)) {
            set_excluded(level, false);
            changed = true;
        }
        for (var side = 0u; side < 2u; side = side + 1u) {
            if (side == 0u && level == 0u) { continue; }
            let neighbour = select(level + 1u, level - 1u, side == 0u);
            if (neighbour >= n_levels || is_excluded(neighbour)) { continue; }
            if (level_count(neighbour) * EXCLUDE_RATIO < own && exclusion(neighbour, false)) {
                set_excluded(neighbour, true);
                changed = true;
            }
        }
    }
    if (changed) {
        var rank = 0;
        for (var l = 0u; l < n_levels; l = l + 1u) {
            table[rank_index(l)] = rank;
            if (!is_excluded(l)) { rank = rank + 1; }
        }
    }
}

fn zigzag(v: i32) -> u32 {
    if (v >= 0) { return u32(v) << 1u; }
    return (u32(-v) << 1u) - 1u;
}

fn predict() -> i32 {
    let head = u32(state[si(S_HEAD)]);
    var acc = 0i;
    for (var i = 0u; i < LMS_ORDER; i = i + 1u) {
        acc = acc + state[si(S_WEIGHTS + i)] * state[si(S_RING + head + i)];
    }
    return acc >> LMS_SHIFT;
}

fn update(difference: i32, error: i32) {
    var head = u32(state[si(S_HEAD)]);
    if (error != 0) {
        for (var i = 0u; i < LMS_ORDER; i = i + 1u) {
            var step = state[si(S_STEPS + head + i)];
            if (error < 0) { step = -step; }
            let w = clamp(state[si(S_WEIGHTS + i)] + step, -32768, 32767);
            state[si(S_WEIGHTS + i)] = w;
        }
    }
    head = select(head - 1u, LMS_ORDER - 1u, head == 0u);
    state[si(S_HEAD)] = i32(head);
    let value = clamp(difference, -2048, 2047);
    let step = 2 * sign(value);
    state[si(S_RING + head)] = value;
    state[si(S_RING + head + LMS_ORDER)] = value;
    state[si(S_STEPS + head)] = step;
    state[si(S_STEPS + head + LMS_ORDER)] = step;
}

// Pass 1: refuse the whole window if any sample contradicts its channel's level table. New
// levels inside the window are checked against earlier samples of the same window directly.
@compute @workgroup_size(64)
fn validate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.channels) { return; }
    let n = params.channels;
    var bad = false;
    for (var t = 0u; t < params.samples; t = t + 1u) {
        let x = samples[t * n + c];
        let level = levels[u32(x + 32768)];
        var known = table[((c << params.bits) + level) * 3u];
        if (known == UNSET) { known = pending[(c << params.bits) + level]; }
        if (known == UNSET) {
            pending[(c << params.bits) + level] = x;
        } else if (known != x) {
            bad = true;
            break;
        }
    }
    // Reset the pending entries this window touched.
    for (var t = 0u; t < params.samples; t = t + 1u) {
        let level = levels[u32(samples[t * n + c] + 32768)];
        pending[(c << params.bits) + level] = UNSET;
    }
    if (bad) { atomicOr(&conflict[0], 1u); out_len[c] = 0xFFFFFFFFu; }
}

// Pass 2: encode the channel's window with its own range coder (skipped entirely on a conflict).
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.channels) { return; }
    if (atomicLoad(&conflict[0]) != 0u) { return; }
    let n = params.channels;
    ch = c;
    n_ch = params.channels;
    n_levels = 1u << params.bits;
    out_base = c * params.max_words;
    out_pos = 0u;
    low_lo = 0u; low_hi = 0u; range_ = 0xFFFFFFFFu; cache = 0u; cache_size = 1u;
    for (var t = 0u; t < params.samples; t = t + 1u) {
        let x = samples[t * n + c];
        let level = levels[u32(x + 32768)];
        let difference = table[rank_index(level)] - table[rank_index(u32(state[si(S_PREV)]))];
        let error = difference - predict();
        if (is_excluded(level)) {
            put_symbol(S_TREE, TREE_ESCAPE);
            put_gamma(S_ESC, 1u);
            code_direct(level, params.bits);
        } else {
            put_residual(S_TREE, S_ESC, zigzag(error));
        }
        update(difference, error);
        state[si(S_PREV)] = i32(level);
        if (table[value_index(level)] == UNSET) {
            put_residual(S_DEV_TREE, S_DEV_ESC, zigzag(x - nominal[level]));
            table[value_index(level)] = x;
        }
        observe(level);
    }
    for (var k = 0u; k < 5u; k = k + 1u) { shift_low(); }
    out_len[c] = out_pos;
}

@group(0) @binding(9) var<storage, read> offsets: array<u32>;
@group(0) @binding(10) var<storage, read_write> compact: array<u32>;
@group(0) @binding(11) var<storage, read_write> pending: array<i32>;

// Pass 3: copy every channel's coded words to its host-computed word offset (no shared words).
@compute @workgroup_size(64)
fn compact_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.channels) { return; }
    let words = (out_len[c] + 3u) / 4u;
    let source = c * params.max_words;
    let dst = offsets[c];
    for (var k = 0u; k < words; k = k + 1u) { compact[dst + k] = out_words[source + k]; }
}
