// SPDX-License-Identifier: AGPL-3.0-or-later
// Bernoulli encode kernel: converts f32 probabilities to packed bitstreams.
// Each thread generates one vec2<u32> word (= 64 packed bits).

// ---- Philox RNG (inlined) ----
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;

fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu; let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu; let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid  = p1 + (p0 >> 16u);
    let mid2 = (mid & 0xFFFFu) + p2;
    let hi = p3 + (mid >> 16u) + (mid2 >> 16u);
    let lo = (mid2 << 16u) | (p0 & 0xFFFFu);
    return vec2<u32>(lo, hi);
}

fn philox_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let r0 = mulhilo(PHILOX_M0, ctr.x);
    let r1 = mulhilo(PHILOX_M1, ctr.z);
    return vec4<u32>(r1.y ^ ctr.y ^ key.x, r1.x, r0.y ^ ctr.w ^ key.y, r0.x);
}

fn philox4x32(counter: vec4<u32>, seed: vec2<u32>) -> vec4<u32> {
    var key = seed;
    var ctr = counter;
    for (var i = 0u; i < 10u; i = i + 1u) {
        ctr = philox_round(ctr, key);
        key.x = key.x + PHILOX_W0;
        key.y = key.y + PHILOX_W1;
    }
    return ctr;
}

// ---- Kernel bindings ----

struct EncodeParams {
    n_inputs: u32,
    words_per_input: u32,
    seed_lo: u32,
    seed_hi: u32,
    n_samples: u32,
    length: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_inputs: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: EncodeParams;

// Compare 4 random u32 values byte-by-byte against threshold,
// producing 16 packed bits (one per byte).
fn threshold_16bits(rnd: vec4<u32>, thresh: u32) -> u32 {
    var bits = 0u;
    for (var r = 0u; r < 4u; r = r + 1u) {
        let val = select(select(select(rnd.x, rnd.y, r == 1u), rnd.z, r == 2u), rnd.w, r == 3u);
        for (var b = 0u; b < 4u; b = b + 1u) {
            let byte_val = (val >> (b * 8u)) & 0xFFu;
            if (byte_val < thresh) {
                bits = bits | (1u << (r * 4u + b));
            }
        }
    }
    return bits;
}

@compute @workgroup_size(256, 1, 1)
fn encode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let sample_idx = gid.y;

    let input_idx = flat_idx / params.words_per_input;
    let word_idx  = flat_idx % params.words_per_input;

    if (input_idx >= params.n_inputs || sample_idx >= params.n_samples) {
        return;
    }

    let prob_idx = sample_idx * params.n_inputs + input_idx;
    let prob = input_probs[prob_idx];
    let out_idx = (sample_idx * params.n_inputs + input_idx) * params.words_per_input + word_idx;

    // Fast paths for p=0 and p=1.
    if (prob <= 0.0) {
        packed_inputs[out_idx] = vec2<u32>(0u, 0u);
        return;
    }
    if (prob >= 1.0) {
        let is_last = word_idx == params.words_per_input - 1u;
        let rem = params.length % 64u;
        if (is_last && rem > 0u) {
            if (rem <= 32u) {
                packed_inputs[out_idx] = vec2<u32>((1u << rem) - 1u, 0u);
            } else {
                packed_inputs[out_idx] = vec2<u32>(0xFFFFFFFFu, (1u << (rem - 32u)) - 1u);
            }
        } else {
            packed_inputs[out_idx] = vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
        }
        return;
    }

    let thresh = u32(clamp(prob, 0.0, 1.0) * 256.0);
    let seed = vec2<u32>(params.seed_lo, params.seed_hi);

    // Generate lower 32 bits (bytes 0-31): 2 Philox calls → 8 u32 → 32 bytes → 32 bits.
    let ctr_lo0 = vec4<u32>(input_idx, word_idx * 4u + 0u, sample_idx, 0u);
    let ctr_lo1 = vec4<u32>(input_idx, word_idx * 4u + 1u, sample_idx, 0u);
    let rnd_lo0 = philox4x32(ctr_lo0, seed);
    let rnd_lo1 = philox4x32(ctr_lo1, seed);
    let lo_bits = threshold_16bits(rnd_lo0, thresh) | (threshold_16bits(rnd_lo1, thresh) << 16u);

    // Generate upper 32 bits (bytes 32-63): 2 more Philox calls.
    let ctr_hi0 = vec4<u32>(input_idx, word_idx * 4u + 2u, sample_idx, 0u);
    let ctr_hi1 = vec4<u32>(input_idx, word_idx * 4u + 3u, sample_idx, 0u);
    let rnd_hi0 = philox4x32(ctr_hi0, seed);
    let rnd_hi1 = philox4x32(ctr_hi1, seed);
    let hi_bits = threshold_16bits(rnd_hi0, thresh) | (threshold_16bits(rnd_hi1, thresh) << 16u);

    packed_inputs[out_idx] = vec2<u32>(lo_bits, hi_bits);
}
