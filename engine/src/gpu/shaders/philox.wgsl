// SPDX-License-Identifier: AGPL-3.0-or-later
// Philox 4x32-10 counter-based PRNG for GPU compute.
// Reference: Salmon et al., SC '11: Parallel random numbers: as easy as 1, 2, 3.

const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;

// 32×32 → 64 bit multiply, returning (lo, hi) as vec2<u32>.
fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

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
    return vec4<u32>(
        r1.y ^ ctr.y ^ key.x,
        r1.x,
        r0.y ^ ctr.w ^ key.y,
        r0.x,
    );
}

// Philox 4×32-10: 10-round counter-based PRNG.
// Each unique (counter, seed) pair produces 4 independent u32 outputs.
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
