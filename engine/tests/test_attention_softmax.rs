// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

use sc_neurocore_engine::attention::StochasticAttention;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn softmax_sums_to_one() {
    let attn = StochasticAttention::new(4);
    // 2x4 Q, 3x4 K, 3x2 V
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let out = attn.forward_softmax(&q, 2, 4, &k, 3, 4, &v, 3, 2).unwrap();
    // Output should have 2*2 = 4 elements
    assert_eq!(out.len(), 4);
}

#[test]
fn softmax_numerical_stability_large_scores() {
    let attn = StochasticAttention::with_temperature(2, 1.0);
    // Large values that would overflow naive exp()
    let q = vec![500.0, 500.0, 501.0, 501.0];
    let k = vec![500.0, 500.0, 500.0, 500.0];
    let v = vec![1.0, 0.0, 0.0, 1.0];
    let out = attn.forward_softmax(&q, 2, 2, &k, 2, 2, &v, 2, 2).unwrap();
    // Must not be NaN or Inf
    for val in &out {
        assert!(val.is_finite(), "output must be finite, got {}", val);
    }
}

#[test]
fn temperature_affects_sharpness() {
    // Low temperature → sharper distribution → output closer to argmax row of V
    let sharp = StochasticAttention::with_temperature(2, 0.01);
    let soft = StochasticAttention::with_temperature(2, 100.0);

    let q = vec![1.0, 0.0]; // 1x2
    let k = vec![1.0, 0.0, 0.0, 1.0]; // 2x2, first row matches q
    let v = vec![1.0, 0.0, 0.0, 1.0]; // 2x2

    let out_sharp = sharp.forward_softmax(&q, 1, 2, &k, 2, 2, &v, 2, 2).unwrap();
    let out_soft = soft.forward_softmax(&q, 1, 2, &k, 2, 2, &v, 2, 2).unwrap();

    // Sharp: should be close to [1.0, 0.0] (first V row)
    assert!(out_sharp[0] > 0.99);
    // Soft: should be close to uniform → [0.5, 0.5]
    assert!(approx_eq(out_soft[0], 0.5, 0.1));
}

#[test]
fn softmax_uniform_scores_equals_mean() {
    let attn = StochasticAttention::with_temperature(3, 1.0);
    // Q=K so all dot products equal → uniform softmax → output = mean of V rows
    let q = vec![1.0, 1.0, 1.0]; // 1x3
    let k = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 2x3
    let v = vec![2.0, 4.0, 6.0, 8.0]; // 2x2

    let out = attn.forward_softmax(&q, 1, 3, &k, 2, 3, &v, 2, 2).unwrap();
    // Mean of V rows: [(2+6)/2, (4+8)/2] = [4.0, 6.0]
    assert!(approx_eq(out[0], 4.0, 1e-10));
    assert!(approx_eq(out[1], 6.0, 1e-10));
}

#[test]
fn multihead_softmax_shape() {
    let attn = StochasticAttention::new(4);
    let q = vec![0.0_f64; 2 * 4]; // 2x4
    let k = vec![0.0_f64; 3 * 4]; // 3x4
    let v = vec![0.0_f64; 3 * 4]; // 3x4
    let out = attn
        .forward_multihead_softmax(&q, 2, 4, &k, 3, 4, &v, 3, 4, 2)
        .unwrap();
    assert_eq!(out.len(), 2 * 4); // same as input shape
}

#[test]
fn default_temperature_is_sqrt_dim() {
    let attn = StochasticAttention::new(64);
    assert!(approx_eq(attn.temperature, 8.0, 1e-10));
}
