use sc_neurocore_engine::grad::{DifferentiableDenseLayer, SurrogateLif, SurrogateType};
use sc_neurocore_engine::neuron::FixedPointLif;

#[test]
fn fast_sigmoid_gradient_at_zero() {
    // FastSigmoid: 1 / (2*k * (1 + k*0)^2) = 1 / (2*25) = 0.02
    let sg = SurrogateType::FastSigmoid { k: 25.0 };
    assert!((sg.grad(0.0) - 0.02).abs() < 1e-6);
}

#[test]
fn fast_sigmoid_gradient_decays_away_from_zero() {
    let sg = SurrogateType::FastSigmoid { k: 25.0 };
    let g0 = sg.grad(0.0);
    let g1 = sg.grad(0.1);
    let g2 = sg.grad(1.0);
    assert!(g0 > g1);
    assert!(g1 > g2);
    assert!(g2 > 0.0);
}

#[test]
fn superspike_gradient_symmetric() {
    let sg = SurrogateType::SuperSpike { k: 100.0 };
    assert!((sg.grad(0.5) - sg.grad(-0.5)).abs() < 1e-6);
}

#[test]
fn superspike_gradient_at_zero_is_one() {
    // SuperSpike: 1 / (1 + k*0)^2 = 1.0
    let sg = SurrogateType::SuperSpike { k: 100.0 };
    assert!((sg.grad(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn fast_sigmoid_differs_from_superspike() {
    // With the same k, the two must differ by the 1/(2k) factor
    let fs = SurrogateType::FastSigmoid { k: 25.0 };
    let ss = SurrogateType::SuperSpike { k: 25.0 };
    let fs_grad = fs.grad(0.0);
    let ss_grad = ss.grad(0.0);
    // fs_grad = 1/(2*25) = 0.02, ss_grad = 1.0
    assert!((ss_grad / fs_grad - 50.0).abs() < 1e-4);
}

#[test]
fn arctan_gradient_is_lorentzian() {
    let sg = SurrogateType::ArcTan { k: 10.0 };
    assert!((sg.grad(0.0) - 1.0).abs() < 1e-6);
    assert!((sg.grad(0.1) - 0.5).abs() < 1e-6);
}

#[test]
fn straight_through_is_unit_box() {
    let sg = SurrogateType::StraightThrough;
    assert_eq!(sg.grad(0.0), 1.0);
    assert_eq!(sg.grad(0.3), 1.0);
    assert_eq!(sg.grad(0.5), 0.0);
    assert_eq!(sg.grad(-0.5), 0.0);
    assert_eq!(sg.grad(1.0), 0.0);
}

#[test]
fn surrogate_lif_forward_matches_plain_lif() {
    let mut plain = FixedPointLif::new(16, 8, 0, 0, 256, 2);
    let mut surr = SurrogateLif::new(16, 8, 0, 0, 256, 2, SurrogateType::FastSigmoid { k: 25.0 });

    for _ in 0..50 {
        let (s1, v1) = plain.step(20, 256, 128, 0);
        let (s2, v2) = surr.forward(20, 256, 128, 0);
        assert_eq!(s1, s2);
        assert_eq!(v1, v2);
    }
}

#[test]
fn backward_produces_nonzero_gradient() {
    let mut surr = SurrogateLif::new(16, 8, 0, 0, 256, 2, SurrogateType::FastSigmoid { k: 25.0 });
    surr.forward(20, 256, 128, 0);
    let grad = surr.backward(1.0);
    assert!(grad.abs() > 0.0);
}

#[test]
fn differentiable_layer_backward_shapes() {
    let mut layer =
        DifferentiableDenseLayer::new(8, 4, 1024, 42, SurrogateType::FastSigmoid { k: 25.0 });
    let out = layer.forward(&[0.5; 8], 42).unwrap();
    assert_eq!(out.len(), 4);

    let (grad_in, grad_w) = layer.backward(&[1.0; 4]).unwrap();
    assert_eq!(grad_in.len(), 8);
    assert_eq!(grad_w.len(), 4);
    assert_eq!(grad_w[0].len(), 8);
}

#[test]
fn weight_update_changes_weights() {
    let mut layer =
        DifferentiableDenseLayer::new(4, 2, 1024, 42, SurrogateType::FastSigmoid { k: 25.0 });
    let w_before = layer.layer.get_weights();
    let _ = layer.forward(&[0.5; 4], 42).unwrap();
    let (_, grad_w) = layer.backward(&[1.0; 2]).unwrap();
    layer.update_weights(&grad_w, 0.1);
    let w_after = layer.layer.get_weights();
    assert_ne!(w_before, w_after);
}
