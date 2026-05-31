// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore GolgiCell integration tests

use sc_neurocore_engine::neurons::GolgiCell;

fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
    let d = v + vhalf;
    if d.abs() < 1e-7 {
        fallback
    } else {
        a * d / (1.0 - (-d / k).exp())
    }
}

fn boltz(v: f64, vh: f64, k: f64) -> f64 {
    1.0 / (1.0 + (-(v - vh) / k).exp())
}

fn gate_alpha_beta(previous: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
    let total = phi * (alpha + beta);
    let steady = alpha / (alpha + beta);
    (steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0)
}

fn gate_inf(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
    (steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0)
}

fn calcium(previous: f64, entry: f64, tau: f64, dt: f64) -> f64 {
    let steady = entry * tau;
    (steady + (previous - steady) * (-dt / tau).exp()).max(0.0)
}

fn reference_step(mut cell: GolgiCell, current: f64) -> GolgiCell {
    let dt_sub = cell.dt / cell.sub_steps as f64;
    let input_current = cell.gain * current;
    for _ in 0..cell.sub_steps {
        let v = cell.v;
        let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
        let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
        let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
        let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
        cell.m = gate_alpha_beta(cell.m, alpha_m, beta_m, 5.0, dt_sub);
        cell.h = gate_alpha_beta(cell.h, alpha_h, beta_h, 5.0, dt_sub);
        let tau_pna = 5.0 + 20.0 / (1.0 + ((v + 48.0) / 10.0).powi(2)).max(0.01);
        cell.p_na = gate_inf(cell.p_na, boltz(v, -48.0, 5.0), tau_pna, dt_sub);
        let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
        let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();
        cell.n = gate_alpha_beta(cell.n, alpha_n, beta_n, 5.0, dt_sub);
        cell.a = gate_inf(cell.a, boltz(v, -27.0, 16.0), 2.0, dt_sub);
        cell.b = gate_inf(cell.b, boltz(v, -80.0, -6.0), 15.0, dt_sub);
        let tau_w = 100.0 / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp());
        cell.w = gate_inf(cell.w, boltz(v, -35.0, 10.0), tau_w, dt_sub);
        cell.m_t = gate_inf(cell.m_t, boltz(v, -52.0, 5.0), 1.0, dt_sub);
        let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).powi(2)).max(0.01);
        cell.s = gate_inf(cell.s, boltz(v, -60.0, -6.5), tau_s, dt_sub);
        let tau_cn = 2.0 + 10.0 / (1.0 + ((v + 20.0) / 10.0).powi(2)).max(0.01);
        cell.c_n = gate_inf(cell.c_n, boltz(v, -20.0, 5.0), tau_cn, dt_sub);
        let tau_r = 50.0 + 200.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
        cell.r = gate_inf(cell.r, boltz(v, -80.0, -10.0), tau_r, dt_sub);

        let g_cat = cell.g_cat * cell.m_t.powi(2) * cell.s;
        let g_can = cell.g_can * cell.c_n.powi(2);
        let ca_inward = g_cat * (v - cell.e_ca) + g_can * (v - cell.e_ca);
        let ca_entry = if ca_inward < 0.0 {
            -ca_inward * 0.001
        } else {
            0.0
        };
        cell.ca = calcium(cell.ca, ca_entry, cell.tau_ca, dt_sub);
        let ca2 = cell.ca * cell.ca;
        let bk_v = boltz(v, 100.0 - 120.0 * ca2 / (ca2 + cell.kd_bk.powi(2)), 15.0);
        let sk_inf = ca2 / (ca2 + cell.kd_sk.powi(2));
        let g_na = cell.g_na_t * cell.m.powi(3) * cell.h + cell.g_na_p * cell.p_na;
        let g_k = cell.g_kdr * cell.n.powi(4)
            + cell.g_ka * cell.a.powi(3) * cell.b
            + cell.g_km * cell.w
            + cell.g_bk * bk_v
            + cell.g_sk * sk_inf;
        let g_ca = g_cat + g_can;
        let g_h = cell.g_h * cell.r;
        let g_total = g_na + g_k + g_ca + g_h + cell.g_l;
        let steady_v = (input_current
            + g_na * cell.e_na
            + g_k * cell.e_k
            + g_ca * cell.e_ca
            + g_h * cell.e_h
            + cell.g_l * cell.e_l)
            / g_total;
        cell.v = steady_v + (cell.v - steady_v) * (-(g_total / cell.c_m) * dt_sub).exp();
    }
    cell
}

fn snapshot(cell: &GolgiCell) -> (f64, f64, f64, f64, f64, f64) {
    (cell.v, cell.m, cell.h, cell.p_na, cell.n, cell.ca)
}

#[test]
fn golgi_uses_exact_gate_calcium_and_conductance_step() {
    let mut cell = GolgiCell::new();
    let expected = reference_step(GolgiCell::new(), 5.0);

    assert_eq!(cell.step(5.0), 0);

    assert!((cell.v - expected.v).abs() <= 1e-12);
    assert!((cell.m - expected.m).abs() <= 1e-12);
    assert!((cell.h - expected.h).abs() <= 1e-12);
    assert!((cell.p_na - expected.p_na).abs() <= 1e-12);
    assert!((cell.n - expected.n).abs() <= 1e-12);
    assert!((cell.ca - expected.ca).abs() <= 1e-12);
}

#[test]
fn golgi_invalid_current_preserves_state() {
    let mut cell = GolgiCell::new();
    for _ in 0..10 {
        cell.step(5.0);
    }
    let before = snapshot(&cell);

    assert_eq!(cell.step(f64::NAN), 0);
    assert_eq!(snapshot(&cell), before);
    assert_eq!(cell.step(f64::INFINITY), 0);
    assert_eq!(snapshot(&cell), before);
}

#[test]
fn golgi_excess_current_preserves_state() {
    let mut cell = GolgiCell::new();
    let before = snapshot(&cell);

    assert_eq!(cell.step(1.0e8), 0);

    assert_eq!(snapshot(&cell), before);
}

#[test]
fn golgi_all_currents_bounded_and_calcium_active() {
    let mut cell = GolgiCell::new();
    let baseline_ca = cell.ca;
    let spikes: i32 = (0..2000).map(|_| cell.step(10.0)).sum();

    assert!(spikes > 0);
    assert!(cell.ca > baseline_ca);
    for gate in [
        cell.m, cell.h, cell.p_na, cell.n, cell.a, cell.b, cell.w, cell.m_t, cell.s, cell.c_n,
        cell.r,
    ] {
        assert!((0.0..=1.0).contains(&gate));
    }
    assert!((-100.0..=60.0).contains(&cell.v));
    assert!(cell.ca >= 0.0);
}
