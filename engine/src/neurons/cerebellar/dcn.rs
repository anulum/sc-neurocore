// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

use super::super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Deep Cerebellar Nuclei Neuron
// ═══════════════════════════════════════════════════════════════════

/// Deep cerebellar nuclei (DCN) neuron — main output of the cerebellum.
///
/// Biophysics: WB Na+/K+ core with T-type Ca²⁺ for post-inhibitory rebound
/// bursting, Ih (HCN) for pacemaker-like activity, persistent Na (INaP) for
/// subthreshold depolarisation, and Ca²⁺-dependent AHP for spike frequency
/// adaptation.
///
/// 7 currents: INa_t, INaP, IK_dr, ICa_T, IAHP, Ih, IL
///
/// Rebound bursting: when Purkinje inhibition is released, T-type Ca²⁺
/// channels that de-inactivated during hyperpolarisation produce a burst.
/// INaP amplifies subthreshold depolarisation. AHP limits burst duration.
///
/// Llinás & Mühlethaler, J Physiol 404:241, 1988; Jahnsen, J Physiol 372:129, 1986.
#[derive(Clone, Debug)]
pub struct DCNNeuron {
    pub v: f64,
    pub h: f64,  // Na_t inactivation
    pub n: f64,  // K_dr activation
    pub p: f64,  // Na_p persistent activation
    pub s: f64,  // T-type Ca²⁺ inactivation (slow)
    pub r: f64,  // Ih activation
    pub ca: f64, // Intracellular Ca²⁺ (µM)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_nap: f64, // Persistent Na
    pub g_k: f64,
    pub g_t: f64,   // T-type Ca²⁺
    pub g_ahp: f64, // Ca²⁺-dependent AHP
    pub g_h: f64,   // Ih
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub kd_ahp: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for DCNNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl DCNNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            h: 0.6,
            n: 0.32,
            p: 0.01,  // NaP activation (low at rest)
            s: 0.8,   // T-type de-inactivated at rest
            r: 0.1,   // Ih partially active
            ca: 0.05, // Resting Ca²⁺ (µM)
            g_na: 35.0,
            g_nap: 0.5, // Persistent Na — amplifies subthreshold
            g_k: 9.0,
            g_t: 0.1,   // T-type Ca²⁺
            g_ahp: 2.0, // Ca²⁺-dependent AHP
            g_h: 0.02,  // Ih — modest
            g_l: 0.2,   // Leak
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_h: -40.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_ca: 150.0,
            kd_ahp: 0.5,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }
        let input = self.gain * current;
        let sub_steps = 20;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;
        let (mut v, mut h, mut n, mut p, mut s, mut r, mut ca) =
            (self.v, self.h, self.n, self.p, self.s, self.r, self.ca);

        for _ in 0..sub_steps {
            // Na_t: WB alpha/beta rates (m³h, m quasi-static)
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            // K_dr: n⁴
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Na_p: persistent Na (Boltzmann, V1/2=-48, k=5)
            let p_inf = 1.0 / (1.0 + (-(v + 48.0) / 5.0).exp());
            let tau_p = 5.0 + 15.0 / (1.0 + ((v + 48.0) / 10.0).powi(2)).max(0.01);

            // T-type Ca²⁺ gating
            let m_t_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).exp());
            let s_inf = 1.0 / (1.0 + ((v + 60.0) / 6.5).exp());
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).exp());

            // Ih gating
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());

            // First-order gates: exact exponential relaxation for the
            // voltage-frozen sub-step, avoiding Euler overshoot in stiff
            // rebound trajectories.
            h = dcn_exact_hh_gate(h, alpha_h, beta_h, self.phi, sub_dt);
            n = dcn_exact_hh_gate(n, alpha_n, beta_n, self.phi, sub_dt);
            p = dcn_exact_relax(p, p_inf, tau_p, sub_dt);
            s = dcn_exact_relax(s, s_inf, tau_s, sub_dt);
            r = dcn_exact_relax(r, r_inf, tau_r, sub_dt);

            // Ca²⁺ dynamics: entry via T-type, decay
            let i_t = self.g_t * m_t_inf.powi(2) * s * (v - self.e_ca);
            let ca_entry = if i_t < 0.0 { -i_t * 0.001 } else { 0.0 };
            ca = dcn_exact_relax(ca, ca_entry * self.tau_ca, self.tau_ca, sub_dt).max(0.0);

            // AHP: Ca²⁺-dependent K (Hill n=2)
            let ahp_inf = ca.powi(2) / (ca.powi(2) + self.kd_ahp.powi(2));

            // Voltage: exact ohmic conductance solution over the sub-step
            // with gates frozen after their exponential update.
            let g_na_eff = self.g_na * m_inf.powi(3) * h;
            let g_nap_eff = self.g_nap * p;
            let g_k_eff = self.g_k * n.powi(4);
            let g_t_eff = self.g_t * m_t_inf.powi(2) * s;
            let g_ahp_eff = self.g_ahp * ahp_inf;
            let g_h_eff = self.g_h * r;
            v = dcn_exact_voltage_step(
                v,
                input,
                self.c_m,
                sub_dt,
                &[
                    (g_na_eff, self.e_na),
                    (g_nap_eff, self.e_na),
                    (g_k_eff, self.e_k),
                    (g_t_eff, self.e_ca),
                    (g_ahp_eff, self.e_k),
                    (g_h_eff, self.e_h),
                    (self.g_l, self.e_l),
                ],
            );

            if v >= self.v_threshold {
                fired = 1;
                v = -60.0;
                s *= 0.5; // T-type inactivation on spike
                ca += 0.5; // Ca²⁺ entry on spike
            }
        }

        if ![v, h, n, p, s, r, ca].iter().all(|value| value.is_finite()) {
            return 0;
        }
        self.v = v.clamp(-100.0, 60.0);
        self.h = h.clamp(0.0, 1.0);
        self.n = n.clamp(0.0, 1.0);
        self.p = p.clamp(0.0, 1.0);
        self.s = s.clamp(0.0, 1.0);
        self.r = r.clamp(0.0, 1.0);
        self.ca = ca.max(0.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.h,
            self.n,
            self.p,
            self.s,
            self.r,
            self.ca,
            self.g_na,
            self.g_nap,
            self.g_k,
            self.g_t,
            self.g_ahp,
            self.g_h,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_h,
            self.e_l,
            self.c_m,
            self.phi,
            self.tau_ca,
            self.kd_ahp,
            self.dt,
            self.v_threshold,
            self.gain,
        ]
        .iter()
        .all(|value| value.is_finite())
            && [self.h, self.n, self.p, self.s, self.r]
                .iter()
                .all(|gate| (0.0..=1.0).contains(gate))
            && self.ca >= 0.0
            && (-100.0..=60.0).contains(&self.v)
            && [
                self.g_na, self.g_nap, self.g_k, self.g_t, self.g_ahp, self.g_h, self.g_l,
            ]
            .iter()
            .all(|g| *g >= 0.0)
            && self.c_m > 0.0
            && self.phi > 0.0
            && self.tau_ca > 0.0
            && self.kd_ahp > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
    }
}

fn dcn_exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
    target + (value - target) * (-dt / tau).exp()
}

fn dcn_exact_hh_gate(value: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
    let rate = phi * (alpha + beta);
    let target = alpha / (alpha + beta);
    target + (value - target) * (-rate * dt).exp()
}

fn dcn_exact_voltage_step(
    v: f64,
    input_current: f64,
    c_m: f64,
    dt: f64,
    conductances: &[(f64, f64)],
) -> f64 {
    let g_total: f64 = conductances.iter().map(|(g, _)| *g).sum();
    if g_total <= 0.0 {
        return v + dt * input_current / c_m;
    }
    let reversal_drive: f64 = conductances.iter().map(|(g, e_rev)| g * e_rev).sum();
    let v_inf = (input_current + reversal_drive) / g_total;
    v_inf + (v - v_inf) * (-dt * g_total / c_m).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- DCN Neuron tests --

    #[test]
    fn dcn_fires_with_input() {
        let mut n = DCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 3,
            "DCN must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn dcn_spontaneous_activity() {
        // DCN neurons fire spontaneously (Llinás & Mühlethaler 1988)
        // INaP + Ih + depolarised leak drive autonomous firing
        let mut n = DCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..20_000 {
            spikes += n.step(0.0);
        }
        // Should show some spontaneous activity (low rate)
        // Without INaP, should be reduced
        let mut no_nap = DCNNeuron::new();
        no_nap.g_nap = 0.0;
        let mut spikes_no = 0;
        for _ in 0..20_000 {
            spikes_no += no_nap.step(0.0);
        }
        assert!(
            spikes >= spikes_no,
            "INaP should contribute to spontaneous firing: with={spikes}, without={spikes_no}"
        );
    }

    #[test]
    fn dcn_rebound_burst() {
        // Hyperpolarisation → T-type de-inactivation → rebound burst
        let mut n = DCNNeuron::new();
        // Hyperpolarise to de-inactivate T-type
        for _ in 0..2000 {
            n.step(-5.0);
        }
        assert!(
            n.s > 0.5,
            "T-type must de-inactivate during hyperpolarisation, s={}",
            n.s
        );

        // Now provide excitation — T-type should help fire
        let mut spikes = 0;
        for _ in 0..200 {
            spikes += n.step(3.0);
        }
        // Compare with pre-inactivated T-type
        let mut n2 = DCNNeuron::new();
        n2.s = 0.05; // pre-inactivated
        let mut spikes2 = 0;
        for _ in 0..200 {
            spikes2 += n2.step(3.0);
        }
        assert!(
            spikes >= spikes2,
            "De-inactivated T-type should facilitate rebound: rebound={spikes} vs inact={spikes2}"
        );
    }

    #[test]
    fn dcn_ih_depolarises() {
        // Ih should depolarise from hyperpolarised potentials
        let mut with_ih = DCNNeuron::new();
        with_ih.v = -80.0;
        let mut no_ih = DCNNeuron::new();
        no_ih.v = -80.0;
        no_ih.g_h = 0.0;

        for _ in 0..1000 {
            with_ih.step(0.0);
            no_ih.step(0.0);
        }
        assert!(
            with_ih.v > no_ih.v,
            "Ih should depolarise from hyperpolarised state: Ih={:.1} vs no_Ih={:.1}",
            with_ih.v,
            no_ih.v
        );
    }

    #[test]
    fn dcn_gate_and_calcium_kinetics_use_closed_form_relaxation() {
        let mut n = DCNNeuron::new();
        n.g_na = 0.0;
        n.g_nap = 0.0;
        n.g_k = 0.0;
        n.g_t = 0.0;
        n.g_ahp = 0.0;
        n.g_h = 0.0;
        n.g_l = 0.0;
        n.gain = 0.0;
        let (v0, h0, n0, p0, s0, r0, ca0) = (n.v, n.h, n.n, n.p, n.s, n.r, n.ca);
        let alpha_h = 0.07 * (-(v0 + 58.0) / 20.0).exp();
        let beta_h = 1.0 / (1.0 + (-(v0 + 28.0) / 10.0).exp());
        let alpha_n = safe_rate(0.01, 34.0, v0, 10.0, 0.1);
        let beta_n = 0.125 * (-(v0 + 44.0) / 80.0).exp();
        let p_inf = 1.0 / (1.0 + (-(v0 + 48.0) / 5.0).exp());
        let tau_p = 5.0 + 15.0 / (1.0 + ((v0 + 48.0) / 10.0).powi(2)).max(0.01);
        let s_inf = 1.0 / (1.0 + ((v0 + 60.0) / 6.5).exp());
        let tau_s = 20.0 + 50.0 / (1.0 + ((v0 + 65.0) / 10.0).exp());
        let r_inf = 1.0 / (1.0 + ((v0 + 80.0) / 10.0).exp());
        let tau_r = 100.0 + 200.0 / (1.0 + ((v0 + 70.0) / 10.0).exp());

        n.step(0.0);

        assert_close(n.v, v0);
        assert_close(n.h, dcn_exact_hh_gate(h0, alpha_h, beta_h, n.phi, n.dt));
        assert_close(n.n, dcn_exact_hh_gate(n0, alpha_n, beta_n, n.phi, n.dt));
        assert_close(n.p, dcn_exact_relax(p0, p_inf, tau_p, n.dt));
        assert_close(n.s, dcn_exact_relax(s0, s_inf, tau_s, n.dt));
        assert_close(n.r, dcn_exact_relax(r0, r_inf, tau_r, n.dt));
        assert_close(n.ca, dcn_exact_relax(ca0, 0.0, n.tau_ca, n.dt));
    }

    #[test]
    fn dcn_negative_input_no_crash() {
        let mut n = DCNNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn dcn_nan_input_stays_finite() {
        let mut n = DCNNeuron::new();
        let before = n.clone();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert_eq!(n.v, before.v);
        assert_eq!(n.ca, before.ca);
    }

    #[test]
    fn dcn_extreme_input_bounded() {
        let mut n = DCNNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn dcn_corrupted_state_preserved_on_step() {
        let mut n = DCNNeuron::new();
        n.h = -0.1;
        let before_v = n.v;
        let before_ca = n.ca;
        assert_eq!(n.step(10.0), 0);
        assert_eq!(n.v, before_v);
        assert_eq!(n.ca, before_ca);
    }

    #[test]
    fn dcn_reset_clears_state() {
        let mut n = DCNNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -60.0);
        assert_eq!(n.s, 0.8);
        assert_eq!(n.r, 0.1);
    }

    #[test]
    fn dcn_gates_bounded() {
        let mut n = DCNNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        for (name, val) in [("h", n.h), ("n", n.n), ("p", n.p), ("s", n.s), ("r", n.r)] {
            assert!((0.0..=1.0).contains(&val), "{name} out of bounds: {val}");
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative: {}", n.ca);
    }

    #[test]
    fn dcn_nap_increases_excitability() {
        // INaP amplifies subthreshold depolarisation
        let mut with_nap = DCNNeuron::new();
        let mut no_nap = DCNNeuron::new();
        no_nap.g_nap = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..5_000 {
            spikes_with += with_nap.step(3.0);
            spikes_no += no_nap.step(3.0);
        }
        assert!(
            spikes_with >= spikes_no,
            "INaP should increase excitability: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn dcn_ahp_limits_rate() {
        // Ca²⁺-AHP should reduce sustained firing rate
        let mut with_ahp = DCNNeuron::new();
        let mut no_ahp = DCNNeuron::new();
        no_ahp.g_ahp = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..5_000 {
            spikes_with += with_ahp.step(8.0);
            spikes_no += no_ahp.step(8.0);
        }
        assert!(
            spikes_no >= spikes_with,
            "AHP removal should increase firing: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn dcn_ca_rises_during_spiking() {
        let mut n = DCNNeuron::new();
        let ca_init = n.ca;
        for _ in 0..5_000 {
            n.step(10.0);
        }
        assert!(
            n.ca > ca_init,
            "Ca²⁺ must rise during spiking: init={ca_init}, now={}",
            n.ca
        );
    }

    #[test]
    fn dcn_has_seven_currents() {
        // Na_t, Na_p, K_dr, Ca_T, AHP, Ih, leak = 7
        let n = DCNNeuron::new();
        assert!(n.g_na > 0.0, "Na_t missing");
        assert!(n.g_nap > 0.0, "Na_p missing");
        assert!(n.g_k > 0.0, "K_dr missing");
        assert!(n.g_t > 0.0, "Ca_T missing");
        assert!(n.g_ahp > 0.0, "AHP missing");
        assert!(n.g_h > 0.0, "Ih missing");
        assert!(n.g_l > 0.0, "Leak missing");
    }

    #[test]
    fn dcn_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = DCNNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    fn assert_close(observed: f64, expected: f64) {
        assert!(
            (observed - expected).abs() <= 1.0e-12,
            "observed {:.17e}, expected {:.17e}",
            observed,
            expected,
        );
    }

    #[test]
    fn dcn_default_matches_constructor_contract() {
        let default = DCNNeuron::default();
        let constructed = DCNNeuron::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.s, constructed.s);
        assert_eq!(default.ca, constructed.ca);
        assert_eq!(default.dt, constructed.dt);
    }
}
