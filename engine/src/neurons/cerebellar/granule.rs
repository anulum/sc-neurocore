// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Granule Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar granule cell — D'Angelo et al. 2001 full model.
///
/// Most numerous neuron in the brain (~50%). Tiny soma (6-8 µm),
/// four short dendrites receiving mossy fibre input at glomeruli,
/// output via parallel fibres to Purkinje cells.
///
/// Full Hodgkin-Huxley-type model with 7 ionic currents:
/// - **INa** (transient Na, m³h): fast spike generation
/// - **IK_dr** (delayed rectifier K, n⁴): repolarisation
/// - **IK_A** (A-type K, a³b): delay to first spike, inter-spike interval
/// - **ICa_T** (T-type Ca²⁺, m_t²s): post-inhibitory rebound bursting
/// - **IK_Ca** (Ca²⁺-activated K, Hill): slow AHP
/// - **Ih** (HCN, r): sag current, resting potential stabilisation
/// - **IL** (leak)
/// - **IGABA** (tonic GABA from Golgi cells)
///
/// Uses 4 sub-steps (dt_sub = 0.125 ms) for Na gating stability.
///
/// D'Angelo et al., J Neurosci 21(3):759, 2001.
/// D'Angelo & De Zeeuw, Trends Neurosci 32:30, 2009 (review).
#[derive(Clone, Debug)]
pub struct GranuleCell {
    pub v: f64,       // Membrane potential (mV)
    pub m: f64,       // Na activation
    pub h: f64,       // Na inactivation
    pub n: f64,       // K_dr activation
    pub a: f64,       // K_A activation
    pub b: f64,       // K_A inactivation
    pub m_t: f64,     // T-type Ca²⁺ activation
    pub s: f64,       // T-type Ca²⁺ inactivation
    pub ca: f64,      // Intracellular Ca²⁺ (µM)
    pub r: f64,       // Ih activation
    pub c_m: f64,     // Capacitance (µF/cm²)
    pub g_na: f64,    // Na conductance
    pub g_kdr: f64,   // K_dr conductance
    pub g_ka: f64,    // K_A conductance
    pub g_t: f64,     // T-type Ca²⁺ conductance
    pub g_kca: f64,   // Ca²⁺-dependent K conductance
    pub g_h: f64,     // Ih conductance
    pub g_l: f64,     // Leak conductance
    pub g_tonic: f64, // Tonic GABA conductance
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64, // Ih reversal (~-40 mV, mixed cation)
    pub e_l: f64,
    pub e_gaba: f64,
    pub tau_ca: f64, // Ca²⁺ decay (ms)
    pub kd_kca: f64, // K_Ca half-saturation (µM)
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for GranuleCell {
    fn default() -> Self {
        Self::new()
    }
}

impl GranuleCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.02,
            h: 0.85,
            n: 0.05,
            a: 0.1,
            b: 0.8,
            m_t: 0.01,
            s: 0.95,
            ca: 0.05,
            r: 0.1,
            c_m: 1.0,
            g_na: 17.0,   // mS/cm² (D'Angelo 2001 Table 1)
            g_kdr: 9.0,   // Delayed rectifier
            g_ka: 1.0,    // A-type K
            g_t: 0.5,     // T-type Ca²⁺
            g_kca: 3.5,   // Ca²⁺-activated K
            g_h: 0.03,    // Ih (small in granule cells)
            g_l: 0.1,     // Leak
            g_tonic: 0.2, // Tonic GABA (strong tonic inhibition)
            e_na: 87.4,   // D'Angelo 2001
            e_k: -84.7,
            e_ca: 129.3,
            e_h: -40.0, // Mixed cation
            e_l: -58.0, // D'Angelo 2001
            e_gaba: -75.0,
            tau_ca: 10.0, // Ca²⁺ decay
            kd_kca: 0.2,  // K_Ca half-sat (µM)
            dt: 0.5,
            sub_steps: 4, // dt_sub = 0.125 ms
            gain: 1.0,
        }
    }

    /// Boltzmann steady-state.
    #[inline]
    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        let z = -(v - vh) / k;
        if z > 60.0 {
            0.0
        } else if z < -60.0 {
            1.0
        } else {
            1.0 / (1.0 + z.exp())
        }
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.m,
            self.h,
            self.n,
            self.a,
            self.b,
            self.m_t,
            self.s,
            self.ca,
            self.r,
            self.c_m,
            self.g_na,
            self.g_kdr,
            self.g_ka,
            self.g_t,
            self.g_kca,
            self.g_h,
            self.g_l,
            self.g_tonic,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_h,
            self.e_l,
            self.e_gaba,
            self.tau_ca,
            self.kd_kca,
            self.dt,
            self.gain,
        ]
        .iter()
        .all(|value| value.is_finite())
            && [
                self.m, self.h, self.n, self.a, self.b, self.m_t, self.s, self.r,
            ]
            .iter()
            .all(|gate| (0.0..=1.0).contains(gate))
            && (-100.0..=60.0).contains(&self.v)
            && self.ca >= 0.0
            && [
                self.g_na,
                self.g_kdr,
                self.g_ka,
                self.g_t,
                self.g_kca,
                self.g_h,
                self.g_l,
                self.g_tonic,
            ]
            .iter()
            .all(|conductance| *conductance >= 0.0)
            && self.c_m > 0.0
            && self.tau_ca > 0.0
            && self.kd_kca > 0.0
            && self.dt > 0.0
            && self.sub_steps > 0
            && self.gain >= 0.0
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }

        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;
        let mut v = self.v;
        let mut m = self.m;
        let mut h = self.h;
        let mut n = self.n;
        let mut a = self.a;
        let mut b = self.b;
        let mut m_t = self.m_t;
        let mut s_gate = self.s;
        let mut ca = self.ca;
        let mut r = self.r;

        for _ in 0..self.sub_steps {
            // Na m gate (fast activation, Boltzmann + tau)
            let m_inf = Self::boltz(v, -30.0, 7.0);
            let tau_m = 0.1 + 0.3 / (1.0 + ((v + 30.0) / 10.0).powi(2)).max(0.01);
            m = granule_exact_relax(m, m_inf, tau_m, dt_sub).clamp(0.0, 1.0);

            // Na h gate (inactivation)
            let h_inf = Self::boltz(v, -52.0, -6.0);
            let tau_h = 0.5 + 5.0 / (1.0 + ((v + 50.0) / 15.0).powi(2)).max(0.01);
            h = granule_exact_relax(h, h_inf, tau_h, dt_sub).clamp(0.0, 1.0);

            // K_dr n gate
            let n_inf = Self::boltz(v, -35.0, 8.0);
            let tau_n = 1.0 + 5.0 / (1.0 + ((v + 35.0) / 15.0).powi(2)).max(0.01);
            n = granule_exact_relax(n, n_inf, tau_n, dt_sub).clamp(0.0, 1.0);

            // K_A a gate (fast activation)
            let a_inf = Self::boltz(v, -50.0, 20.0);
            let tau_a = 2.0;
            a = granule_exact_relax(a, a_inf, tau_a, dt_sub).clamp(0.0, 1.0);

            // K_A b gate (slow inactivation)
            let b_inf = Self::boltz(v, -70.0, -6.0);
            let tau_b = 50.0;
            b = granule_exact_relax(b, b_inf, tau_b, dt_sub).clamp(0.0, 1.0);

            // T-type Ca²⁺ m_t (fast activation)
            let mt_inf = Self::boltz(v, -52.0, 5.0);
            let tau_mt = 1.0;
            m_t = granule_exact_relax(m_t, mt_inf, tau_mt, dt_sub).clamp(0.0, 1.0);

            // T-type Ca²⁺ s (slow inactivation)
            let s_inf = Self::boltz(v, -60.0, -6.5);
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).powi(2)).max(0.01);
            s_gate = granule_exact_relax(s_gate, s_inf, tau_s, dt_sub).clamp(0.0, 1.0);

            // Ih r gate (slow activation at hyperpolarised V)
            let r_inf = Self::boltz(v, -80.0, -10.0);
            let tau_r = 50.0 + 200.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
            r = granule_exact_relax(r, r_inf, tau_r, dt_sub).clamp(0.0, 1.0);

            // Ca²⁺ dynamics
            let i_ca_t = self.g_t * m_t * m_t * s_gate * (v - self.e_ca);
            let ca_entry = if i_ca_t < 0.0 { -i_ca_t * 0.001 } else { 0.0 }; // Inward Ca²⁺
            ca = granule_exact_relax(ca, ca_entry * self.tau_ca, self.tau_ca, dt_sub).max(0.0);

            // K_Ca (Hill function of Ca²⁺)
            let kca_inf = ca * ca / (ca * ca + self.kd_kca * self.kd_kca);

            // Ionic conductances with exact voltage relaxation.
            let g_na_eff = self.g_na * m.powi(3) * h;
            let g_kdr_eff = self.g_kdr * n.powi(4);
            let g_ka_eff = self.g_ka * a.powi(3) * b;
            let g_t_eff = self.g_t * m_t * m_t * s_gate;
            let g_kca_eff = self.g_kca * kca_inf;
            let g_h_eff = self.g_h * r;
            v = granule_exact_voltage_step(
                v,
                input,
                self.c_m,
                dt_sub,
                &[
                    (g_na_eff, self.e_na),
                    (g_kdr_eff, self.e_k),
                    (g_ka_eff, self.e_k),
                    (g_t_eff, self.e_ca),
                    (g_kca_eff, self.e_k),
                    (g_h_eff, self.e_h),
                    (self.g_l, self.e_l),
                    (self.g_tonic, self.e_gaba),
                ],
            )
            .clamp(-100.0, 60.0);

            if ![v, m, h, n, a, b, m_t, s_gate, ca, r]
                .iter()
                .all(|value| value.is_finite())
            {
                return 0;
            }
        }

        self.v = v;
        self.m = m;
        self.h = h;
        self.n = n;
        self.a = a;
        self.b = b;
        self.m_t = m_t;
        self.s = s_gate;
        self.ca = ca;
        self.r = r;

        // Spike: V crosses 0 mV
        if self.v >= 0.0 && v_prev < 0.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

fn granule_exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
    target + (value - target) * (-dt / tau).exp()
}

fn granule_exact_voltage_step(
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

    // -- Granule Cell tests --

    #[test]
    fn granule_fires_with_strong_input() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(15.0);
        }
        assert!(
            spikes > 10,
            "Granule cell must fire with strong excitatory input, got {spikes}"
        );
    }

    #[test]
    fn granule_silent_at_rest() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Granule cell must be silent without input (tonic GABA inhibition)"
        );
    }

    #[test]
    fn granule_no_fire_weak_input() {
        // Tonic GABA raises effective threshold
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(1.0);
        }
        assert!(
            spikes == 0,
            "Weak input should not overcome tonic GABA, got {spikes}"
        );
    }

    #[test]
    fn granule_tonic_gaba_raises_threshold() {
        // Compare firing with and without tonic GABA
        let mut with_gaba = GranuleCell::new();
        let mut no_gaba = GranuleCell::new();
        no_gaba.g_tonic = 0.0;

        let input = 8.0;
        let mut spikes_gaba = 0;
        let mut spikes_no_gaba = 0;
        for _ in 0..10_000 {
            spikes_gaba += with_gaba.step(input);
            spikes_no_gaba += no_gaba.step(input);
        }
        assert!(
            spikes_no_gaba > spikes_gaba,
            "Removing tonic GABA must increase firing: no_gaba={spikes_no_gaba} vs gaba={spikes_gaba}"
        );
    }

    #[test]
    fn granule_has_seven_currents() {
        // D'Angelo 2001 model must have all 7 ionic currents
        let n = GranuleCell::new();
        assert!(n.g_na > 0.0, "Must have INa");
        assert!(n.g_kdr > 0.0, "Must have IK_dr");
        assert!(n.g_ka > 0.0, "Must have IK_A");
        assert!(n.g_t > 0.0, "Must have ICa_T");
        assert!(n.g_kca > 0.0, "Must have IK_Ca");
        assert!(n.g_h > 0.0, "Must have Ih");
        assert!(n.g_l > 0.0, "Must have IL");
    }

    #[test]
    fn granule_t_type_deinactivates_at_rest() {
        // T-type inactivation s should be high at rest (de-inactivated)
        let mut n = GranuleCell::new();
        for _ in 0..5000 {
            n.step(0.0);
        }
        assert!(
            n.s > 0.5,
            "T-type must be partially de-inactivated at rest, s={}",
            n.s
        );
    }

    #[test]
    fn granule_gate_and_calcium_kinetics_use_closed_form_relaxation() {
        let mut n = GranuleCell::new();
        n.g_na = 0.0;
        n.g_kdr = 0.0;
        n.g_ka = 0.0;
        n.g_t = 0.0;
        n.g_kca = 0.0;
        n.g_h = 0.0;
        n.g_l = 0.0;
        n.g_tonic = 0.0;
        n.gain = 0.0;
        n.sub_steps = 1;
        let (v0, m0, h0, n0, a0, b0, mt0, s0, ca0, r0) =
            (n.v, n.m, n.h, n.n, n.a, n.b, n.m_t, n.s, n.ca, n.r);
        let m_inf = GranuleCell::boltz(v0, -30.0, 7.0);
        let tau_m = 0.1 + 0.3 / (1.0 + ((v0 + 30.0) / 10.0).powi(2)).max(0.01);
        let h_inf = GranuleCell::boltz(v0, -52.0, -6.0);
        let tau_h = 0.5 + 5.0 / (1.0 + ((v0 + 50.0) / 15.0).powi(2)).max(0.01);
        let n_inf = GranuleCell::boltz(v0, -35.0, 8.0);
        let tau_n = 1.0 + 5.0 / (1.0 + ((v0 + 35.0) / 15.0).powi(2)).max(0.01);
        let a_inf = GranuleCell::boltz(v0, -50.0, 20.0);
        let b_inf = GranuleCell::boltz(v0, -70.0, -6.0);
        let mt_inf = GranuleCell::boltz(v0, -52.0, 5.0);
        let s_inf = GranuleCell::boltz(v0, -60.0, -6.5);
        let tau_s = 20.0 + 50.0 / (1.0 + ((v0 + 65.0) / 10.0).powi(2)).max(0.01);
        let r_inf = GranuleCell::boltz(v0, -80.0, -10.0);
        let tau_r = 50.0 + 200.0 / (1.0 + ((v0 + 80.0) / 20.0).powi(2)).max(0.01);

        n.step(0.0);

        assert_close_granule(n.v, v0);
        assert_close_granule(n.m, granule_exact_relax(m0, m_inf, tau_m, n.dt));
        assert_close_granule(n.h, granule_exact_relax(h0, h_inf, tau_h, n.dt));
        assert_close_granule(n.n, granule_exact_relax(n0, n_inf, tau_n, n.dt));
        assert_close_granule(n.a, granule_exact_relax(a0, a_inf, 2.0, n.dt));
        assert_close_granule(n.b, granule_exact_relax(b0, b_inf, 50.0, n.dt));
        assert_close_granule(n.m_t, granule_exact_relax(mt0, mt_inf, 1.0, n.dt));
        assert_close_granule(n.s, granule_exact_relax(s0, s_inf, tau_s, n.dt));
        assert_close_granule(n.ca, granule_exact_relax(ca0, 0.0, n.tau_ca, n.dt));
        assert_close_granule(n.r, granule_exact_relax(r0, r_inf, tau_r, n.dt));
    }

    #[test]
    fn granule_ca_rises_with_spiking() {
        // Ca²⁺ should increase during spiking activity
        let mut n = GranuleCell::new();
        let ca0 = n.ca;
        for _ in 0..5000 {
            n.step(8.0);
        }
        assert!(
            n.ca > ca0,
            "Ca²⁺ should rise in the T-current firing regime: ca0={ca0}, ca_now={}",
            n.ca
        );
    }

    #[test]
    fn granule_negative_input_no_crash() {
        let mut n = GranuleCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite(), "Must stay finite with negative input");
        assert!(n.v >= -100.0, "Must be bounded");
    }

    #[test]
    fn granule_nan_input_stays_finite() {
        let mut n = GranuleCell::new();
        let before = n.clone();
        n.step(f64::NAN);
        assert!(n.v.is_finite(), "NaN input must not corrupt state");
        assert_eq!(n.v, before.v);
        assert_eq!(n.ca, before.ca);
        assert_eq!(n.s, before.s);
    }

    #[test]
    fn granule_corrupted_state_preserved_on_step() {
        let mut n = GranuleCell::new();
        n.m = -0.1;
        let before = n.clone();
        assert_eq!(n.step(10.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.m, before.m);
        assert_eq!(n.ca, before.ca);
    }

    #[test]
    fn granule_extreme_input_bounded() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(
            n.v.is_finite() && n.v <= 60.0,
            "Extreme input must stay bounded"
        );
    }

    #[test]
    fn granule_reset_clears_state() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert_eq!(n.v, -70.0);
        assert_eq!(n.s, 0.95);
        assert_eq!(n.m, 0.02);
    }

    #[test]
    fn granule_high_input_resistance() {
        // Small soma → large voltage response to small current
        let mut n = GranuleCell::new();
        let v_before = n.v;
        // Single step with moderate input
        n.step(5.0);
        let dv = n.v - v_before;
        assert!(
            dv > 0.5,
            "High Rin should give large voltage change, got dv={dv}"
        );
    }

    #[test]
    fn granule_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = GranuleCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(10.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "10k exact-integrator steps must complete in <100ms, took {}ms",
            elapsed.as_millis()
        );
    }

    fn assert_close_granule(observed: f64, expected: f64) {
        assert!(
            (observed - expected).abs() <= 1.0e-12,
            "observed {:.17e}, expected {:.17e}",
            observed,
            expected,
        );
    }

    #[test]
    fn granule_default_matches_constructor_contract() {
        let default = GranuleCell::default();
        let constructed = GranuleCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.ca, constructed.ca);
        assert_eq!(default.dt, constructed.dt);
        assert_eq!(default.sub_steps, constructed.sub_steps);
    }
}
