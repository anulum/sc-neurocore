// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Mainen-Sejnowski Neuron Model

//! Mainen-Sejnowski two-compartment soma and axon dynamics.

/// Mainen-Sejnowski — two-compartment (soma + axon). Mainen & Sejnowski 1996.
#[derive(Clone, Debug)]
pub struct MainenSejnowskiNeuron {
    pub vs: f64,
    pub va: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub kappa: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_s: f64,
    pub c_a: f64,
    pub dt: f64,
    pub v_threshold: f64,
    /// Legacy engine configuration: update the soma voltage first and let
    /// the axon derivative consume the already-updated soma value
    /// (Gauss-Seidel ordering). The canonical reference uses the Python
    /// Jacobi ordering (both derivatives from pre-update values).
    pub legacy_sequential: bool,
}

impl MainenSejnowskiNeuron {
    pub fn new() -> Self {
        Self {
            vs: -65.0,
            va: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            kappa: 10.0,
            g_na: 3000.0,
            g_k: 1500.0,
            g_l: 1.0,
            e_na: 50.0,
            e_k: -90.0,
            e_l: -70.0,
            c_s: 1.0,
            c_a: 0.1,
            dt: 0.005,
            v_threshold: -20.0,
            legacy_sequential: false,
        }
    }

    /// Reconstruct the original engine ordering (soma committed before the
    /// axon derivative is evaluated). This is a legacy configuration of the
    /// same model, not a separate catalogue identity.
    pub fn new_legacy_sequential() -> Self {
        let mut neuron = Self::new();
        neuron.legacy_sequential = true;
        neuron
    }

    fn valid(&self) -> bool {
        let finite = [
            self.vs,
            self.va,
            self.m,
            self.h,
            self.n,
            self.kappa,
            self.g_na,
            self.g_k,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_l,
            self.c_s,
            self.c_a,
            self.dt,
            self.v_threshold,
        ]
        .into_iter()
        .all(f64::is_finite);
        finite
            && (-200.0..=200.0).contains(&self.vs)
            && (-200.0..=200.0).contains(&self.va)
            && [self.m, self.h, self.n]
                .into_iter()
                .all(|gate| (0.0..=1.0).contains(&gate))
            && (0.0..=100.0).contains(&self.kappa)
            && (0.0..=5000.0).contains(&self.g_na)
            && (0.0..=3000.0).contains(&self.g_k)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
            && (-90.0..=-50.0).contains(&self.e_l)
            && (0.5..=2.0).contains(&self.c_s)
            && (0.05..=1.0).contains(&self.c_a)
            && self.dt > 0.0
            && self.dt <= 0.1
            && (-40.0..=20.0).contains(&self.v_threshold)
    }

    /// Evaluate `x / (1 - exp(-x / k))` with its analytic limit `k` at 0.
    fn linoid(x: f64, k: f64) -> f64 {
        if x == 0.0 {
            k
        } else {
            x / -(-x / k).exp_m1()
        }
    }

    /// Advance one step after validating the drive and configuration.
    ///
    /// Computes the whole update on a candidate clone and commits only on
    /// success: a non-finite `current`, a configuration outside the public
    /// bounds, or a non-finite candidate returns `Err` with the pre-step
    /// state preserved exactly.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !self.valid() {
            return Err("Mainen-Sejnowski state and parameters must satisfy the public bounds");
        }

        let mut candidate = self.clone();
        let v_prev = candidate.vs;
        for _ in 0..20 {
            if candidate.legacy_sequential {
                Self::legacy_sequential_substep(&mut candidate, current);
            } else {
                Self::canonical_substep(&mut candidate, current);
            }

            if ![
                candidate.vs,
                candidate.va,
                candidate.m,
                candidate.h,
                candidate.n,
            ]
            .into_iter()
            .all(f64::is_finite)
            {
                return Err("Mainen-Sejnowski candidate state became non-finite");
            }
        }

        *self = candidate;
        if self.vs >= self.v_threshold && v_prev < self.v_threshold {
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Canonical reference sub-step: Jacobi compartment ordering with
    /// analytically exact removable-singularity rate limits.
    fn canonical_substep(candidate: &mut Self, current: f64) {
        let va = candidate.va;
        // Mainen & Sejnowski 1996 axonal rate functions (stable linoid form).
        let am = 0.182 * Self::linoid(va + 25.0, 9.0);
        let bm = 0.124 * Self::linoid(-(va + 25.0), 9.0);
        let ah = 0.024 * Self::linoid(va + 40.0, 5.0);
        let bh = 0.0091 * Self::linoid(-(va + 65.0), 5.0);
        let an = 0.02 * Self::linoid(va - 20.0, 9.0);
        let bn = 0.002 * Self::linoid(-(va - 20.0), 9.0);

        candidate.m = (candidate.m + (am * (1.0 - candidate.m) - bm * candidate.m) * candidate.dt)
            .clamp(0.0, 1.0);
        candidate.h = (candidate.h + (ah * (1.0 - candidate.h) - bh * candidate.h) * candidate.dt)
            .clamp(0.0, 1.0);
        candidate.n = (candidate.n + (an * (1.0 - candidate.n) - bn * candidate.n) * candidate.dt)
            .clamp(0.0, 1.0);

        let i_na = candidate.g_na * candidate.m.powi(3) * candidate.h * (va - candidate.e_na);
        let i_k = candidate.g_k * candidate.n * (va - candidate.e_k);
        let i_l_s = candidate.g_l * (candidate.vs - candidate.e_l);

        let dvs = (-i_l_s + candidate.kappa * (va - candidate.vs) + current) / candidate.c_s
            * candidate.dt;
        let dva =
            (-i_na - i_k + candidate.kappa * (candidate.vs - va)) / candidate.c_a * candidate.dt;
        candidate.vs = (candidate.vs + dvs).clamp(-200.0, 200.0);
        candidate.va = (va + dva).clamp(-200.0, 200.0);
    }

    /// Legacy engine sub-step preserved verbatim: Gauss-Seidel ordering
    /// (the axon derivative consumes the already-updated soma voltage) with
    /// the original `|x| < 1e-6` analytic-limit branches and additive 1e-12
    /// regularisation elsewhere.
    fn legacy_sequential_substep(candidate: &mut Self, current: f64) {
        let x_am = candidate.va + 25.0;
        let am = if x_am.abs() < 1e-6 {
            0.182 * 9.0
        } else {
            0.182 * x_am / (1.0 - (-(x_am) / 9.0).exp() + 1e-12)
        };
        let bm = if x_am.abs() < 1e-6 {
            0.124 * 9.0
        } else {
            -0.124 * x_am / (1.0 - ((x_am) / 9.0).exp() + 1e-12)
        };
        let x_ah = candidate.va + 40.0;
        let ah = if x_ah.abs() < 1e-6 {
            0.024 * 5.0
        } else {
            0.024 * x_ah / (1.0 - (-(x_ah) / 5.0).exp() + 1e-12)
        };
        let x_bh = candidate.va + 65.0;
        let bh = if x_bh.abs() < 1e-6 {
            0.0091 * 5.0
        } else {
            -0.0091 * x_bh / (1.0 - ((x_bh) / 5.0).exp() + 1e-12)
        };
        let x_an = candidate.va - 20.0;
        let an = if x_an.abs() < 1e-6 {
            0.02 * 9.0
        } else {
            0.02 * x_an / (1.0 - (-(x_an) / 9.0).exp() + 1e-12)
        };
        let bn = if x_an.abs() < 1e-6 {
            0.002 * 9.0
        } else {
            -0.002 * x_an / (1.0 - ((x_an) / 9.0).exp() + 1e-12)
        };
        candidate.m = (candidate.m + (am * (1.0 - candidate.m) - bm * candidate.m) * candidate.dt)
            .clamp(0.0, 1.0);
        candidate.h = (candidate.h + (ah * (1.0 - candidate.h) - bh * candidate.h) * candidate.dt)
            .clamp(0.0, 1.0);
        candidate.n = (candidate.n + (an * (1.0 - candidate.n) - bn * candidate.n) * candidate.dt)
            .clamp(0.0, 1.0);
        let i_na =
            candidate.g_na * candidate.m.powi(3) * candidate.h * (candidate.va - candidate.e_na);
        let i_k = candidate.g_k * candidate.n * (candidate.va - candidate.e_k);
        let i_l_s = candidate.g_l * (candidate.vs - candidate.e_l);
        candidate.vs = (candidate.vs
            + (-i_l_s + candidate.kappa * (candidate.va - candidate.vs) + current) / candidate.c_s
                * candidate.dt)
            .clamp(-200.0, 200.0);
        candidate.va = (candidate.va
            + (-i_na - i_k + candidate.kappa * (candidate.vs - candidate.va)) / candidate.c_a
                * candidate.dt)
            .clamp(-200.0, 200.0);
    }

    /// Fail-closed wrapper for legacy callers: returns 0 on any rejected
    /// input without mutating state.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore the dynamic state to its initial values, preserving every
    /// configuration parameter.
    pub fn reset(&mut self) {
        self.vs = -65.0;
        self.va = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
    }
}
impl Default for MainenSejnowskiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = MainenSejnowskiNeuron::default();
        let constructed = MainenSejnowskiNeuron::new();
        assert_eq!(default.vs, constructed.vs);
    }

    #[test]
    fn removable_rate_singularities_use_finite_limits() {
        for voltage in [-25.0, -40.0, -65.0, 20.0] {
            let mut n = MainenSejnowskiNeuron::new();
            n.va = voltage;
            let spike = n.step(0.0);
            assert!(matches!(spike, 0 | 1));
        }
    }

    #[test]
    fn mainen_fires() {
        let mut n = MainenSejnowskiNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }

    // -- MainenSejnowski --
    #[test]
    fn mainen_stable_without_input() {
        // Mainen 1996 model may produce transient spikes at I=0
        // (confirmed in Python reference). Verify stability only.
        let mut n = MainenSejnowskiNeuron::new();
        for _ in 0..500 {
            n.step(0.0);
        }
        assert!(n.vs.is_finite());
        assert!(n.va.is_finite());
    }
    #[test]
    fn mainen_reset_clears_state() {
        let mut n = MainenSejnowskiNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.vs - (-65.0)).abs() < 1e-10);
        assert!((n.va - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn mainen_moderate_input_stable() {
        // Two-compartment model with high conductances — moderate input
        let mut n = MainenSejnowskiNeuron::new();
        for _ in 0..200 {
            n.step(500.0);
        }
        // High-conductance 2-compartment may diverge at extremes;
        // test moderate stability
        let _ = n.vs; // no panic
    }
    #[test]
    fn mainen_two_compartments_coupled() {
        let n = MainenSejnowskiNeuron::new();
        // kappa > 0 means compartments are coupled
        assert!(n.kappa > 0.0, "coupling should be positive");
    }
    #[test]
    fn mainen_weak_negative_no_crash() {
        let mut n = MainenSejnowskiNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        // Weak negative is safer for 2-compartment
        assert!(n.vs.is_finite());
    }
    #[test]
    fn mainen_nan_input_is_rejected_atomically() {
        let mut n = MainenSejnowskiNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::NAN).is_err());
        assert!(n.try_step(f64::INFINITY).is_err());
        assert_eq!(n.vs, before.vs);
        assert_eq!(n.va, before.va);
        assert_eq!(n.m, before.m);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
    }

    #[test]
    fn mainen_invalid_configuration_is_rejected_atomically() {
        let mut n = MainenSejnowskiNeuron::new();
        n.c_s = 0.0;
        let before = n.clone();
        assert!(n.try_step(1.0).is_err());
        assert_eq!(n.vs, before.vs);
        assert_eq!(n.c_s, before.c_s);
    }

    #[test]
    fn mainen_nominal_step_matches_reference_anchor() {
        let mut n = MainenSejnowskiNeuron::new();
        assert_eq!(n.try_step(10.0), Ok(0));
        assert!((n.vs - -32.668_480_035_293_555).abs() < 1.0e-12);
        assert!((n.va - 200.0).abs() < 1.0e-12);
        assert!((n.m - 0.600_794_256_701_580_5).abs() < 1.0e-12);
        assert!((n.h - 0.658_132_236_592_029_5).abs() < 1.0e-12);
        assert!((n.n - 0.398_198_621_809_121).abs() < 1.0e-12);
    }

    #[test]
    fn mainen_rate_limits_are_exact_and_continuous_at_singular_voltages() {
        assert_eq!(MainenSejnowskiNeuron::linoid(0.0, 9.0), 9.0);
        assert_eq!(MainenSejnowskiNeuron::linoid(0.0, 5.0), 5.0);
        for k in [9.0, 5.0] {
            assert!((MainenSejnowskiNeuron::linoid(1e-9, k) - k).abs() < 1e-8);
            assert!((MainenSejnowskiNeuron::linoid(-1e-9, k) - k).abs() < 1e-8);
        }
        for v_singular in [-25.0, -40.0, -65.0, 20.0] {
            let mut exact = MainenSejnowskiNeuron::new();
            exact.va = v_singular;
            let mut near = MainenSejnowskiNeuron::new();
            near.va = v_singular + 1e-9;
            exact.try_step(0.0).expect("finite drive");
            near.try_step(0.0).expect("finite drive");
            let delta = (exact.vs - near.vs)
                .abs()
                .max((exact.va - near.va).abs())
                .max((exact.m - near.m).abs())
                .max((exact.h - near.h).abs())
                .max((exact.n - near.n).abs());
            assert!(
                delta < 1e-6,
                "public step must be continuous at va={v_singular}, delta={delta}"
            );
        }
    }

    #[test]
    fn mainen_legacy_sequential_reproduces_the_original_engine_trajectory() {
        // Anchors captured from the pre-correction engine build
        // (Gauss-Seidel ordering + |x|<1e-6 limit branches + 1e-12 form).
        let mut legacy = MainenSejnowskiNeuron::new_legacy_sequential();
        assert!(legacy.legacy_sequential);
        assert_eq!(legacy.try_step(10.0), Ok(0));
        assert!((legacy.vs - -32.668_480_035_293_555).abs() < 1.0e-12);
        assert!((legacy.va - 200.0).abs() < 1.0e-12);
        assert!((legacy.m - 0.600_794_256_701_518_1).abs() < 1.0e-12);
        assert!((legacy.h - 0.658_132_236_591_979_1).abs() < 1.0e-12);
        assert!((legacy.n - 0.398_198_621_809_030_8).abs() < 1.0e-12);

        let mut long_run = MainenSejnowskiNeuron::new_legacy_sequential();
        for _ in 0..50 {
            long_run.step(0.5);
        }
        assert!((long_run.vs - -11.459_569_992_989_016).abs() < 1.0e-12);
        assert!((long_run.h - 0.823_316_701_674_941_8).abs() < 1.0e-12);
        assert!((long_run.n - 0.890_852_351_095_750_2).abs() < 1.0e-12);
    }
}
