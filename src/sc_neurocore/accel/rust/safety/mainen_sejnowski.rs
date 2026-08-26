// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for MainenSejnowskiNeuron

#[derive(Debug, Clone)]
/// Standalone safety mirror of the two-compartment soma+axon reduction, matching the Python
/// reference recurrence and its atomic fail-closed contract.
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
}

impl MainenSejnowskiNeuron {
    /// Construct the canonical repository default configuration.
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
        }
    }

    fn linoid(x: f64, k: f64) -> f64 {
        if x == 0.0 {
            k
        } else {
            x / -(-x / k).exp_m1()
        }
    }

    /// Advance one step; `Err` preserves the pre-step state exactly for a
    /// non-finite drive, an out-of-bounds configuration, or a non-finite
    /// candidate.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !validate_mainen_sejnowski(self) {
            return Err("Mainen-Sejnowski state and parameters must satisfy the public bounds");
        }

        let mut candidate = self.clone();
        let v_prev = candidate.vs;
        for _ in 0..20 {
            let va = candidate.va;
            let am = 0.182 * Self::linoid(va + 25.0, 9.0);
            let bm = 0.124 * Self::linoid(-(va + 25.0), 9.0);
            let ah = 0.024 * Self::linoid(va + 40.0, 5.0);
            let bh = 0.0091 * Self::linoid(-(va + 65.0), 5.0);
            let an = 0.02 * Self::linoid(va - 20.0, 9.0);
            let bn = 0.002 * Self::linoid(-(va - 20.0), 9.0);

            candidate.m = (candidate.m
                + (am * (1.0 - candidate.m) - bm * candidate.m) * candidate.dt)
                .clamp(0.0, 1.0);
            candidate.h = (candidate.h
                + (ah * (1.0 - candidate.h) - bh * candidate.h) * candidate.dt)
                .clamp(0.0, 1.0);
            candidate.n = (candidate.n
                + (an * (1.0 - candidate.n) - bn * candidate.n) * candidate.dt)
                .clamp(0.0, 1.0);

            let i_na = candidate.g_na * candidate.m.powi(3) * candidate.h * (va - candidate.e_na);
            let i_k = candidate.g_k * candidate.n * (va - candidate.e_k);
            let i_l_s = candidate.g_l * (candidate.vs - candidate.e_l);

            let dvs = (-i_l_s + candidate.kappa * (va - candidate.vs) + current) / candidate.c_s
                * candidate.dt;
            let dva = (-i_na - i_k + candidate.kappa * (candidate.vs - va)) / candidate.c_a
                * candidate.dt;
            candidate.vs = (candidate.vs + dvs).clamp(-200.0, 200.0);
            candidate.va = (va + dva).clamp(-200.0, 200.0);

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

    /// Restore dynamic state to the initial values, preserving parameters.
    pub fn reset(&mut self) {
        self.vs = -65.0;
        self.va = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
    }
}

/// Return whether every state and configuration field is finite and
/// inside the public descriptor bounds.
pub fn validate_mainen_sejnowski(state: &MainenSejnowskiNeuron) -> bool {
    let finite = [
        state.vs,
        state.va,
        state.m,
        state.h,
        state.n,
        state.kappa,
        state.g_na,
        state.g_k,
        state.g_l,
        state.e_na,
        state.e_k,
        state.e_l,
        state.c_s,
        state.c_a,
        state.dt,
        state.v_threshold,
    ]
    .into_iter()
    .all(f64::is_finite);
    finite
        && (-200.0..=200.0).contains(&state.vs)
        && (-200.0..=200.0).contains(&state.va)
        && [state.m, state.h, state.n]
            .into_iter()
            .all(|gate| (0.0..=1.0).contains(&gate))
        && (0.0..=100.0).contains(&state.kappa)
        && (0.0..=5000.0).contains(&state.g_na)
        && (0.0..=3000.0).contains(&state.g_k)
        && (0.0..=5.0).contains(&state.g_l)
        && (30.0..=70.0).contains(&state.e_na)
        && (-100.0..=-70.0).contains(&state.e_k)
        && (-90.0..=-50.0).contains(&state.e_l)
        && (0.5..=2.0).contains(&state.c_s)
        && (0.05..=1.0).contains(&state.c_a)
        && state.dt > 0.0
        && state.dt <= 0.1
        && (-40.0..=20.0).contains(&state.v_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_step_matches_reference_anchor() {
        let mut state = MainenSejnowskiNeuron::new();
        assert_eq!(state.step(10.0), Ok(0));
        assert!((state.vs - -32.668_480_035_293_555).abs() < 1.0e-12);
        assert!((state.va - 200.0).abs() < 1.0e-12);
        assert!((state.m - 0.600_794_256_701_580_5).abs() < 1.0e-12);
        assert!((state.h - 0.658_132_236_592_029_5).abs() < 1.0e-12);
        assert!((state.n - 0.398_198_621_809_121).abs() < 1.0e-12);
    }

    #[test]
    fn rate_limits_are_exact_at_singular_points() {
        assert_eq!(MainenSejnowskiNeuron::linoid(0.0, 9.0), 9.0);
        assert!((MainenSejnowskiNeuron::linoid(1e-9, 5.0) - 5.0).abs() < 1e-8);
    }

    #[test]
    fn invalid_drive_is_atomic() {
        let mut state = MainenSejnowskiNeuron::new();
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert!(state.step(f64::INFINITY).is_err());
        assert_eq!(state.vs, before.vs);
        assert_eq!(state.va, before.va);
    }

    #[test]
    fn invalid_configuration_is_atomic() {
        let mut state = MainenSejnowskiNeuron::new();
        state.c_s = 0.0;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.vs, before.vs);
        assert_eq!(state.c_s, before.c_s);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = MainenSejnowskiNeuron::new();
        state.kappa = 20.0;
        state.vs = -30.0;
        state.reset();
        assert_eq!(state.vs, -65.0);
        assert_eq!(state.kappa, 20.0);
    }
}
