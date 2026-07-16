// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Escape-Rate Neuron

/// Escape-rate neuron — stochastic IF with exponential hazard. Gerstner 2000.
#[derive(Clone, Debug)]
pub struct EscapeRateNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub rho_0: f64,
    pub delta_u: f64,
    pub resistance: f64,
    pub dt: f64,
    pub rng_state: u16,
    pub initial_seed: u16,
}

impl EscapeRateNeuron {
    pub fn new(seed: u64) -> Self {
        let narrowed = (seed & u64::from(u16::MAX)) as u16;
        let initial_seed = if narrowed == 0 { 0xACE1 } else { narrowed };
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            rho_0: 0.001,
            delta_u: 3.0,
            resistance: 1.0,
            dt: 1.0,
            rng_state: initial_seed,
            initial_seed,
        }
    }

    pub fn valid(&self) -> bool {
        self.v.is_finite()
            && self.v_rest.is_finite()
            && self.v_reset.is_finite()
            && self.v_threshold.is_finite()
            && self.tau_m.is_finite()
            && self.tau_m > 0.0
            && self.rho_0.is_finite()
            && self.rho_0 > 0.0
            && self.delta_u.is_finite()
            && self.delta_u > 0.0
            && self.resistance.is_finite()
            && self.resistance > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.rng_state != 0
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.v.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.tau_m.is_finite()
            || self.tau_m <= 0.0
            || !self.rho_0.is_finite()
            || self.rho_0 <= 0.0
            || !self.delta_u.is_finite()
            || self.delta_u <= 0.0
            || !self.resistance.is_finite()
            || self.resistance <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || self.rng_state == 0
            || !current.is_finite()
        {
            return Err("invalid escape-rate state or input");
        }
        let v_inf = self.v_rest + self.resistance * current;
        let decay = (-self.dt / self.tau_m).exp();
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return Err("non-finite escape-rate membrane candidate");
        }
        let hazard = self.rho_0
            * ((next_v - self.v_threshold) / self.delta_u)
                .clamp(-700.0, 700.0)
                .exp()
            * self.dt;
        if !hazard.is_finite() || hazard < 0.0 {
            return Err("non-finite escape hazard");
        }
        let p_spike = -(-hazard).exp_m1();
        if !p_spike.is_finite() || !(0.0..=1.0).contains(&p_spike) {
            return Err("invalid escape probability");
        }
        let mut sample = self.rng_state;
        for _ in 0..8 {
            let feedback = (sample ^ (sample >> 2) ^ (sample >> 3) ^ (sample >> 5)) & 1;
            sample = (sample >> 1) | (feedback << 15);
        }
        let threshold = if p_spike <= 0.0 {
            0_u32
        } else if p_spike >= 1.0 {
            65_536_u32
        } else {
            (p_spike * 65_535.0).floor() as u32 + 1
        };
        self.rng_state = sample;
        if u32::from(sample) < threshold {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = next_v;
            Ok(0)
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.rng_state = self.initial_seed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_rate_stochastic() {
        let mut n = EscapeRateNeuron::new(42);
        let total: i32 = (0..1000).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
    }
    #[test]
    fn escape_rate_exact_flow_matches_closed_form() {
        let mut n = EscapeRateNeuron::new(42);
        n.v = -65.0;
        n.dt = 5.0;
        n.rho_0 = 1.0e-12;
        let current = 10.0;
        let v0 = n.v;
        let v_inf = n.v_rest + n.resistance * current;
        let euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau_m * n.dt;
        let expected = v_inf + (v0 - v_inf) * (-n.dt / n.tau_m).exp();

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected).abs() < 1e-14);
        assert!((n.v - euler).abs() > 1e-3);
    }
    #[test]
    fn escape_rate_reset_clears_state() {
        let mut n = EscapeRateNeuron::new(42);
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn escape_rate_bounded() {
        let mut n = EscapeRateNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn escape_rate_nan_no_panic() {
        let mut n = EscapeRateNeuron::new(42);
        let before = n.v;
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(n.v, before);
    }
    #[test]
    fn escape_rate_invalid_state_does_not_mutate() {
        let mut n = EscapeRateNeuron::new(42);
        n.v = -65.0;
        n.tau_m = 0.0;
        assert_eq!(n.step(1.0), 0);
        assert_eq!(n.v, -65.0);
    }
    #[test]
    fn escape_rate_seed_varies() {
        let mut n1 = EscapeRateNeuron::new(1);
        let mut n2 = EscapeRateNeuron::new(999);
        let t1: i32 = (0..1000).map(|_| n1.step(30.0)).sum();
        let t2: i32 = (0..1000).map(|_| n2.step(30.0)).sum();
        assert!(t1 > 0 && t2 > 0);
    }
}
