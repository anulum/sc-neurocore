// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Retained normalized energy-gated exact-flow SC model.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;

#[derive(Clone, Debug)]
pub struct SCNormalizedEnergyLIFNeuron {
    pub v: f64,
    pub epsilon: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_e: f64,
    pub alpha: f64,
    pub epsilon_0: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl SCNormalizedEnergyLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            epsilon: 1.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_e: 500.0,
            alpha: 0.1,
            epsilon_0: 1.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }
    pub fn valid(&self) -> bool {
        [
            self.v,
            self.epsilon,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau_m,
            self.tau_e,
            self.alpha,
            self.epsilon_0,
            self.resistance,
            self.dt,
        ]
        .into_iter()
        .all(f64::is_finite)
            && (V_MIN..=V_MAX).contains(&self.v)
            && (V_MIN..=V_MAX).contains(&self.v_reset)
            && (0.0..=self.epsilon_0).contains(&self.epsilon)
            && self.epsilon_0 >= 0.0
            && self.alpha >= 0.0
            && self.tau_m > 0.0
            && self.tau_e > 0.0
            && self.resistance > 0.0
            && self.dt > 0.0
            && self.dt <= self.tau_m
            && self.dt <= self.tau_e
            && self.v_threshold > self.v_rest
            && self.v_threshold > self.v_reset
    }
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.valid() || !current.is_finite() {
            return Err("invalid SC normalized EnergyLIF state, configuration, or current");
        }
        let md = (-self.dt / self.tau_m).exp();
        let ed = (-self.dt / self.tau_e).exp();
        let de = self.epsilon - self.epsilon_0;
        let epsilon = self.epsilon_0 + de * ed;
        let steady = self.epsilon_0 * self.tau_m * (1.0 - md);
        let rate = 1.0 / self.tau_m - 1.0 / self.tau_e;
        let transient = if rate.abs() < 1.0e-12 {
            de * md * self.dt
        } else {
            de * md * (rate * self.dt).exp_m1() / rate
        };
        let v = self.v_rest
            + (self.v - self.v_rest) * md
            + (self.resistance * current / self.tau_m) * (steady + transient);
        if !(v.is_finite()
            && (V_MIN..=V_MAX).contains(&v)
            && epsilon.is_finite()
            && (0.0..=self.epsilon_0).contains(&epsilon))
        {
            return Err("SC normalized EnergyLIF candidate outside safety envelope");
        }
        if v >= self.v_threshold && epsilon > 0.1 {
            self.v = self.v_reset;
            self.epsilon = (epsilon - self.alpha).max(0.0);
            return Ok(1);
        }
        self.v = v;
        self.epsilon = epsilon;
        Ok(0)
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(-1)
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.epsilon = self.epsilon_0;
    }
}
impl Default for SCNormalizedEnergyLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_trace_is_frozen() {
        let mut n = SCNormalizedEnergyLIFNeuron::new();
        let drive = [30.0, 0.0, 50.0, 10.0];
        let events: i32 = (0..256).map(|i| n.step(drive[i % drive.len()])).sum();
        assert_eq!(events, 3);
        assert!((n.v - (-52.508269792668216)).abs() < 1e-12);
    }
}
