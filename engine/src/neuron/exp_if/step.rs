// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exponential integrate-and-fire checked step

use super::{ExpIfError, ExpIfNeuron};

impl ExpIfNeuron {
    /// Advance one checked update without mutating the receiver on rejection.
    ///
    /// While refractory, `v` is held at `v_reset` and the remaining time counts down by `dt`.
    /// Otherwise the SC profile takes one RK4 step, and the source profile one Heun (RK2)
    /// step, of `tau dv/dt = -(v - v_rest) + delta_t * exp((v - v_rh) / delta_t) + current`,
    /// with `v` capped at `v_threshold` inside the right-hand side. Reaching `v_threshold`
    /// sets `v` to `v_reset` and starts the refractory period. Returns 1 on a spike, else 0.
    pub fn try_step(&mut self, current: f64) -> Result<i32, ExpIfError> {
        if !self.v.is_finite()
            || !current.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.dt.is_finite()
            || !self.refractory_period.is_finite()
            || !self.refractory_remaining.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.dt <= 0.0
            || self.refractory_period < 0.0
            || self.refractory_remaining < 0.0
            || self.refractory_remaining > self.refractory_period
            || self.v_threshold <= self.v_rh
            || self.v >= self.v_threshold
            || self.v_rest >= self.v_threshold
            || self.v_reset >= self.v_threshold
            || (self.source_profile
                && (self.v_rest != -65.0
                    || self.v_reset != -68.0
                    || self.v_threshold != -30.0
                    || self.v_rh != -59.9
                    || self.delta_t != 3.48
                    || self.tau != 10.0
                    || self.dt >= 0.02
                    || self.refractory_period != 1.7))
        {
            return Err(if !current.is_finite() {
                ExpIfError::InvalidInput
            } else {
                ExpIfError::InvalidState
            });
        }

        if self.refractory_remaining > 0.0 {
            self.refractory_remaining = (self.refractory_remaining - self.dt).max(0.0);
            self.v = self.v_reset;
            return Ok(0);
        }

        let inv_delta_t = 1.0 / self.delta_t;
        let k1 = self.rhs(self.v, current, inv_delta_t);
        let predictor = self.v + self.dt * k1;
        let k2 = self.rhs(
            if self.source_profile {
                predictor
            } else {
                self.v + 0.5 * self.dt * k1
            },
            current,
            inv_delta_t,
        );
        let (k3, k4, next_v) = if self.source_profile {
            (0.0, 0.0, self.v + 0.5 * self.dt * (k1 + k2))
        } else {
            let k3 = self.rhs(self.v + 0.5 * self.dt * k2, current, inv_delta_t);
            let k4 = self.rhs(self.v + self.dt * k3, current, inv_delta_t);
            (
                k3,
                k4,
                self.v + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            )
        };
        if !k1.is_finite()
            || !k2.is_finite()
            || !k3.is_finite()
            || !k4.is_finite()
            || !next_v.is_finite()
        {
            return Err(ExpIfError::NonFiniteUpdate);
        }

        self.inv_delta_t = inv_delta_t;
        self.dt_div_tau = self.dt / self.tau;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_remaining = self.refractory_period;
            Ok(1)
        } else {
            self.v = next_v;
            Ok(0)
        }
    }
}
