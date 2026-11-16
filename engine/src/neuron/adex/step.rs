// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AdEx checked baseline-Euler step

use super::AdExNeuron;

impl AdExNeuron {
    /// Advance one checked baseline-Euler step without partial mutation.
    ///
    /// `dv = ((-(v - v_rest) + delta_t * exp((v - v_rh) / delta_t)) / tau + (-w + current) / c_m) * dt`,
    /// with the exponent clamped to `[-20, 20]`, and `dw = (a * (v - v_rest) - w) / tau_w * dt`;
    /// both use the old state. Reaching `v_threshold` sets `v` to `v_reset` and adds `b` to
    /// the updated `w`. Returns 1 on a spike, else 0.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.v.is_finite()
            || !self.w.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.tau_w.is_finite()
            || !self.a.is_finite()
            || !self.b.is_finite()
            || !self.c_m.is_finite()
            || !self.dt.is_finite()
            || !current.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.tau_w <= 0.0
            || self.c_m <= 0.0
            || self.dt <= 0.0
        {
            return Err("invalid AdEx state, parameters, timestep, or input");
        }

        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = ((-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current) / self.c_m)
            * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        let next_v = self.v + dv;
        let next_w = self.w + dw;
        if !exp_term.is_finite()
            || !dv.is_finite()
            || !dw.is_finite()
            || !next_v.is_finite()
            || !next_w.is_finite()
        {
            return Err("non-finite AdEx integrator candidate");
        }

        if next_v >= self.v_threshold {
            let spike_w = next_w + self.b;
            if !spike_w.is_finite() {
                return Err("non-finite AdEx spike-adaptation candidate");
            }
            self.v = self.v_reset;
            self.w = spike_w;
            Ok(1)
        } else {
            self.v = next_v;
            self.w = next_w;
            Ok(0)
        }
    }
}
