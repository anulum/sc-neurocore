// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Courbage-Nekorkin-Vdovin map neuron

//! Courbage-Nekorkin-Vdovin discrete map neuron.

/// Courbage-Nekorkin-Vdovin (2007) discontinuous two-dimensional spiking map.
///
/// Canonical map (Chaos 17:043109; arXiv:0712.2097, eqs. 3-5):
///
/// ```text
/// x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
/// y(n+1) = y(n) + eps*(x(n) - J)
/// F(x)   = -m0*x        for x <= Jmin
///          m1*(x - a)   for Jmin < x < Jmax
///          -m0*(x - 1)  for x >= Jmax
/// H(z)   = 1 for z >= 0, else 0
/// Jmin   = a*m1/(m0 + m1),  Jmax = (m0 + a*m1)/(m0 + m1)
/// ```
///
/// Defaults place the model in the published chaotic spiking-bursting regime.
/// The arithmetic is exact (no transcendental functions), so `simulate` is
/// bit-identical to the Python NumPy reference and the Julia/Go backends.
#[derive(Clone, Debug)]
pub struct CourageNekorkinMapNeuron {
    pub x: f64,
    pub y: f64,
    pub m0: f64,
    pub m1: f64,
    pub a: f64,
    pub d: f64,
    pub j: f64,
    pub beta: f64,
    pub eps: f64,
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            m0: 0.0864,
            m1: 0.65,
            a: 0.2,
            d: 0.235,
            j: 0.2,
            beta: 0.085,
            eps: 0.02,
            x_threshold: 0.235,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let am1 = self.a * self.m1;
        let den = self.m0 + self.m1;
        let jmin = am1 / den;
        let jmax = (self.m0 + am1) / den;
        let fx = if self.x <= jmin {
            -self.m0 * self.x
        } else if self.x < jmax {
            self.m1 * (self.x - self.a)
        } else {
            -self.m0 * (self.x - 1.0)
        };
        let h = if (self.x - self.d) >= 0.0 { 1.0 } else { 0.0 };
        let x_new = self.x + fx - self.y - self.beta * h + current;
        let y_new = self.y + self.eps * (self.x - self.j);
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// upward-crossing spike count. Reuses `step` so the trace is bit-identical
    /// to the per-step path and to the Python reference. The final state is left
    /// in `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}
impl Default for CourageNekorkinMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cournekorkin_fires() {
        let mut n = CourageNekorkinMapNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    #[test]
    fn cn_default_sustained_bounded_spiking() {
        // The default parameters are the published chaotic spiking-bursting
        // regime: sustained firing with a bounded (clip-free) trajectory.
        let mut n = CourageNekorkinMapNeuron::new();
        let mut spikes = 0i64;
        let mut max_abs = 0.0f64;
        for _ in 0..20_000 {
            spikes += n.step(0.0) as i64;
            max_abs = max_abs.max(n.x.abs());
        }
        assert!(spikes > 1000, "expected sustained spiking, got {spikes}");
        assert!(
            max_abs < 10.0,
            "trajectory must stay bounded, got {max_abs}"
        );
    }

    #[test]
    fn cn_breakpoints_inside_region() {
        // Default discontinuity d sits strictly inside (Jmin, Jmax) — eq. 6.
        let n = CourageNekorkinMapNeuron::new();
        let am1 = n.a * n.m1;
        let den = n.m0 + n.m1;
        let jmin = am1 / den;
        let jmax = (n.m0 + am1) / den;
        assert!(jmin < n.d && n.d < jmax);
        assert!(n.j > 0.0 && n.j < n.d);
        assert!(n.m0 < 1.0);
    }

    #[test]
    fn cn_f_piecewise_branches() {
        // F lower/middle/upper branches (eq. 4).
        let n = CourageNekorkinMapNeuron::new();
        let am1 = n.a * n.m1;
        let den = n.m0 + n.m1;
        let jmin = am1 / den;
        let jmax = (n.m0 + am1) / den;
        // Replicate the in-step branch selection on a probe value per region.
        let f = |x: f64| {
            if x <= jmin {
                -n.m0 * x
            } else if x < jmax {
                n.m1 * (x - n.a)
            } else {
                -n.m0 * (x - 1.0)
            }
        };
        assert_eq!(f(jmin - 0.05), -n.m0 * (jmin - 0.05));
        let mid = 0.5 * (jmin + jmax);
        assert_eq!(f(mid), n.m1 * (mid - n.a));
        assert_eq!(f(jmax + 0.05), -n.m0 * (jmax + 0.05 - 1.0));
    }

    #[test]
    fn cn_heaviside_subtracts_beta_above_d() {
        // At x >= d the Heaviside term removes exactly beta from the x update.
        let mut with_beta = CourageNekorkinMapNeuron::new();
        with_beta.x = 0.30; // >= d
        let mut no_beta = with_beta.clone();
        no_beta.beta = 0.0;
        with_beta.step(0.0);
        no_beta.step(0.0);
        assert!((with_beta.x - no_beta.x - (-0.085)).abs() < 1e-15);
    }

    #[test]
    fn cn_simulate_matches_repeated_step() {
        let (trace, spikes) = CourageNekorkinMapNeuron::new().simulate(500, 0.0);
        let mut stepper = CourageNekorkinMapNeuron::new();
        let mut manual = Vec::with_capacity(500);
        let mut sp = 0i64;
        for _ in 0..500 {
            sp += stepper.step(0.0) as i64;
            manual.push(stepper.x);
        }
        assert_eq!(trace, manual);
        assert_eq!(spikes, sp);
        assert_eq!(
            stepper.x,
            CourageNekorkinMapNeuron::new().simulate(500, 0.0).0[499]
        );
    }

    #[test]
    fn cn_reset_clears_state() {
        let mut n = CourageNekorkinMapNeuron::new();
        for _ in 0..100 {
            n.step(0.0);
        }
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
    }

    #[test]
    fn cn_deterministic() {
        let (a, sa) = CourageNekorkinMapNeuron::new().simulate(1000, 0.05);
        let (b, sb) = CourageNekorkinMapNeuron::new().simulate(1000, 0.05);
        assert_eq!(a, b);
        assert_eq!(sa, sb);
    }

    #[test]
    fn cn_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = CourageNekorkinMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(0.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }
}
