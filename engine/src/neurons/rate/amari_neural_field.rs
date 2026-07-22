// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Amari neural-field model

/// Amari neural field — continuous attractor with Mexican hat kernel.
#[derive(Clone, Debug)]
pub struct AmariNeuralField {
    pub u: Vec<f64>,
    pub n: usize,
    pub tau: f64,
    pub a_exc: f64,
    pub a_width: f64,
    pub b_inh: f64,
    pub b_width: f64,
    pub dx: f64,
    pub dt: f64,
    w: Vec<f64>,
}

impl AmariNeuralField {
    pub fn new(n: usize) -> Self {
        let dx = 0.5;
        let mut w = vec![0.0; n];
        let a_exc = 1.5;
        let a_width = 1.0;
        let b_inh = 0.75;
        let b_width = 2.0;
        // Exponential Mexican hat kernel, rolled to match Python's np.roll
        for i in 0..n {
            let x = ((i as isize - n as isize / 2).unsigned_abs() as f64) * dx;
            w[i] = a_exc * (-a_width * x).exp() - b_inh * (-b_width * x).exp();
        }
        // Roll by -n/2 to match Python's np.roll(k, -n//2)
        w.rotate_left(n / 2);
        Self {
            u: vec![0.0; n],
            n,
            tau: 10.0,
            a_exc,
            a_width,
            b_inh,
            b_width,
            dx,
            dt: 0.5,
            w,
        }
    }
    pub fn step(&mut self, input: &[f64]) -> f64 {
        let n = self.n;
        // f(u) = max(0, u)
        let f_u: Vec<f64> = self.u.iter().map(|&v| v.max(0.0)).collect();
        // Circular convolution (matching Python FFT-based conv)
        let mut conv = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                let idx = (i + n - j) % n;
                s += self.w[idx] * f_u[j];
            }
            conv[i] = s * self.dx;
        }
        for i in 0..n {
            let inp = if i < input.len() { input[i] } else { 0.0 };
            self.u[i] += (-self.u[i] + conv[i] + inp) / self.tau * self.dt;
        }
        // Return mean of ReLU activations
        self.u.iter().map(|&v| v.max(0.0)).sum::<f64>() / n as f64
    }
    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amari_activates() {
        let mut n = AmariNeuralField::new(32);
        let inp = vec![0.5; 32];
        for _ in 0..100 {
            n.step(&inp);
        }
        assert!(n.u.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn amari_reset() {
        let mut n = AmariNeuralField::new(32);
        let inp = vec![0.5; 32];
        for _ in 0..100 {
            n.step(&inp);
        }
        n.reset();
        assert!(n.u.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn amari_bounded() {
        let mut n = AmariNeuralField::new(16);
        let inp = vec![1e3; 16];
        for _ in 0..1000 {
            n.step(&inp);
        }
        assert!(n.u.iter().all(|x| x.is_finite()));
    }
}
