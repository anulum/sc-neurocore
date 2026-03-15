// SPDX-License-Identifier: AGPL-3.0-or-later
//! Discrete map neuron models.

/// Chialvo 1995 — 2D discrete map neuron.
#[derive(Clone, Debug)]
pub struct ChialvoMapNeuron {
    pub x: f64, pub y: f64,
    pub a: f64, pub b: f64, pub c: f64, pub k: f64,
    pub x_threshold: f64,
}

impl ChialvoMapNeuron {
    pub fn new() -> Self {
        Self { x: 0.0, y: 0.0, a: 0.89, b: 0.6, c: 0.28, k: 0.04, x_threshold: 1.0 }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let x_new = self.x * self.x * (self.y - self.x).exp() + self.k + current;
        let y_new = self.a * self.y - self.b * self.x + self.c;
        self.x = x_new; self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold { 1 } else { 0 }
    }
    pub fn reset(&mut self) { self.x = 0.0; self.y = 0.0; }
}
impl Default for ChialvoMapNeuron { fn default() -> Self { Self::new() } }

/// Rulkov 2001 — piecewise nonlinear map for fast/slow bursting.
#[derive(Clone, Debug)]
pub struct RulkovMapNeuron {
    pub x: f64, pub y: f64,
    pub alpha: f64, pub sigma: f64, pub mu: f64,
    pub x_threshold: f64,
}

impl RulkovMapNeuron {
    pub fn new() -> Self {
        Self { x: -1.0, y: -3.0, alpha: 4.0, sigma: -1.6, mu: 0.001, x_threshold: 0.0 }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let f = if self.x <= 0.0 {
            self.alpha / (1.0 - self.x) + self.y
        } else if self.x < self.alpha + self.y {
            self.alpha + self.y
        } else {
            -1.0
        };
        let x_new = f + current;
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        self.x = x_new; self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold { 1 } else { 0 }
    }
    pub fn reset(&mut self) { self.x = -1.0; self.y = -3.0; }
}
impl Default for RulkovMapNeuron { fn default() -> Self { Self::new() } }

/// Ibarz-Tanaka map — piecewise-linear spiking map.
#[derive(Clone, Debug)]
pub struct IbarzTanakaMapNeuron {
    pub x: f64, pub y: f64,
    pub alpha: f64, pub beta: f64, pub mu: f64, pub sigma: f64,
    pub x_threshold: f64, pub x_reset: f64,
}

impl IbarzTanakaMapNeuron {
    pub fn new() -> Self {
        Self { x: -1.0, y: -2.5, alpha: 3.65, beta: 0.25, mu: 0.0005, sigma: -1.6, x_threshold: 3.0, x_reset: -1.0 }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let f = if self.x <= 0.0 {
            self.alpha / (1.0 - self.x)
        } else {
            self.alpha + self.beta * self.x
        };
        let x_new = f + self.y + current;
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        self.x = x_new; self.y = y_new;
        if self.x >= self.x_threshold {
            self.x = self.x_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) { self.x = -1.0; self.y = -2.5; }
}
impl Default for IbarzTanakaMapNeuron { fn default() -> Self { Self::new() } }

/// Medvedev map — piecewise monotone 1D neuron map.
#[derive(Clone, Debug)]
pub struct MedvedevMapNeuron {
    pub x: f64, pub alpha: f64, pub beta: f64, pub x_threshold: f64,
}

impl MedvedevMapNeuron {
    pub fn new() -> Self {
        Self { x: 0.0, alpha: 3.5, beta: 0.5, x_threshold: 0.9 }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let f = self.alpha * self.x * (1.0 - self.x) + self.beta * current;
        self.x = f - f.floor();
        if self.x >= self.x_threshold && x_prev < self.x_threshold { 1 } else { 0 }
    }
    pub fn reset(&mut self) { self.x = 0.0; }
}
impl Default for MedvedevMapNeuron { fn default() -> Self { Self::new() } }

/// Cazelles logistic map neuron — coupled 2D logistic with slow variable.
#[derive(Clone, Debug)]
pub struct CazellesMapNeuron {
    pub x: f64, pub y: f64,
    pub a: f64, pub epsilon: f64, pub sigma: f64,
    pub x_threshold: f64,
}

impl CazellesMapNeuron {
    pub fn new() -> Self {
        Self { x: 0.1, y: 0.0, a: 3.8, epsilon: 0.01, sigma: 0.5, x_threshold: 0.9 }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let f = self.a * self.x * (1.0 - self.x);
        let x_new = (f - self.y + current).clamp(-2.0, 2.0);
        let y_new = self.y + self.epsilon * (self.x - self.sigma);
        self.x = x_new; self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold { 1 } else { 0 }
    }
    pub fn reset(&mut self) { self.x = 0.1; self.y = 0.0; }
}
impl Default for CazellesMapNeuron { fn default() -> Self { Self::new() } }

/// Courbage-Nekorkin piecewise-linear Lorenz-type map.
#[derive(Clone, Debug)]
pub struct CourageNekorkinMapNeuron {
    pub x: f64, pub y: f64,
    pub alpha: f64, pub beta: f64, pub j: f64,
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    pub fn new() -> Self {
        Self { x: 0.0, y: 0.0, alpha: 3.0, beta: 0.001, j: 0.1, x_threshold: 1.0 }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let f = if self.x.abs() < 1.0 { self.alpha * self.x } else { self.alpha * self.x.signum() };
        let x_new = f - self.y + self.j + current;
        let y_new = self.y + self.beta * self.x;
        self.x = x_new; self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold { 1 } else { 0 }
    }
    pub fn reset(&mut self) { self.x = 0.0; self.y = 0.0; }
}
impl Default for CourageNekorkinMapNeuron { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn chialvo_fires() { let mut n = ChialvoMapNeuron::new(); let t: i32 = (0..1000).map(|_| n.step(1.0)).sum(); assert!(t > 0); }
    #[test] fn rulkov_fires() { let mut n = RulkovMapNeuron::new(); let t: i32 = (0..2000).map(|_| n.step(0.5)).sum(); assert!(t > 0); }
    #[test] fn ibarz_fires() { let mut n = IbarzTanakaMapNeuron::new(); let t: i32 = (0..2000).map(|_| n.step(2.0)).sum(); assert!(t > 0); }
    #[test] fn medvedev_fires() { let mut n = MedvedevMapNeuron { x: 0.5, ..Default::default() }; let t: i32 = (0..500).map(|_| n.step(0.1)).sum(); assert!(t > 0); }
    #[test] fn cazelles_fires() { let mut n = CazellesMapNeuron::new(); let t: i32 = (0..200).map(|_| n.step(0.0)).sum(); assert!(t > 0); }
    #[test] fn cournekorkin_fires() { let mut n = CourageNekorkinMapNeuron::new(); let t: i32 = (0..200).map(|_| n.step(0.5)).sum(); assert!(t > 0); }
}
