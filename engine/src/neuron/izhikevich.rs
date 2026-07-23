// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Floating-point Izhikevich neuron

/// Izhikevich neuron (floating-point).
///
/// Standard model from IEEE TNN 14(6), 2003:
///   v' = 0.04*v² + 5*v + 140 - u + I
///   u' = a*(b*v - u)
///   if v >= 30: v ← c, u ← u + d
#[derive(Clone, Debug)]
pub struct Izhikevich {
    pub v: f64,
    pub u: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub dt: f64,
}

impl Izhikevich {
    /// Regular spiking defaults: a=0.02, b=0.2, c=-65, d=8, dt=1.0.
    pub fn new(a: f64, b: f64, c: f64, d: f64, dt: f64) -> Self {
        Self {
            v: c,
            u: b * c,
            a,
            b,
            c,
            d,
            dt,
        }
    }

    /// Regular spiking preset.
    pub fn regular_spiking() -> Self {
        Self::new(0.02, 0.2, -65.0, 8.0, 1.0)
    }

    /// Advance one step. Returns 1 on spike, 0 otherwise.
    pub fn step(&mut self, current: f64) -> i32 {
        let half = self.dt * 0.5;
        for _ in 0..2 {
            let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current) * half;
            let du = (self.a * (self.b * self.v - self.u)) * half;
            self.v += dv;
            self.u += du;
        }

        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            1
        } else {
            0
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }
}

#[cfg(test)]
mod tests {
    use super::Izhikevich;

    #[test]
    fn regular_spiking_preset_fires_with_current() {
        let mut neuron = Izhikevich::regular_spiking();
        let spikes: i32 = (0..100).map(|_| neuron.step(10.0)).sum();
        assert!(spikes > 0, "RS neuron must fire with I=10");
    }

    #[test]
    fn regular_spiking_preset_is_silent_without_input() {
        let mut neuron = Izhikevich::regular_spiking();
        let spikes: i32 = (0..100).map(|_| neuron.step(0.0)).sum();
        assert_eq!(spikes, 0, "no spikes without input");
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut neuron = Izhikevich::regular_spiking();
        for _ in 0..50 {
            neuron.step(10.0);
        }
        neuron.reset();
        assert_eq!(neuron.v, neuron.c);
        assert!((neuron.u - neuron.b * neuron.c).abs() < 1e-12);
    }

    #[test]
    fn chattering_preset_fires_more_than_regular_spiking() {
        let mut chattering = Izhikevich::new(0.02, 0.2, -50.0, 2.0, 1.0);
        let mut regular_spiking = Izhikevich::regular_spiking();
        let mut chattering_spikes = 0;
        let mut regular_spikes = 0;
        for _ in 0..200 {
            chattering_spikes += chattering.step(10.0);
            regular_spikes += regular_spiking.step(10.0);
        }
        assert!(
            chattering_spikes > regular_spikes,
            "chattering ({chattering_spikes}) should fire more than RS ({regular_spikes})"
        );
    }
}
