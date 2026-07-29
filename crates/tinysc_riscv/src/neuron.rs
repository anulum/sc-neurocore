// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Spiking Neuron Models (no_std)

//! LIF and Izhikevich spiking neurons for SC-domain inference.
//!
//! Operates on `u32`-packed bitstream slices. Membrane potential is
//! tracked as a `u16` popcount accumulator — no floating-point needed.

use crate::bitstream;

/// Leaky Integrate-and-Fire neuron (SC domain).
///
/// Membrane potential = running popcount of input bitstream.
/// Leak = right-shift per tick (exponential decay).
/// Fires when potential exceeds `threshold`.
pub struct LifNeuron {
    pub potential: u32,
    pub threshold: u32,
    pub leak_shift: u8,
    pub refractory_ticks: u8,
    remaining_refractory: u8,
}

impl LifNeuron {
    /// Create a new LIF neuron.
    ///
    /// - `threshold`: popcount level that triggers a spike.
    /// - `leak_shift`: right-shift applied per tick (1 = halve, 2 = quarter, etc.).
    /// - `refractory`: number of ticks after spike during which neuron cannot fire.
    pub const fn new(threshold: u32, leak_shift: u8, refractory: u8) -> Self {
        Self {
            potential: 0,
            threshold,
            leak_shift,
            refractory_ticks: refractory,
            remaining_refractory: 0,
        }
    }

    /// Feed one packed bitstream tick into the neuron. Returns `true` if spike.
    #[inline]
    pub fn tick(&mut self, input_words: &[u32]) -> bool {
        if self.remaining_refractory > 0 {
            self.remaining_refractory -= 1;
            return false;
        }

        let excitation = bitstream::popcount_slice(input_words);
        self.potential = self.potential.wrapping_add(excitation);
        self.potential >>= self.leak_shift;

        if self.potential >= self.threshold {
            self.potential = 0;
            self.remaining_refractory = self.refractory_ticks;
            true
        } else {
            false
        }
    }

    /// Reset neuron state.
    #[inline]
    pub fn reset(&mut self) {
        self.potential = 0;
        self.remaining_refractory = 0;
    }
}

/// Izhikevich neuron (fixed-point SC domain).
///
/// Uses Q8.8 fixed-point arithmetic (i16) for membrane voltage and
/// recovery variable — no FPU required.
pub struct IzhikevichNeuron {
    pub v: i16,         // Q8.8 membrane voltage
    pub u: i16,         // Q8.8 recovery variable
    pub a: i16,         // Q8.8 time scale of u
    pub b: i16,         // Q8.8 sensitivity of u to v
    pub c: i16,         // Q8.8 after-spike reset of v
    pub d: i16,         // Q8.8 after-spike reset of u
    pub threshold: i16, // Q8.8 spike threshold
}

impl IzhikevichNeuron {
    /// Regular spiking (RS) preset.
    pub const fn regular_spiking() -> Self {
        Self {
            v: -65 * 256, // -65.0 in Q8.8
            u: -13 * 256, // b * v
            a: 5,         // 0.02 in Q8.8
            b: 51,        // 0.2 in Q8.8
            c: -65 * 256,
            d: 2 * 256, // 8.0 in Q8.8 (actually d=8 → 2048)
            threshold: 30 * 256,
        }
    }

    /// Fast spiking (FS) preset.
    pub const fn fast_spiking() -> Self {
        Self {
            v: -65 * 256,
            u: -13 * 256,
            a: 26, // 0.1 in Q8.8
            b: 51,
            c: -65 * 256,
            d: 512, // 2.0 in Q8.8
            threshold: 30 * 256,
        }
    }

    /// Step the neuron with SC-domain input (popcount as current).
    ///
    /// `input_popcount`: popcount of the incoming bitstream tick.
    /// Returns `true` on spike.
    #[inline]
    pub fn tick(&mut self, input_popcount: u32) -> bool {
        let i_q88 = (input_popcount as i16).saturating_mul(256);

        // dv = 0.04*v² + 5*v + 140 - u + I (simplified Q8.8)
        let v_sq = ((self.v as i32) * (self.v as i32)) >> 8;
        let dv =
            ((v_sq >> 3) + (self.v as i32) * 5 + 140 * 256 - self.u as i32 + i_q88 as i32) >> 4;
        self.v = self.v.saturating_add(dv as i16);

        // du = a * (b*v - u)
        let bv = ((self.b as i32) * (self.v as i32)) >> 8;
        let du = ((self.a as i32) * (bv - self.u as i32)) >> 8;
        self.u = self.u.saturating_add(du as i16);

        if self.v >= self.threshold {
            self.v = self.c;
            self.u = self.u.saturating_add(self.d);
            true
        } else {
            false
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.v = -65 * 256;
        self.u = ((self.b as i32 * self.v as i32) >> 8) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_no_spike_below_threshold() {
        let mut n = LifNeuron::new(100, 0, 0);
        let input = [0x0000_000Fu32]; // 4 ones
        assert!(!n.tick(&input));
    }

    #[test]
    fn test_lif_spike_above_threshold() {
        let mut n = LifNeuron::new(10, 0, 0);
        let input = [u32::MAX]; // 32 ones
        assert!(n.tick(&input));
        assert_eq!(n.potential, 0, "potential should reset after spike");
    }

    #[test]
    fn test_lif_refractory() {
        let mut n = LifNeuron::new(10, 0, 2);
        let input = [u32::MAX];
        assert!(n.tick(&input));
        assert!(!n.tick(&input), "should be in refractory");
        assert!(!n.tick(&input), "still in refractory");
        assert!(n.tick(&input), "refractory over");
    }

    #[test]
    fn test_lif_leak() {
        let mut n = LifNeuron::new(1000, 1, 0);
        let input = [0x0000_00FFu32]; // 8 ones
        n.tick(&input);
        assert!(
            n.potential < 8,
            "leak should reduce potential: {}",
            n.potential
        );
    }

    #[test]
    fn test_lif_reset() {
        let mut n = LifNeuron::new(10, 0, 0);
        n.potential = 999;
        n.remaining_refractory = 5;
        n.reset();
        assert_eq!(n.potential, 0);
        assert_eq!(n.remaining_refractory, 0);
    }

    #[test]
    fn test_izh_regular_spiking_creation() {
        let n = IzhikevichNeuron::regular_spiking();
        assert_eq!(n.v, -65 * 256);
        assert!(n.threshold > 0);
    }

    #[test]
    fn test_izh_fast_spiking_creation() {
        let n = IzhikevichNeuron::fast_spiking();
        assert!(n.a > IzhikevichNeuron::regular_spiking().a);
    }

    #[test]
    fn test_izh_spike_on_high_input() {
        let mut n = IzhikevichNeuron::regular_spiking();
        let mut spiked = false;
        for _ in 0..500 {
            if n.tick(200) {
                spiked = true;
                break;
            }
        }
        assert!(spiked, "should spike under sustained high input");
    }

    #[test]
    fn test_izh_reset() {
        let mut n = IzhikevichNeuron::regular_spiking();
        n.v = 1000;
        n.reset();
        assert_eq!(n.v, -65 * 256);
    }
}
