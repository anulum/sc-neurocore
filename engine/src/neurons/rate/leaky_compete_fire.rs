// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Leaky compete-and-fire neuron model

/// Leaky Compete-and-Fire — winner-take-all with lateral inhibition. Oster et al. 2009.
#[derive(Clone, Debug)]
pub struct LeakyCompeteFireNeuron {
    pub v: Vec<f64>,
    pub n_units: usize,
    pub tau: f64,
    pub v_threshold: f64,
    pub w_inh: f64,
    pub dt: f64,
}

impl LeakyCompeteFireNeuron {
    pub fn new(n_units: usize) -> Self {
        Self {
            v: vec![0.0; n_units],
            n_units,
            tau: 10.0,
            v_threshold: 1.0,
            w_inh: 0.5,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, currents: &[f64]) -> Vec<i32> {
        let n = self.n_units;
        for i in 0..n {
            let c = if i < currents.len() { currents[i] } else { 0.0 };
            self.v[i] += (-self.v[i] + c) / self.tau * self.dt;
        }
        let mut spikes = vec![0i32; n];
        for i in 0..n {
            if self.v[i] >= self.v_threshold {
                spikes[i] = 1;
                self.v[i] = 0.0;
                for j in 0..n {
                    if j != i {
                        self.v[j] = (self.v[j] - self.w_inh).max(0.0);
                    }
                }
            }
        }
        spikes
    }

    pub fn reset(&mut self) {
        self.v.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcf_fires_with_strong_input() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        let inp = vec![5.0, 0.0, 0.0];
        let mut any_spike = false;
        for _ in 0..200 {
            let spikes = n.step(&inp);
            if spikes.contains(&1) {
                any_spike = true;
            }
        }
        assert!(any_spike, "LeakyCompeteFire should fire with strong input");
    }

    #[test]
    fn lcf_silent_without_input() {
        let mut n = LeakyCompeteFireNeuron::new(4);
        let inp = vec![0.0; 4];
        for _ in 0..200 {
            let spikes = n.step(&inp);
            assert!(
                spikes.iter().all(|&s| s == 0),
                "should be silent at zero input"
            );
        }
    }

    #[test]
    fn lcf_winner_take_all() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        // Unit 0 receives strong input, others receive moderate
        let inp = vec![5.0, 2.0, 2.0];
        let mut spike_counts = [0i32; 3];
        for _ in 0..1000 {
            let spikes = n.step(&inp);
            for (i, &s) in spikes.iter().enumerate() {
                spike_counts[i] += s;
            }
        }
        // Winner (unit 0) should spike more than losers due to lateral inhibition
        assert!(
            spike_counts[0] > spike_counts[1],
            "unit 0 ({}) should spike more than unit 1 ({}) — winner-take-all",
            spike_counts[0],
            spike_counts[1]
        );
    }

    #[test]
    fn lcf_lateral_inhibition_suppresses() {
        let mut n = LeakyCompeteFireNeuron::new(2);
        n.w_inh = 2.0; // Strong inhibition
        let inp = vec![3.0, 3.0];
        let mut spike_counts = [0i32; 2];
        for _ in 0..500 {
            let spikes = n.step(&inp);
            for (i, &s) in spikes.iter().enumerate() {
                spike_counts[i] += s;
            }
        }
        // With equal input + strong inhibition, total spikes should be less
        // than with no inhibition (competitive suppression)
        let mut n_no_inh = LeakyCompeteFireNeuron::new(2);
        n_no_inh.w_inh = 0.0;
        let mut total_no_inh = 0i32;
        for _ in 0..500 {
            let spikes = n_no_inh.step(&inp);
            total_no_inh += spikes.iter().sum::<i32>();
        }
        let total_inh: i32 = spike_counts.iter().sum();
        assert!(
            total_inh <= total_no_inh,
            "inhibition ({}) should reduce total spikes vs no inhibition ({})",
            total_inh,
            total_no_inh
        );
    }

    #[test]
    fn lcf_reset_clears_state() {
        let mut n = LeakyCompeteFireNeuron::new(4);
        let inp = vec![3.0; 4];
        for _ in 0..100 {
            n.step(&inp);
        }
        n.reset();
        assert!(
            n.v.iter().all(|&x| x == 0.0),
            "reset must zero all voltages"
        );
    }

    #[test]
    fn lcf_voltages_bounded() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        let inp = vec![1e6, 1e6, 1e6];
        for _ in 0..1000 {
            n.step(&inp);
        }
        assert!(
            n.v.iter().all(|x| x.is_finite()),
            "voltages must stay finite under extreme input"
        );
    }

    #[test]
    fn lcf_negative_input_no_crash() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        let inp = vec![-10.0, -5.0, -1.0];
        for _ in 0..500 {
            n.step(&inp);
        }
        assert!(
            n.v.iter().all(|x| x.is_finite()),
            "must handle negative input"
        );
    }

    #[test]
    fn lcf_output_length_matches_units() {
        let n_units = 7;
        let mut n = LeakyCompeteFireNeuron::new(n_units);
        let inp = vec![1.0; n_units];
        let spikes = n.step(&inp);
        assert_eq!(spikes.len(), n_units, "output length must match n_units");
    }
}
