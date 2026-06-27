// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety kernel for quantum spin pool MPS

#![allow(unused_variables, dead_code, non_snake_case, non_camel_case_types)]

/// High-performance spin-pool telemetry kernel.
///
/// Provides entanglement-map benchmark updates matching the legacy
/// accelerator workload. Publication ATP efficiency requires the Python
/// exact state/RDM path and is not inferred from this telemetry map.
#[derive(Debug, Clone)]
pub struct QuantumSpinChainMPS {
    pub sites: usize,
    pub bond_dim: usize,
    pub correlation_length: f64,
    pub update_rate: f64,
    pub entanglement_map: Vec<f64>,
    pub measurement_count: u64,
}

impl QuantumSpinChainMPS {
    /// Create a new spin chain with uniform entanglement.
    pub fn new(sites: usize, bond_dim: usize) -> Self {
        let uniform = 1.0 / sites as f64;
        Self {
            sites,
            bond_dim,
            correlation_length: 2.0,
            update_rate: 0.1,
            entanglement_map: vec![uniform; sites],
            measurement_count: 0,
        }
    }

    /// Simulate wavefunction collapse with exponential influence kernel.
    pub fn apply_measurement(&mut self, site_idx: usize, intensity: f64) {
        let alpha = self.update_rate;
        let one_minus_alpha = 1.0 - alpha;

        let mut total: f64 = 0.0;
        for i in 0..self.sites {
            let distance = (i as f64 - site_idx as f64).abs();
            let influence = (-distance / self.correlation_length).exp() * intensity;
            self.entanglement_map[i] =
                one_minus_alpha * self.entanglement_map[i] + alpha * influence;
            total += self.entanglement_map[i];
        }

        // Normalise
        if total > 0.0 {
            let inv_total = 1.0 / total;
            for val in self.entanglement_map.iter_mut() {
                *val *= inv_total;
            }
        }

        self.measurement_count += 1;
    }

    /// Publication ATP efficiency is unavailable in this telemetry kernel.
    pub fn get_local_atp_efficiency(&self, _site_idx: usize) -> Result<f64, &'static str> {
        Err("publication ATP efficiency requires the Python exact two-site singlet RDM")
    }

    /// Return bounded benchmark telemetry for non-publication workloads.
    pub fn get_local_atp_telemetry(&self, site_idx: usize) -> f64 {
        self.entanglement_map[site_idx].clamp(0.0, 1.0)
    }

    /// Apply a global phase shift to all entanglement values.
    pub fn apply_phase_shift(&mut self, phi: f64) {
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let uniform = 1.0 / self.sites as f64;

        let mut total: f64 = 0.0;
        for i in 0..self.sites {
            let val = self.entanglement_map[i];
            self.entanglement_map[i] = (val * cos_phi + uniform * sin_phi).max(0.0);
            total += self.entanglement_map[i];
        }

        if total > 0.0 {
            let inv_total = 1.0 / total;
            for val in self.entanglement_map.iter_mut() {
                *val *= inv_total;
            }
        }
    }

    /// Return mean entanglement across all sites.
    pub fn get_avg_entanglement(&self) -> f64 {
        let total: f64 = self.entanglement_map.iter().sum();
        total / self.sites as f64
    }

    /// Reset to uniform entanglement distribution.
    pub fn reset(&mut self) {
        let uniform = 1.0 / self.sites as f64;
        for val in self.entanglement_map.iter_mut() {
            *val = uniform;
        }
        self.measurement_count = 0;
    }
}

/// Benchmark: create chain, run n_steps measurements, return avg entanglement.
pub fn benchmark_spin_chain(sites: usize, n_steps: usize) -> f64 {
    let mut chain = QuantumSpinChainMPS::new(sites, 16);
    for step in 0..n_steps {
        let site = step % sites;
        chain.apply_measurement(site, 1.0);
        chain.apply_phase_shift(0.01 * step as f64);
    }
    chain.get_avg_entanglement()
}

// ─── Hybrid Fisher-Posner LIF Neuron ───

/// LIF neuron with quantum-metabolic coupling — Rust kernel.
#[derive(Debug, Clone)]
pub struct HybridFisherPosnerLIF_Rust {
    pub neuron_id: usize,
    pub Vm: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub tau_m: f64,
    pub atp_level: f64,
    pub atp_consumption: f64,
    pub total_spikes: u64,
    pub metabolic_failures: u64,
}

impl HybridFisherPosnerLIF_Rust {
    pub fn new(neuron_id: usize) -> Self {
        Self {
            neuron_id,
            Vm: -70.0,
            v_rest: -70.0,
            v_threshold: -50.0,
            v_reset: -70.0,
            tau_m: 20.0,
            atp_level: 1.0,
            atp_consumption: 0.05,
            total_spikes: 0,
            metabolic_failures: 0,
        }
    }
}

/// Step all neurons in a batch, return spike count.
///
/// Fused kernel: ATP regeneration + LIF integration + spike decision
/// + quantum measurement feedback — all in a single pass.
pub fn batch_step_population(
    neurons: &mut [HybridFisherPosnerLIF_Rust],
    pool: &mut QuantumSpinChainMPS,
    currents: &[f64],
) -> usize {
    let mut spike_count: usize = 0;

    for (i, neuron) in neurons.iter_mut().enumerate() {
        // 1. Telemetry-modulated ATP regeneration for benchmark parity only.
        let eff = pool.get_local_atp_telemetry(i);
        let r_atp = eff * 0.01;
        neuron.atp_level = (neuron.atp_level + r_atp).min(1.0);

        // 2. Metabolic pump current
        let i_pump = (eff - 0.5) * 2.0 * neuron.atp_level;

        // 3. LIF integration (forward Euler, dt=1.0)
        let dv = (-(neuron.Vm - neuron.v_rest) + currents[i] + i_pump) / neuron.tau_m;
        neuron.Vm += dv;

        // 4. Spike decision with metabolic gate
        if neuron.Vm >= neuron.v_threshold {
            if neuron.atp_level >= neuron.atp_consumption {
                neuron.Vm = neuron.v_reset;
                neuron.atp_level -= neuron.atp_consumption;
                neuron.total_spikes += 1;
                spike_count += 1;
                pool.apply_measurement(i, 1.0);
            } else {
                neuron.Vm = neuron.v_threshold - 1.0;
                neuron.metabolic_failures += 1;
            }
        }
    }

    spike_count
}

/// Benchmark: full population step with quantum feedback.
pub fn benchmark_population(n_neurons: usize, n_steps: usize) -> u64 {
    let mut pool = QuantumSpinChainMPS::new(n_neurons, 16);
    let mut neurons: Vec<HybridFisherPosnerLIF_Rust> = (0..n_neurons)
        .map(|i| HybridFisherPosnerLIF_Rust::new(i))
        .collect();
    let mut currents = vec![25.0_f64; n_neurons];

    let mut total_spikes: u64 = 0;
    for step in 0..n_steps {
        for i in 0..n_neurons {
            currents[i] = 20.0 + 10.0 * ((step * 7 + i * 3) as f64 * 0.01).sin();
        }
        total_spikes += batch_step_population(&mut neurons, &mut pool, &currents) as u64;
    }

    total_spikes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let chain = QuantumSpinChainMPS::new(8, 16);
        assert_eq!(chain.sites, 8);
        assert_eq!(chain.entanglement_map.len(), 8);
        let sum: f64 = chain.entanglement_map.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_preserves_normalisation() {
        let mut chain = QuantumSpinChainMPS::new(8, 16);
        chain.apply_measurement(3, 1.0);
        let sum: f64 = chain.entanglement_map.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atp_efficiency_fail_closed() {
        let chain = QuantumSpinChainMPS::new(8, 16);
        assert!(chain.get_local_atp_efficiency(0).is_err());
    }

    #[test]
    fn test_telemetry_range() {
        let chain = QuantumSpinChainMPS::new(8, 16);
        for i in 0..8 {
            let eff = chain.get_local_atp_telemetry(i);
            assert!(
                (0.0..=1.0).contains(&eff),
                "Telemetry {} out of range at site {}",
                eff,
                i
            );
        }
    }

    #[test]
    fn test_non_locality() {
        let mut chain = QuantumSpinChainMPS::new(8, 16);
        for _ in 0..50 {
            chain.apply_measurement(0, 1.0);
        }
        let eff_near = chain.get_local_atp_telemetry(1);
        let eff_far = chain.get_local_atp_telemetry(7);
        assert!(
            eff_near > eff_far,
            "Non-locality failed: near={} far={}",
            eff_near,
            eff_far
        );
    }

    #[test]
    fn test_reset() {
        let mut chain = QuantumSpinChainMPS::new(4, 16);
        chain.apply_measurement(0, 1.0);
        chain.reset();
        assert_eq!(chain.measurement_count, 0);
        let expected = 1.0 / 4.0;
        for val in &chain.entanglement_map {
            assert!((val - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_benchmark() {
        let result = benchmark_spin_chain(8, 100);
        assert!(result >= 0.0);
    }

    #[test]
    fn test_neuron_init() {
        let n = HybridFisherPosnerLIF_Rust::new(5);
        assert_eq!(n.neuron_id, 5);
        assert_eq!(n.Vm, -70.0);
        assert_eq!(n.atp_level, 1.0);
    }

    #[test]
    fn test_batch_step() {
        let mut pool = QuantumSpinChainMPS::new(8, 16);
        let mut neurons: Vec<HybridFisherPosnerLIF_Rust> =
            (0..8).map(|i| HybridFisherPosnerLIF_Rust::new(i)).collect();
        let currents = vec![50.0; 8];
        let mut total_spikes = 0;
        for _ in 0..100 {
            total_spikes += batch_step_population(&mut neurons, &mut pool, &currents);
        }
        assert!(total_spikes > 0, "No spikes produced in 100 steps");
    }

    #[test]
    fn test_metabolic_failure() {
        let mut pool = QuantumSpinChainMPS::new(4, 16);
        let mut neurons = vec![HybridFisherPosnerLIF_Rust::new(0)];
        neurons[0].atp_level = 0.01;
        neurons[0].Vm = -45.0; // above threshold
        let currents = vec![0.0];
        batch_step_population(&mut neurons, &mut pool, &currents);
        assert!(neurons[0].metabolic_failures > 0);
    }

    #[test]
    fn test_population_benchmark() {
        let total = benchmark_population(8, 100);
        assert!(total > 0);
    }
}
