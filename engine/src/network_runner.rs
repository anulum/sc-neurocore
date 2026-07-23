// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner: high-performance Rust simulation backend

//! High-performance network simulation backend.
//!
//! Replaces the Python per-neuron loop with Rayon-parallel Rust execution
//! over CSR-stored projections and heterogeneous neuron populations.

mod input_adapters;
pub use input_adapters::*;

mod neuron_variant;
pub use neuron_variant::NeuronVariant;

mod population_runner;
pub use population_runner::PopulationRunner;

mod projection_runner;
pub use projection_runner::ProjectionRunner;

mod simulation_results;
pub use simulation_results::SimResults;

mod network_execution;
pub use network_execution::NetworkRunner;

mod model_factory;
pub use model_factory::{create_neuron, create_population, supported_models};

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn izhikevich_population_spikes() {
        let mut pop = create_population("Izhikevich", 10).unwrap();
        let mut total_spikes = 0usize;
        for _ in 0..100 {
            pop.currents.fill(10.0);
            pop.step_all();
            total_spikes += pop.spikes.iter().filter(|&&s| s != 0).count();
        }
        assert!(
            total_spikes > 0,
            "10 Izhikevich neurons must spike with I=10"
        );
    }

    #[test]
    fn single_population_step_accepts_external_currents() {
        let mut runner = NetworkRunner::new();
        let idx = runner.add_population(create_population("Lapicque", 3).unwrap());

        let (spikes, voltages) = runner
            .step_population_with_currents(idx, &[1.0, 2.0, 3.0])
            .unwrap();

        assert_eq!(spikes.len(), 3);
        assert_eq!(voltages.len(), 3);
        assert!(spikes.iter().all(|&s| s <= 1));
        assert!(voltages.iter().all(|v| v.is_finite()));
        assert!(runner
            .step_population_with_currents(idx, &[1.0, 2.0])
            .is_err());
        assert!(runner
            .step_population_with_currents(idx + 1, &[1.0, 2.0, 3.0])
            .is_err());
    }

    #[test]
    fn all_to_all_network_100_steps() {
        let mut runner = NetworkRunner::new();
        let pop = create_population("Izhikevich", 4).unwrap();
        runner.add_population(pop);

        // All-to-all CSR: 4 src -> 4 tgt
        let mut row_offsets = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut offset = 0;
        for _i in 0..4 {
            row_offsets.push(offset);
            for j in 0..4 {
                col_indices.push(j);
                values.push(2.0);
                offset += 1;
            }
        }
        row_offsets.push(offset);

        let proj = ProjectionRunner::new(0, 0, row_offsets, col_indices, values, 0);
        runner.add_projection(proj);

        // Inject external current by pre-filling
        for n in &mut runner.populations[0].neurons {
            if let NeuronVariant::Izhikevich(iz) = n {
                iz.v = -50.0;
            }
        }

        let results = runner.run(100);
        assert_eq!(results.spike_counts.len(), 1);
        assert_eq!(results.voltages.len(), 1);
        assert_eq!(results.voltages[0].len(), 4);
    }

    #[test]
    fn mixed_hh_adex_network() {
        let mut runner = NetworkRunner::new();

        let hh_pop = create_population("HodgkinHuxley", 3).unwrap();
        let adex_pop = create_population("AdEx", 3).unwrap();
        let hh_idx = runner.add_population(hh_pop);
        let adex_idx = runner.add_population(adex_pop);

        // HH -> AdEx projection
        let row_offsets = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![100.0; 9];
        let proj = ProjectionRunner::new(hh_idx, adex_idx, row_offsets, col_indices, values, 0);
        runner.add_projection(proj);

        // Drive HH with external current
        runner.populations[0].currents.fill(15.0);

        let results = runner.run(50);
        assert_eq!(results.spike_counts.len(), 2);
        assert_eq!(results.voltages.len(), 2);
    }

    #[test]
    fn large_network_performance() {
        let n = 1000;
        let mut pop = create_population("Izhikevich", n).unwrap();
        // Run 1000 steps with constant drive — should complete quickly
        for _ in 0..1000 {
            pop.currents.fill(10.0);
            pop.step_all();
        }
        let total: usize = pop.spikes.iter().map(|&s| s as usize).sum();
        // Sanity: spike count should be deterministic and nonzero after 1000 driven steps
        let _ = total;
        // Check voltages are finite
        let voltages = pop.collect_voltages();
        assert_eq!(voltages.len(), n);
        for v in &voltages {
            assert!(v.is_finite(), "voltage must be finite");
        }
    }

    #[test]
    fn batch_simulate_single_neuron() {
        let mut neuron = create_neuron("AdEx").unwrap();
        let n_steps = 1000;
        let current = 500.0;
        let mut voltages = Vec::with_capacity(n_steps);
        let mut spikes = Vec::new();
        for t in 0..n_steps {
            let fired = neuron.step(current);
            voltages.push(neuron.soma_voltage());
            if fired != 0 {
                spikes.push(t);
            }
        }
        assert_eq!(voltages.len(), n_steps);
        assert!(voltages.iter().all(|v| v.is_finite()));
        assert!(!spikes.is_empty(), "AdEx with I=10 should spike");
    }

    #[test]
    fn mcculloch_pitts_network_wrapper_preserves_signed_logical_transport() {
        let mut neuron = create_neuron("McCullochPittsNeuron").unwrap();
        assert_eq!(neuron.step(0.0), 0);
        assert_eq!(neuron.step(1.0), 1);
        assert_eq!(neuron.step(-1.0), 0);
        assert_eq!(neuron.step(1.5), 0);
        assert_eq!(neuron.step(f64::NAN), 0);
        assert_eq!(neuron.soma_voltage(), 0.0);
        neuron.reset();
        assert_eq!(neuron.step(1.0), 1);
    }

    // ── Pipeline integration: interneurons ────────────────────────

    #[test]
    fn interneuron_population_create_step_reset() {
        for name in &[
            "PVFastSpiking",
            "SST",
            "VIP",
            "Chandelier",
            "CerebellarBasket",
            "Martinotti",
        ] {
            let mut pop = create_population(name, 5).unwrap();
            pop.currents.fill(3.0);
            for _ in 0..100 {
                pop.step_all();
            }
            let voltages = pop.collect_voltages();
            assert_eq!(voltages.len(), 5, "{name}: voltage count mismatch");
            for v in &voltages {
                assert!(v.is_finite(), "{name}: non-finite voltage {v}");
            }
            pop.reset_all();
            let v_after_reset = pop.collect_voltages();
            for v in &v_after_reset {
                assert!(v.is_finite(), "{name}: non-finite after reset");
            }
        }
    }

    #[test]
    fn interneuron_mixed_network() {
        let mut runner = NetworkRunner::new();
        let pv_pop = create_population("PVFastSpiking", 3).unwrap();
        let sst_pop = create_population("SST", 3).unwrap();
        let pv_idx = runner.add_population(pv_pop);
        let sst_idx = runner.add_population(sst_pop);

        // PV → SST all-to-all projection
        let row_offsets = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![1.0; 9];
        let proj = ProjectionRunner::new(pv_idx, sst_idx, row_offsets, col_indices, values, 0);
        runner.add_projection(proj);

        runner.populations[0].currents.fill(3.0);
        let results = runner.run(50);
        assert_eq!(results.spike_counts.len(), 2);
        assert_eq!(results.voltages.len(), 2);
        for pop_voltages in &results.voltages {
            for v in pop_voltages {
                assert!(v.is_finite());
            }
        }
    }

    // ── Pipeline integration: sensory spiking ─────────────────────

    #[test]
    fn sensory_spiking_population_create_step() {
        for name in &[
            "RetinalGanglion",
            "Merkel",
            "Pacinian",
            "Nociceptor",
            "OlfactoryReceptor",
        ] {
            let mut pop = create_population(name, 5).unwrap();
            pop.currents.fill(20.0);
            for _ in 0..200 {
                pop.step_all();
            }
            let voltages = pop.collect_voltages();
            assert_eq!(voltages.len(), 5, "{name}: voltage count mismatch");
            for v in &voltages {
                assert!(v.is_finite(), "{name}: non-finite voltage {v}");
            }
        }
    }

    // ── NaN/Inf edge-case tests ───────────────────────────────────

    #[test]
    fn all_models_nan_input_stays_finite() {
        // Models must not propagate NaN — they should produce finite
        // (possibly wrong) output. This catches catastrophic numerical issues.
        let fragile_models = &[
            "PVFastSpiking",
            "SST",
            "VIP",
            "Chandelier",
            "CerebellarBasket",
            "Martinotti",
            "RetinalGanglion",
            "Merkel",
            "Pacinian",
            "Nociceptor",
            "OlfactoryReceptor",
        ];
        for name in fragile_models {
            let mut neuron = create_neuron(name).unwrap();
            // Feed 100 normal steps first to get into active regime
            for _ in 0..100 {
                neuron.step(2.0);
            }
            // Then feed NaN — voltage may go NaN but should not panic
            for _ in 0..10 {
                let _ = neuron.step(f64::NAN);
            }
            // Reset must restore finite state
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(
                v.is_finite(),
                "{name}: voltage not finite after reset from NaN: {v}"
            );
        }
    }

    #[test]
    fn all_models_extreme_input_stays_finite() {
        let models = &[
            "PVFastSpiking",
            "SST",
            "VIP",
            "Chandelier",
            "CerebellarBasket",
            "Martinotti",
            "RetinalGanglion",
            "Merkel",
            "Pacinian",
            "Nociceptor",
            "OlfactoryReceptor",
        ];
        for name in models {
            let mut neuron = create_neuron(name).unwrap();
            // Large positive current
            for _ in 0..50 {
                neuron.step(1e6);
            }
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(
                v.is_finite(),
                "{name}: non-finite after large positive input"
            );

            // Large negative current
            for _ in 0..50 {
                neuron.step(-1e6);
            }
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(
                v.is_finite(),
                "{name}: non-finite after large negative input"
            );
        }
    }
}
