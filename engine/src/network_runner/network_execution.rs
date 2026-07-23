// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner execution orchestration

//! Multi-population simulation orchestration over populations and projections.

use super::simulation_results::pack_spike_event;
use super::{PopulationRunner, ProjectionRunner, SimResults};

pub struct NetworkRunner {
    pub populations: Vec<PopulationRunner>,
    pub projections: Vec<ProjectionRunner>,
}

fn propagate_projection(populations: &mut [PopulationRunner], projection: &mut ProjectionRunner) {
    let src = projection.src_pop;
    let tgt = projection.tgt_pop;
    assert!(
        src < populations.len(),
        "projection source population index {src} out of range"
    );
    assert!(
        tgt < populations.len(),
        "projection target population index {tgt} out of range"
    );

    if src == tgt {
        let population = &mut populations[src];
        let spikes = population.spikes.clone();
        projection.propagate(&spikes, &mut population.currents);
    } else if src < tgt {
        let (before_target, target_and_after) = populations.split_at_mut(tgt);
        projection.propagate(
            &before_target[src].spikes,
            &mut target_and_after[0].currents,
        );
    } else {
        let (before_source, source_and_after) = populations.split_at_mut(src);
        projection.propagate(
            &source_and_after[0].spikes,
            &mut before_source[tgt].currents,
        );
    }
}

impl NetworkRunner {
    pub fn new() -> Self {
        Self {
            populations: Vec::new(),
            projections: Vec::new(),
        }
    }

    pub fn add_population(&mut self, pop: PopulationRunner) -> usize {
        let idx = self.populations.len();
        self.populations.push(pop);
        idx
    }

    pub fn add_projection(&mut self, proj: ProjectionRunner) {
        self.projections.push(proj);
    }

    pub fn step_population_with_currents(
        &mut self,
        pop_idx: usize,
        currents: &[f64],
    ) -> Result<(Vec<u8>, Vec<f64>), String> {
        let pop = self
            .populations
            .get_mut(pop_idx)
            .ok_or_else(|| format!("population index {pop_idx} out of range"))?;
        pop.set_currents(currents)?;
        pop.step_all();
        Ok((pop.spikes.clone(), pop.collect_voltages()))
    }

    pub fn run(&mut self, n_steps: usize) -> SimResults {
        let n_pops = self.populations.len();
        let mut spike_counts = vec![0usize; n_pops];
        let mut spike_data: Vec<Vec<u64>> = vec![Vec::new(); n_pops];

        for t in 0..n_steps {
            // Reset currents
            for pop in &mut self.populations {
                pop.reset_currents();
            }

            let (populations, projections) = (&mut self.populations, &mut self.projections);
            for projection in projections {
                propagate_projection(populations, projection);
            }

            // Step all populations
            for (pop_idx, pop) in self.populations.iter_mut().enumerate() {
                pop.step_all();
                for (nid, &spike) in pop.spikes.iter().enumerate() {
                    if spike != 0 {
                        spike_counts[pop_idx] += 1;
                        spike_data[pop_idx].push(pack_spike_event(nid, t));
                    }
                }
            }
        }

        let voltages: Vec<Vec<f64>> = self
            .populations
            .iter()
            .map(|p| p.collect_voltages())
            .collect();

        SimResults {
            spike_counts,
            spike_data,
            voltages,
        }
    }
}

impl Default for NetworkRunner {
    fn default() -> Self {
        Self::new()
    }
}
