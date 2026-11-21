// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner execution orchestration

//! Multi-population simulation orchestration over populations and projections.

#![deny(missing_docs)]

use super::simulation_results::pack_spike_event;
use super::{admit_csr, PopulationRunner, ProjectionRunner, SimResults};

/// Neuron populations connected by CSR projections, advanced together one step at a time.
pub struct NetworkRunner {
    /// The populations, indexed in the order they were added.
    pub populations: Vec<PopulationRunner>,
    /// The projections, applied on every step in the order they were added.
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
    /// Construct a network with no population and no projection.
    pub fn new() -> Self {
        Self {
            populations: Vec::new(),
            projections: Vec::new(),
        }
    }

    /// Append a population and return its index.
    pub fn add_population(&mut self, pop: PopulationRunner) -> usize {
        let idx = self.populations.len();
        self.populations.push(pop);
        idx
    }

    /// Append a projection without checking it.
    ///
    /// [`Self::admit_projection`] checks the population indices and the connectivity
    /// first. A projection whose source or target index does not exist makes
    /// [`Self::run`] panic.
    pub fn add_projection(&mut self, proj: ProjectionRunner) {
        self.projections.push(proj);
    }

    /// Admit and append a signed CSR projection, leaving the network unchanged on refusal.
    ///
    /// Both population indices must exist and the connectivity must pass
    /// [`admit_csr`] against their sizes; only then
    /// is the projection (and its delay buffer) allocated and appended.
    pub fn admit_projection(
        &mut self,
        src: usize,
        tgt: usize,
        row_offsets: &[i64],
        col_indices: &[i64],
        values: Vec<f64>,
        delay: usize,
    ) -> Result<(), String> {
        let n_pops = self.populations.len();
        let source = self.populations.get(src).ok_or_else(|| {
            format!("projection source population index {src} out of range ({n_pops} populations)")
        })?;
        let target = self.populations.get(tgt).ok_or_else(|| {
            format!("projection target population index {tgt} out of range ({n_pops} populations)")
        })?;
        let (offsets, columns) = admit_csr(
            source.len(),
            target.len(),
            row_offsets,
            col_indices,
            &values,
        )?;
        self.add_projection(ProjectionRunner::new(
            src, tgt, offsets, columns, values, delay,
        ));
        Ok(())
    }

    /// Validate population selection and current shape before worker admission.
    pub(crate) fn validate_population_currents(
        &self,
        pop_idx: usize,
        currents: &[f64],
    ) -> Result<(), String> {
        self.populations
            .get(pop_idx)
            .ok_or_else(|| format!("population index {pop_idx} out of range"))?
            .validate_currents(currents)
    }

    /// Set one population's input currents, advance only that population by one step, and
    /// return its spike flags and soma voltages.
    ///
    /// No projection is applied and no other population moves. An index that does not
    /// exist, or a current vector whose length differs from the population's size, returns
    /// `Err` and leaves the population unchanged.
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

    /// Run the whole network for `n_steps` steps and return its spikes and final voltages.
    ///
    /// Each step first sets every population's currents to zero. Each projection then adds,
    /// in the order the projections were added, its source population's spike flags as the
    /// previous step left them to its target's currents; on the first step these are the
    /// flags an earlier call left, zero for a new population. Every population then
    /// advances, in index order. A projection with a delay of `d` steps adds flags that are
    /// `d` propagations older, so a spike reaches its target's currents `d + 1` steps after
    /// it is emitted. Spike events are packed as `(neuron << 32) | step`, with steps
    /// counted from 0 in this call; the voltages are each population's soma voltages after
    /// the last step.
    ///
    /// # Panics
    ///
    /// When a projection names a population that does not exist, or when a spike occurs at
    /// a step above `u32::MAX`.
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
