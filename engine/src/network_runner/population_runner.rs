// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner population execution

//! Parallel execution and state management for one heterogeneous neuron population.

use rayon::prelude::*;

use super::NeuronVariant;

pub struct PopulationRunner {
    pub(super) neurons: Vec<NeuronVariant>,
    pub(super) spikes: Vec<u8>,
    pub(super) currents: Vec<f64>,
}

const CHUNK_SIZE: usize = 256;

impl PopulationRunner {
    pub fn new(neurons: Vec<NeuronVariant>) -> Self {
        let n = neurons.len();
        Self {
            neurons,
            spikes: vec![0u8; n],
            currents: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    pub fn step_all(&mut self) {
        let neurons = &mut self.neurons;
        let spikes = &mut self.spikes;
        let currents = &self.currents;

        neurons
            .par_chunks_mut(CHUNK_SIZE)
            .zip(spikes.par_chunks_mut(CHUNK_SIZE))
            .zip(currents.par_chunks(CHUNK_SIZE))
            .for_each(|((n_chunk, s_chunk), c_chunk)| {
                for i in 0..n_chunk.len() {
                    let fired = n_chunk[i].step(c_chunk[i]);
                    s_chunk[i] = if fired != 0 { 1 } else { 0 };
                }
            });
    }

    pub fn reset_all(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
        self.spikes.fill(0);
        self.currents.fill(0.0);
    }

    pub fn reset_currents(&mut self) {
        self.currents.fill(0.0);
    }

    pub fn set_currents(&mut self, currents: &[f64]) -> Result<(), String> {
        if currents.len() != self.currents.len() {
            return Err(format!(
                "current vector length mismatch: got {}, expected {}",
                currents.len(),
                self.currents.len()
            ));
        }
        self.currents.copy_from_slice(currents);
        Ok(())
    }

    pub fn collect_voltages(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.soma_voltage()).collect()
    }
}
