// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Deterministic dense IF replay and signed linear readout

//! Owned dense IF replay under the dense-if-f64-sequential-v1 numerical profile.

#[path = "ann_to_snn_resources.rs"]
mod resources;

#[path = "ann_to_snn_parameters.rs"]
mod parameters;
pub use parameters::{DenseLayer, OutputMode, ReplayError};

/// Owned output responses, final states and optional layer-major traces.
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Batch-major final IF counts or cumulative signed linear output.
    pub output: Vec<f64>,
    /// One batch-major final state vector per weighted layer.
    pub final_state: Vec<Vec<f64>>,
    /// One time/batch/output ordered state trace per weighted layer.
    pub state_trace: Vec<Vec<f64>>,
    /// One time/batch/output ordered spike trace per IF layer.
    pub spike_trace: Vec<Vec<f64>>,
}

/// An owned checked feedforward IF stack; input encoding belongs to the frontend.
#[derive(Debug, Clone)]
pub struct ConvertedSNN {
    layers: Vec<DenseLayer>,
    initial_fraction: f64,
    output_mode: OutputMode,
}

impl ConvertedSNN {
    /// Require a nonempty connected stack and a finite default membrane shift.
    pub fn new(
        layers: Vec<DenseLayer>,
        initial_fraction: f64,
        output_mode: OutputMode,
    ) -> Result<Self, ReplayError> {
        if layers.is_empty()
            || !initial_fraction.is_finite()
            || layers
                .windows(2)
                .any(|pair| pair[0].outputs != pair[1].inputs)
        {
            return Err(ReplayError::InvalidInput);
        }
        Ok(Self {
            layers,
            initial_fraction,
            output_mode,
        })
    }

    /// Replay complete explicit frames with independently owned state buffers.
    ///
    /// Shape contains (steps, batch). Frames use time/batch/input order. Initial states and returned traces use
    /// the same layer-major, then time/batch/output order as Python. Multiplication
    /// and addition remain separate; input columns are reduced in ascending order.
    /// Invalid late input or overflow returns no partially updated caller state.
    /// Numeric reservation matches Python: 8 * (2P + 2F + 2S + 2O + H + 5M + I).
    /// The byte budget excludes caller buffers and interpreter/allocator overhead.
    pub fn replay(
        &self,
        frames: &[f64],
        shape: (usize, usize),
        initial: Option<&[Vec<f64>]>,
        trace: bool,
        binary: bool,
        max_working_bytes: usize,
    ) -> Result<ReplayResult, ReplayError> {
        let borrowed = initial.map(|states| states.iter().map(Vec::as_slice).collect::<Vec<_>>());
        self.replay_borrowed(
            frames,
            shape,
            borrowed.as_deref(),
            trace,
            binary,
            max_working_bytes,
        )
    }

    /// Replay borrowed initial-state slices without an intermediate numeric copy.
    ///
    /// Semantics, reservation, owned results and error atomicity match `replay`.
    /// Initial slices are read-only and copied once into owned final states.
    pub fn replay_borrowed(
        &self,
        frames: &[f64],
        shape: (usize, usize),
        initial: Option<&[&[f64]]>,
        trace: bool,
        binary: bool,
        max_working_bytes: usize,
    ) -> Result<ReplayResult, ReplayError> {
        let (steps, batch) = shape;
        let expected = steps
            .checked_mul(batch)
            .and_then(|n| n.checked_mul(self.layers[0].inputs));
        if expected != Some(frames.len()) {
            return Err(ReplayError::InvalidInput);
        }
        if initial.is_some_and(|v| v.len() != self.layers.len()) {
            return Err(ReplayError::InvalidInput);
        }
        if max_working_bytes == 0 || max_working_bytes > isize::MAX as usize {
            return Err(ReplayError::InvalidInput);
        }
        let coefficients = self
            .layers
            .iter()
            .try_fold(0usize, |n, layer| {
                n.checked_add(layer.weights.len())?
                    .checked_add(layer.bias.as_ref().map_or(0, Vec::len))
            })
            .ok_or(ReplayError::ResourceLimit)?;
        let widths: Vec<usize> = self.layers.iter().map(|layer| layer.outputs).collect();
        let work = resources::replay_bytes(
            coefficients,
            self.layers[0].inputs,
            &widths,
            steps,
            batch,
            trace,
            self.output_mode == OutputMode::Linear,
        )
        .ok_or(ReplayError::ResourceLimit)?;
        if work > max_working_bytes {
            return Err(ReplayError::ResourceLimit);
        }
        if frames
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0 || (binary && *v != 0.0 && *v != 1.0))
        {
            return Err(ReplayError::InvalidInput);
        }
        let spiking = self.layers.len() - usize::from(self.output_mode == OutputMode::Linear);
        let mut states = Vec::with_capacity(self.layers.len());
        let mut state_trace = Vec::new();
        let mut spike_trace = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let width = batch
                .checked_mul(layer.outputs)
                .ok_or(ReplayError::ResourceLimit)?;
            let state = if let Some(values) = initial {
                if values[index].len() != width || values[index].iter().any(|v| !v.is_finite()) {
                    return Err(ReplayError::InvalidInput);
                }
                values[index].to_vec()
            } else {
                let shift = if index < spiking {
                    layer.initial_fraction.unwrap_or(self.initial_fraction) * layer.threshold
                } else {
                    0.0
                };
                if !shift.is_finite() {
                    return Err(ReplayError::Overflow);
                }
                vec![shift; width]
            };
            states.push(state);
            if trace {
                let length = steps.checked_mul(width).ok_or(ReplayError::ResourceLimit)?;
                state_trace.push(vec![0.0; length]);
                if index < spiking {
                    spike_trace.push(vec![0.0; length]);
                }
            }
        }
        let output_width = batch
            .checked_mul(self.layers.last().unwrap().outputs)
            .ok_or(ReplayError::ResourceLimit)?;
        let mut output = vec![0.0; output_width];
        if batch != 0 {
            for step in 0..steps {
                let offset = step * batch * self.layers[0].inputs;
                let mut drive = frames[offset..offset + batch * self.layers[0].inputs].to_vec();
                for (index, layer) in self.layers.iter().enumerate() {
                    let mut events = vec![0.0; batch * layer.outputs];
                    for row in 0..batch {
                        for node in 0..layer.outputs {
                            let mut current = 0.0;
                            for column in 0..layer.inputs {
                                let product = drive[row * layer.inputs + column]
                                    * layer.weights[node * layer.inputs + column];
                                current += product;
                            }
                            if let Some(bias) = &layer.bias {
                                current += bias[node];
                            }
                            let slot = row * layer.outputs + node;
                            states[index][slot] += current;
                            if !current.is_finite() || !states[index][slot].is_finite() {
                                return Err(ReplayError::Overflow);
                            }
                            if index < spiking {
                                let event = f64::from(states[index][slot] >= layer.threshold);
                                states[index][slot] -= event * layer.threshold;
                                events[slot] = event;
                                if trace {
                                    spike_trace[index][step * batch * layer.outputs + slot] = event;
                                }
                                if index == self.layers.len() - 1 {
                                    output[slot] += event;
                                }
                            }
                            if trace {
                                state_trace[index][step * batch * layer.outputs + slot] =
                                    states[index][slot];
                            }
                        }
                    }
                    drive = events;
                }
            }
        }
        if self.output_mode == OutputMode::Linear {
            output = states.last().unwrap().clone();
        }
        Ok(ReplayResult {
            output,
            final_state: states,
            state_trace,
            spike_trace,
        })
    }

    /// Select the first maximal finite response from each row of a replay result.
    pub fn classify(&self, result: &ReplayResult, batch: usize) -> Result<Vec<usize>, ReplayError> {
        let width = self.layers.last().unwrap().outputs;
        if batch.checked_mul(width) != Some(result.output.len())
            || result.output.iter().any(|v| !v.is_finite())
        {
            return Err(ReplayError::InvalidInput);
        }
        Ok(result
            .output
            .chunks_exact(width)
            .map(|row| {
                let mut best = 0;
                for i in 1..width {
                    if row[i] > row[best] {
                        best = i;
                    }
                }
                best
            })
            .collect())
    }
}

#[cfg(test)]
#[path = "ann_to_snn_tests.rs"]
mod tests;
