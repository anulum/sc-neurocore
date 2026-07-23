// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner projection propagation

//! CSR synaptic projection propagation with optional discrete axonal delay.

/// CSR-stored synaptic projection with optional axonal delay.
pub struct ProjectionRunner {
    pub src_pop: usize,
    pub tgt_pop: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
    delay_steps: usize,
    delay_buffer: Vec<Vec<u8>>,
    buf_idx: usize,
}

impl ProjectionRunner {
    pub fn new(
        src_pop: usize,
        tgt_pop: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        delay_steps: usize,
    ) -> Self {
        let n_delay = if delay_steps > 0 { delay_steps } else { 0 };
        let n_src = if row_offsets.is_empty() {
            0
        } else {
            row_offsets.len() - 1
        };
        let delay_buffer = if n_delay > 0 {
            vec![vec![0u8; n_src]; n_delay]
        } else {
            Vec::new()
        };
        Self {
            src_pop,
            tgt_pop,
            row_offsets,
            col_indices,
            values,
            delay_steps: n_delay,
            delay_buffer,
            buf_idx: 0,
        }
    }

    /// Scatter spikes through CSR connectivity into target current buffer.
    pub fn propagate(&mut self, src_spikes: &[u8], tgt_currents: &mut [f64]) {
        let spikes = if self.delay_steps > 0 {
            let delayed = &self.delay_buffer[self.buf_idx];
            let out: Vec<u8> = delayed.clone();
            self.delay_buffer[self.buf_idx] = src_spikes.to_vec();
            self.buf_idx = (self.buf_idx + 1) % self.delay_steps;
            out
        } else {
            src_spikes.to_vec()
        };

        let n_src = self.row_offsets.len().saturating_sub(1);
        for i in 0..n_src {
            if spikes.get(i).copied().unwrap_or(0) == 0 {
                continue;
            }
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];
            for k in start..end {
                let j = self.col_indices[k];
                if j < tgt_currents.len() {
                    tgt_currents[j] += self.values[k];
                }
            }
        }
    }
}
