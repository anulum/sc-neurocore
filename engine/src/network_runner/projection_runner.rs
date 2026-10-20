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

/// Admit signed CSR connectivity from `n_src` source rows into `n_tgt` target columns.
///
/// Returns the unsigned row offsets and column indices when `row_offsets` has
/// exactly `n_src + 1` entries starting at zero and never decreasing, the last
/// offset lies within both `col_indices` and `values` (which have equal
/// lengths), and every used column is in `0..n_tgt`. Every other input is
/// refused with a message naming the first violation, so no driven connection
/// is silently dropped and propagation never indexes outside the storage.
pub fn admit_csr(
    n_src: usize,
    n_tgt: usize,
    row_offsets: &[i64],
    col_indices: &[i64],
    values: &[f64],
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if row_offsets.len() != n_src + 1 {
        return Err(format!(
            "row_offsets has {} entries; a source population of {n_src} neurons needs {}",
            row_offsets.len(),
            n_src + 1
        ));
    }
    if row_offsets[0] != 0 {
        return Err(format!(
            "row_offsets must start at 0, got {}",
            row_offsets[0]
        ));
    }
    if let Some(i) = (0..n_src).find(|&i| row_offsets[i + 1] < row_offsets[i]) {
        return Err(format!(
            "row_offsets must not decrease: entry {} is {} after {}",
            i + 1,
            row_offsets[i + 1],
            row_offsets[i]
        ));
    }
    if col_indices.len() != values.len() {
        return Err(format!(
            "col_indices has {} entries but values has {}",
            col_indices.len(),
            values.len()
        ));
    }
    let used = row_offsets[n_src];
    if used as u64 > col_indices.len() as u64 {
        return Err(format!(
            "the last row offset {used} exceeds the {} stored connections",
            col_indices.len()
        ));
    }
    let used = used as usize;
    if let Some(k) =
        (0..used).find(|&k| col_indices[k] < 0 || col_indices[k] as u64 >= n_tgt as u64)
    {
        return Err(format!(
            "col_indices[{k}] = {} is outside a target population of {n_tgt} neurons",
            col_indices[k]
        ));
    }
    let offsets = row_offsets.iter().map(|&x| x as usize).collect();
    let columns = col_indices.iter().map(|&x| x as usize).collect();
    Ok((offsets, columns))
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
