// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sliding-window bitstream probability estimator

/// Sliding-window bitstream probability estimator.
///
/// Mirrors Python's `BitstreamAverager`.
#[derive(Clone, Debug)]
pub struct BitstreamAverager {
    buffer: Vec<u8>,
    index: usize,
    filled: bool,
    running_sum: u64,
}

impl BitstreamAverager {
    /// Create a positive sliding window, panicking on invalid size or allocation failure.
    ///
    /// Use [`Self::try_new`] for fallible construction.
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self::try_new(window).expect("cannot allocate bitstream averager buffer")
    }

    /// Create a positive sliding window with fallible buffer allocation.
    ///
    /// # Errors
    ///
    /// Returns `window must be > 0` for an empty window, or
    /// `Bitstream averager buffer cannot be allocated` for capacity or allocation failure.
    pub fn try_new(window: usize) -> Result<Self, &'static str> {
        if window == 0 {
            return Err("window must be > 0");
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(window)
            .map_err(|_| "Bitstream averager buffer cannot be allocated")?;
        buffer.resize(window, 0);
        Ok(Self {
            buffer,
            index: 0,
            filled: false,
            running_sum: 0,
        })
    }

    /// Add a binary sample, panicking before mutation for another byte value.
    ///
    /// Use [`Self::try_push`] for nonconsuming input refusal.
    pub fn push(&mut self, bit: u8) {
        self.try_push(bit).expect("bit must be 0 or 1");
    }

    /// Add a binary sample without changing state when the byte is invalid.
    ///
    /// # Errors
    ///
    /// Returns `Bit must be 0 or 1.` for a byte greater than one.
    pub fn try_push(&mut self, bit: u8) -> Result<(), &'static str> {
        if bit > 1 {
            return Err("Bit must be 0 or 1.");
        }
        let old = self.buffer[self.index];
        self.buffer[self.index] = bit;

        if self.filled {
            self.running_sum = self.running_sum - old as u64 + bit as u64;
        } else {
            self.running_sum += bit as u64;
        }

        self.index += 1;
        if self.index == self.buffer.len() {
            self.index = 0;
            self.filled = true;
        }
        Ok(())
    }

    /// Estimate the probability using only observed samples until the window fills.
    pub fn estimate(&self) -> f64 {
        if !self.filled {
            if self.index == 0 {
                return 0.0;
            }
            return self.running_sum as f64 / self.index as f64;
        }
        self.running_sum as f64 / self.buffer.len() as f64
    }

    /// Discard all observations while retaining the allocated window.
    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.index = 0;
        self.filled = false;
        self.running_sum = 0;
    }

    /// Return the number of samples retained after the window fills.
    pub fn window(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
#[path = "bitstream_averager/tests.rs"]
mod tests;
