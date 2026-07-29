// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC NetworkRunner (no_std)

//! Fixed-capacity network runner for bare-metal SC inference.
//!
//! Uses stack-allocated arrays (no heap) with compile-time capacity
//! bounds. Supports multi-layer feed-forward SC networks with
//! configurable neuron types and bitstream lengths.

use crate::lfsr::Lfsr16;
use crate::neuron::LifNeuron;

/// Maximum neurons per layer (compile-time bound for stack allocation).
pub const MAX_NEURONS_PER_LAYER: usize = 64;

/// Maximum layers in a network.
pub const MAX_LAYERS: usize = 8;

/// Maximum bitstream words per neuron (32 bits/word × 16 = 512 bit max).
pub const MAX_BS_WORDS: usize = 16;

/// Per-layer configuration.
pub struct LayerConfig {
    pub num_neurons: usize,
    pub bitstream_length: u16,
    pub threshold: u32,
    pub leak_shift: u8,
    pub lfsr_seed: u16,
}

impl LayerConfig {
    pub const fn new(
        num_neurons: usize,
        bitstream_length: u16,
        threshold: u32,
        leak_shift: u8,
        lfsr_seed: u16,
    ) -> Self {
        Self {
            num_neurons,
            bitstream_length,
            threshold,
            leak_shift,
            lfsr_seed,
        }
    }

    /// Number of u32 words per bitstream.
    pub const fn words_per_bs(&self) -> usize {
        (self.bitstream_length as usize).div_ceil(32)
    }
}

/// Fixed-capacity layer state.
pub struct LayerState {
    pub neurons: [LifNeuron; MAX_NEURONS_PER_LAYER],
    pub num_neurons: usize,
    pub lfsr: Lfsr16,
    pub config: LayerConfig,
    /// Output spike flags for this layer (one bit per neuron).
    pub spike_mask: u64,
}

impl LayerState {
    pub fn new(config: LayerConfig) -> Self {
        let mut neurons = [const { LifNeuron::new(0, 0, 0) }; MAX_NEURONS_PER_LAYER];
        for neuron in neurons
            .iter_mut()
            .take(config.num_neurons.min(MAX_NEURONS_PER_LAYER))
        {
            *neuron = LifNeuron::new(config.threshold, config.leak_shift, 1);
        }
        Self {
            neurons,
            num_neurons: config.num_neurons.min(MAX_NEURONS_PER_LAYER),
            lfsr: Lfsr16::new(config.lfsr_seed),
            config,
            spike_mask: 0,
        }
    }

    /// Run one timestep: encode input, feed to neurons, collect spikes.
    pub fn tick(&mut self, input_popcount: u32) {
        let words = self.config.words_per_bs();
        let mut bs_buf = [0u32; MAX_BS_WORDS];
        self.spike_mask = 0;

        for i in 0..self.num_neurons {
            let threshold_q16 = (input_popcount.min(65535) as u16).wrapping_add(i as u16 * 7919);
            self.lfsr.encode_into(
                threshold_q16,
                self.config.bitstream_length as usize,
                &mut bs_buf[..words],
            );

            if self.neurons[i].tick(&bs_buf[..words]) {
                self.spike_mask |= 1u64 << i;
            }
        }
    }

    /// Return popcount of active spikes (for feeding to next layer).
    pub fn output_popcount(&self) -> u32 {
        self.spike_mask.count_ones()
    }

    /// Reset all neurons.
    pub fn reset(&mut self) {
        for i in 0..self.num_neurons {
            self.neurons[i].reset();
        }
        self.spike_mask = 0;
        self.lfsr.reset(self.config.lfsr_seed);
    }
}

/// Fixed-capacity network runner.
pub struct NetworkRunner {
    pub layers: [Option<LayerState>; MAX_LAYERS],
    pub num_layers: usize,
    pub total_ticks: u64,
}

impl NetworkRunner {
    /// Create an empty network.
    pub const fn new() -> Self {
        Self {
            layers: [const { None }; MAX_LAYERS],
            num_layers: 0,
            total_ticks: 0,
        }
    }

    /// Add a layer. Returns layer index, or `None` if full.
    pub fn add_layer(&mut self, config: LayerConfig) -> Option<usize> {
        if self.num_layers >= MAX_LAYERS {
            return None;
        }
        let idx = self.num_layers;
        self.layers[idx] = Some(LayerState::new(config));
        self.num_layers += 1;
        Some(idx)
    }

    /// Run one timestep through all layers.
    ///
    /// Returns the output spike mask of the final layer.
    pub fn tick(&mut self, input_popcount: u32) -> u64 {
        let mut current_input = input_popcount;
        for i in 0..self.num_layers {
            if let Some(ref mut layer) = self.layers[i] {
                layer.tick(current_input);
                current_input = layer.output_popcount();
            }
        }
        self.total_ticks += 1;
        self.layers[self.num_layers.saturating_sub(1)]
            .as_ref()
            .map_or(0, |l| l.spike_mask)
    }

    /// Reset the entire network.
    pub fn reset(&mut self) {
        for i in 0..self.num_layers {
            if let Some(ref mut layer) = self.layers[i] {
                layer.reset();
            }
        }
        self.total_ticks = 0;
    }

    /// Get total spike count across all layers for the last tick.
    pub fn total_spikes(&self) -> u32 {
        let mut total = 0u32;
        for i in 0..self.num_layers {
            if let Some(ref layer) = self.layers[i] {
                total += layer.output_popcount();
            }
        }
        total
    }
}

impl Default for NetworkRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LayerConfig {
        LayerConfig::new(4, 256, 10, 1, 0xACE1)
    }

    #[test]
    fn test_layer_state_creation() {
        let s = LayerState::new(test_config());
        assert_eq!(s.num_neurons, 4);
        assert_eq!(s.spike_mask, 0);
    }

    #[test]
    fn test_layer_tick() {
        let mut s = LayerState::new(test_config());
        s.tick(100);
        // Just verify no panic; spike behavior depends on LFSR output
    }

    #[test]
    fn test_layer_reset() {
        let mut s = LayerState::new(test_config());
        s.tick(100);
        s.reset();
        assert_eq!(s.spike_mask, 0);
    }

    #[test]
    fn test_network_add_layer() {
        let mut net = NetworkRunner::new();
        let idx = net.add_layer(test_config());
        assert_eq!(idx, Some(0));
        assert_eq!(net.num_layers, 1);
    }

    #[test]
    fn test_network_max_layers() {
        let mut net = NetworkRunner::new();
        for i in 0..MAX_LAYERS {
            let seed = 0xACE1u16.wrapping_add(i as u16 * 1000);
            assert!(net
                .add_layer(LayerConfig::new(2, 64, 5, 0, seed.max(1)))
                .is_some());
        }
        assert!(net.add_layer(test_config()).is_none());
    }

    #[test]
    fn test_network_tick() {
        let mut net = NetworkRunner::new();
        net.add_layer(LayerConfig::new(4, 64, 5, 0, 0xACE1));
        net.add_layer(LayerConfig::new(2, 64, 3, 0, 0xBEEF));
        let _mask = net.tick(20);
        assert_eq!(net.total_ticks, 1);
    }

    #[test]
    fn test_network_reset() {
        let mut net = NetworkRunner::new();
        net.add_layer(test_config());
        net.tick(10);
        net.tick(10);
        net.reset();
        assert_eq!(net.total_ticks, 0);
    }

    #[test]
    fn test_empty_network() {
        let mut net = NetworkRunner::new();
        let mask = net.tick(42);
        assert_eq!(mask, 0);
    }

    #[test]
    fn test_total_spikes() {
        let mut net = NetworkRunner::new();
        net.add_layer(LayerConfig::new(4, 64, 5, 0, 0xACE1));
        net.tick(100);
        let spikes = net.total_spikes();
        assert!(spikes <= 4); // max 4 neurons
    }
}
