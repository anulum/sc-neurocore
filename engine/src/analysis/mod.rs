// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike train analysis (Rust engine)

//! Spike train analysis: statistics, distances, correlation, information, decoding,
//! dimensionality reduction, spectral and point-process measures, surrogates and
//! spike-sorting quality. Each submodule below holds one family.

#![deny(missing_docs)]

/// Basic spike train operations: spike times, inter-spike intervals, rates, counts and binning.
pub mod basic;
pub(crate) mod bindings;
/// Granger causality and directed connectivity: pairwise, conditional, spectral, PDC and DTF.
pub mod causality;
/// Cross-correlation, synchrony, coherence, tiling and covariance between spike trains.
pub mod correlation;
/// Population decoders: population vector, Bayesian, maximum-likelihood, LDA and naive Bayes.
pub mod decoding;
/// Dimensionality reduction of spike train populations: PCA, demixed PCA and factor analysis.
pub mod dimensionality;
/// Spike train distances: van Rossum, Victor-Purpura, ISI, SPIKE and SPIKE-sync.
pub mod distance;
/// Gaussian Process Factor Analysis: smooth latent trajectories fitted by EM.
pub mod gpfa;
/// Information measures: mutual information, transfer entropy and spike train entropies.
pub mod information;
/// Spike-LFP coupling: phase locking value, spike-field coherence and spike phase histograms.
pub mod lfp;
/// Network analysis: functional connectivity, unitary events, cell assemblies, synfire chains.
pub mod network;
pub mod neural_decoders;
/// Spike directionality, train ordering and higher-order patterns.
pub mod patterns;
/// Point process models: conditional intensity, ISI hazard and survivor, renewal density.
pub mod point_process;
/// Firing rate estimation: instantaneous rate, population rate and PSTH.
pub mod rate;
/// Spike sorting quality metrics: isolation distance, L-ratio, d', ISI violations and more.
pub mod sorting_quality;
/// SPADE: spike pattern detection and evaluation.
pub mod spade;
/// Spectral analysis of spike trains: the power spectrum.
pub mod spectral;
/// Bootstrap significance testing.
pub mod statistics;
/// Spike-triggered analysis and receptive fields: STA, STC, place fields and tuning curves.
pub mod stimulus;
/// Surrogate spike trains for significance testing: shuffles, dithers and renewal processes.
pub mod surrogates;
/// Temporal patterns: bursts, first-spike latency, response onset and change points.
pub mod temporal;
/// Variability and regularity: CV, CV2, local variation, Fano factor and ISI entropy.
pub mod variability;
/// Spike waveform shape: width, amplitude, slopes, half-width and peak-to-trough ratio.
pub mod waveform;
