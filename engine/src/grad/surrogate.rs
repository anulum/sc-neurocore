// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Surrogate Gradient Components

//! # Surrogate Gradient Components
//!
//! Differentiable wrappers around SC forward operators.

#![deny(missing_docs)]

use rayon::prelude::*;

/// Surrogate for the derivative of the spike step, as a function of the membrane's offset
/// `x` from threshold.
///
/// Each parameter must be finite and above 0. The constructors ([`Self::fast_sigmoid`] and
/// the others) and [`Self::check`] refuse any other value, and so do
/// [`SurrogateLif::try_new`] and [`DifferentiableDenseLayer::try_new`] for a variant built
/// directly. Unchecked, a `k` of 0 makes the fast sigmoid infinite and the SuperSpike and
/// arctan forms 1 everywhere; a negative `k` makes the fast sigmoid negative and the
/// SuperSpike form infinite at `|x| = -1/k`. A `width` of 0 gives 0 everywhere, and a
/// negative one a derivative whose size grows with `|x|`, negative for the triangular pulse.
#[derive(Clone, Debug)]
pub enum SurrogateType {
    /// Fast Sigmoid: d/dx = 1 / (2k * (1 + k|x|)^2)
    FastSigmoid {
        /// Steepness; the derivative peaks at `x = 0` with `1 / (2k)`.
        k: f32,
    },
    /// SuperSpike: d/dx = 1 / (1 + k|x|)^2
    SuperSpike {
        /// Steepness; the derivative peaks at `x = 0` with 1.
        k: f32,
    },
    /// ArcTan: d/dx = 1 / (1 + (kx)^2)
    ArcTan {
        /// Steepness; the derivative peaks at `x = 0` with 1.
        k: f32,
    },
    /// Straight-through estimator, as a window: 1 where `|x| < 0.5`, 0 elsewhere.
    StraightThrough,
    /// Triangular pulse: max(0, 1 - |x|/width) / width
    Triangular {
        /// Half-width of the pulse; the peak, at `x = 0`, is `1 / width`, so the area is 1.
        width: f32,
    },
    /// Piecewise linear: max(0, 1 - |x|/width)
    PiecewiseLinear {
        /// Half-width of the pulse; the peak, at `x = 0`, is 1.
        width: f32,
    },
}

/// Return `value` when it is finite and above 0, or `reason`.
fn positive(value: f32, reason: &'static str) -> Result<f32, &'static str> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(reason)
    }
}

/// Why a steepness is refused.
const K_REFUSED: &str = "surrogate k must be finite and above 0";

/// Why a width is refused.
const WIDTH_REFUSED: &str = "surrogate width must be finite and above 0";

impl SurrogateType {
    /// A fast sigmoid of steepness `k`.
    ///
    /// # Errors
    ///
    /// When `k` is not finite and above 0.
    pub fn fast_sigmoid(k: f32) -> Result<Self, &'static str> {
        Ok(Self::FastSigmoid {
            k: positive(k, K_REFUSED)?,
        })
    }

    /// A SuperSpike surrogate of steepness `k`.
    ///
    /// # Errors
    ///
    /// When `k` is not finite and above 0.
    pub fn super_spike(k: f32) -> Result<Self, &'static str> {
        Ok(Self::SuperSpike {
            k: positive(k, K_REFUSED)?,
        })
    }

    /// An arctan surrogate of steepness `k`.
    ///
    /// # Errors
    ///
    /// When `k` is not finite and above 0.
    pub fn arc_tan(k: f32) -> Result<Self, &'static str> {
        Ok(Self::ArcTan {
            k: positive(k, K_REFUSED)?,
        })
    }

    /// A triangular pulse of half-width `width`.
    ///
    /// # Errors
    ///
    /// When `width` is not finite and above 0.
    pub fn triangular(width: f32) -> Result<Self, &'static str> {
        Ok(Self::Triangular {
            width: positive(width, WIDTH_REFUSED)?,
        })
    }

    /// A piecewise-linear pulse of half-width `width`.
    ///
    /// # Errors
    ///
    /// When `width` is not finite and above 0.
    pub fn piecewise_linear(width: f32) -> Result<Self, &'static str> {
        Ok(Self::PiecewiseLinear {
            width: positive(width, WIDTH_REFUSED)?,
        })
    }

    /// Say whether this surrogate's parameter is finite and above 0.
    ///
    /// # Errors
    ///
    /// `"surrogate k must be finite and above 0"` or `"surrogate width must be finite and
    /// above 0"`; the straight-through window has no parameter and is always accepted.
    pub fn check(&self) -> Result<(), &'static str> {
        match self {
            Self::FastSigmoid { k } | Self::SuperSpike { k } | Self::ArcTan { k } => {
                positive(*k, K_REFUSED).map(|_| ())
            }
            Self::StraightThrough => Ok(()),
            Self::Triangular { width } | Self::PiecewiseLinear { width } => {
                positive(*width, WIDTH_REFUSED).map(|_| ())
            }
        }
    }

    /// Evaluate surrogate derivative at membrane offset `x`.
    pub fn grad(&self, x: f32) -> f32 {
        match self {
            Self::FastSigmoid { k } => {
                // Zenke & Vogels 2021 normalization includes 1/(2k).
                let denom = 1.0 + k * x.abs();
                1.0 / (2.0 * k * denom * denom)
            }
            Self::SuperSpike { k } => {
                // Zenke & Ganguli 2018 unnormalized surrogate.
                let denom = 1.0 + k * x.abs();
                1.0 / (denom * denom)
            }
            Self::ArcTan { k } => 1.0 / (1.0 + (k * x).powi(2)),
            Self::StraightThrough => {
                if x.abs() < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Triangular { width } => {
                let v = 1.0 - x.abs() / width;
                if v > 0.0 {
                    v / width
                } else {
                    0.0
                }
            }
            Self::PiecewiseLinear { width } => {
                let v = 1.0 - x.abs() / width;
                v.max(0.0)
            }
        }
    }
}

/// LIF neuron with surrogate gradient support.
///
/// Each `forward` caches the offset from threshold, `(u - v_threshold) / 2^fraction`, of the
/// membrane `u` before any reset: the integrated value the step compared with the threshold.
/// On a spike step that is the membrane that crossed threshold, not the `v_reset` the step
/// returns, because the surrogate stands in for the derivative of the spike
/// `S[n] = Θ(U[n] - ϑ)` and the reset acts on `U[n + 1]` (Neftci, Mostafa & Zenke 2019,
/// arXiv:1901.09948v2, p. 5, eq. 5). On a refractory step the step returns no spike and the
/// held `v_rest` whatever the membrane and input are, so the output does not depend on the
/// membrane there: `backward` gives exactly 0 for that step, the derivative of the forward
/// pass, not a surrogate value.
pub struct SurrogateLif {
    /// The fixed-point LIF that runs each step.
    pub lif: crate::neuron::FixedPointLif,
    /// Surrogate evaluated by `backward` at a cached offset.
    pub surrogate: SurrogateType,
    /// Per cached step: the membrane's offset from threshold, and whether the step was
    /// refractory.
    membrane_trace: Vec<(f32, bool)>,
}

impl SurrogateLif {
    /// Construct a surrogate-enabled fixed-point LIF neuron.
    ///
    /// # Panics
    ///
    /// If [`SurrogateType::check`] refuses the surrogate, or
    /// [`FixedPointLif::try_new`](crate::neuron::FixedPointLif::try_new) the configuration, with
    /// the refusal's words; [`Self::try_new`] returns the refusal instead. At 4.1.0 a refused
    /// surrogate was built.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: SurrogateType,
    ) -> Self {
        Self::try_new(
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
            surrogate,
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Construct a surrogate-enabled fixed-point LIF neuron, or return why the configuration is
    /// refused.
    ///
    /// # Errors
    ///
    /// The surrogate's refusal from [`SurrogateType::check`], checked first; else the reason
    /// [`FixedPointLif::try_new`](crate::neuron::FixedPointLif::try_new) gives: a
    /// `data_width` outside `[1, 16]`, a `fraction` not below it, or a negative
    /// `refractory_period`.
    pub fn try_new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: SurrogateType,
    ) -> Result<Self, &'static str> {
        surrogate.check()?;
        Ok(Self {
            lif: crate::neuron::FixedPointLif::try_new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
            )?,
            surrogate,
            membrane_trace: Vec::new(),
        })
    }

    /// Forward LIF step, returning what the LIF's `step` returns and caching, for the backward
    /// pass, the offset of the membrane before any reset and whether the step was refractory.
    pub fn forward(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let refractory = self.lif.refractory_counter > 0;
        let (spike, v_out, membrane) = self.lif.step_with_membrane(leak_k, gain_k, i_t, noise_in);
        let scale = (1_u32 << self.lif.fraction) as f32;
        let v_norm = (membrane as f32 - self.lif.v_threshold as f32) / scale;
        self.membrane_trace.push((v_norm, refractory));
        (spike, v_out)
    }

    /// Backward pass through last cached membrane value.
    ///
    /// The value is removed, so repeated calls walk back through the cached steps, latest
    /// first. Returns `grad_output` times the surrogate at the value, or 0 for a refractory
    /// step, whose output does not depend on the membrane.
    ///
    /// # Errors
    ///
    /// When no step is cached.
    pub fn backward(&mut self, grad_output: f32) -> Result<f32, String> {
        let (v_norm, refractory) = self
            .membrane_trace
            .pop()
            .ok_or_else(|| "backward() called without matching forward()".to_string())?;
        if refractory {
            return Ok(0.0);
        }
        Ok(grad_output * self.surrogate.grad(v_norm))
    }

    /// Clear stored membrane trace.
    pub fn clear_trace(&mut self) {
        self.membrane_trace.clear();
    }

    /// Reset neuron state and clear trace.
    pub fn reset(&mut self) {
        self.lif.reset();
        self.clear_trace();
    }

    /// Number of cached forward steps.
    pub fn trace_len(&self) -> usize {
        self.membrane_trace.len()
    }
}

/// Dense SC layer with surrogate gradient backward pass.
pub struct DifferentiableDenseLayer {
    /// The dense SC layer that runs the forward pass and holds the weights.
    pub layer: crate::layer::DenseLayer,
    /// Surrogate evaluated by `backward` at each cached output minus 0.5.
    pub surrogate: SurrogateType,
    input_cache: Vec<f64>,
    output_cache: Vec<f64>,
}

impl DifferentiableDenseLayer {
    /// Construct a differentiable dense SC layer.
    ///
    /// # Panics
    ///
    /// If [`SurrogateType::check`] refuses the surrogate, with the refusal's words;
    /// [`Self::try_new`] returns the refusal instead. At 4.1.0 a refused surrogate was built.
    pub fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        surrogate: SurrogateType,
    ) -> Self {
        Self::try_new(n_inputs, n_neurons, length, seed, surrogate)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Construct a differentiable dense SC layer, or return why its surrogate is refused.
    ///
    /// # Errors
    ///
    /// The surrogate's refusal from [`SurrogateType::check`].
    pub fn try_new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        surrogate: SurrogateType,
    ) -> Result<Self, &'static str> {
        surrogate.check()?;
        Ok(Self {
            layer: crate::layer::DenseLayer::new(n_inputs, n_neurons, length, seed),
            surrogate,
            input_cache: Vec::new(),
            output_cache: Vec::new(),
        })
    }

    /// Forward pass and cache activations for backward pass.
    ///
    /// # Errors
    ///
    /// Those of [`crate::layer::DenseLayer::forward`]; the cache then keeps the previous pass.
    pub fn forward(&mut self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        let out = self.layer.forward(input_values, seed)?;
        self.input_cache = input_values.to_vec();
        self.output_cache = out.clone();
        Ok(out)
    }

    /// Backward pass producing input and weight gradients.
    ///
    /// Neuron `j`'s local gradient is `grad_output[j]` times the surrogate at its cached output
    /// minus 0.5. The gradient of the weight from input `i` is that local gradient times the
    /// cached input `i`, and input `i`'s gradient sums the local gradients times their weights
    /// from `i`.
    ///
    /// # Errors
    ///
    /// When no valid forward pass is cached, or `grad_output` does not hold one value per
    /// neuron.
    pub fn backward(&self, grad_output: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
        if self.input_cache.len() != self.layer.n_inputs {
            return Err("backward() called before a valid forward() input cache.".to_string());
        }
        if self.output_cache.len() != self.layer.n_neurons {
            return Err("backward() called before a valid forward() output cache.".to_string());
        }
        if grad_output.len() != self.layer.n_neurons {
            return Err(format!(
                "Expected grad_output length {}, got {}.",
                self.layer.n_neurons,
                grad_output.len()
            ));
        }

        let mut grad_input = vec![0.0_f64; self.layer.n_inputs];
        let mut grad_weights = vec![vec![0.0_f64; self.layer.n_inputs]; self.layer.n_neurons];

        // Compute surr and local_grad for all neurons
        let local_grads: Vec<f64> = (0..self.layer.n_neurons)
            .map(|j| {
                let surr = self.surrogate.grad((self.output_cache[j] - 0.5) as f32) as f64;
                grad_output[j] * surr
            })
            .collect();

        // Parallel weight gradient computation
        grad_weights
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, row_grad_weights)| {
                let local_grad = local_grads[j];
                let (chunks_gw, remainder_gw) = row_grad_weights.as_chunks_mut::<4>();
                let (chunks_inp, remainder_inp) = self.input_cache.as_chunks::<4>();
                for (cgw, cinp) in chunks_gw.iter_mut().zip(chunks_inp) {
                    cgw[0] = local_grad * cinp[0];
                    cgw[1] = local_grad * cinp[1];
                    cgw[2] = local_grad * cinp[2];
                    cgw[3] = local_grad * cinp[3];
                }
                for (gw, &inp) in remainder_gw.iter_mut().zip(remainder_inp) {
                    *gw = local_grad * inp;
                }
            });

        // Serial input gradient accumulation
        for (j, &local_grad) in local_grads.iter().enumerate() {
            let row_weights = &self.layer.weights[j];
            let (chunks_gi, remainder_gi) = grad_input.as_chunks_mut::<4>();
            let (chunks_w, remainder_w) = row_weights.as_chunks::<4>();
            for (cgi, cw) in chunks_gi.iter_mut().zip(chunks_w) {
                cgi[0] += local_grad * cw[0];
                cgi[1] += local_grad * cw[1];
                cgi[2] += local_grad * cw[2];
                cgi[3] += local_grad * cw[3];
            }
            for (gi, &w) in remainder_gi.iter_mut().zip(remainder_w) {
                *gi += local_grad * w;
            }
        }

        Ok((grad_input, grad_weights))
    }

    /// Apply gradient descent update and clamp weights to `[0, 1]`.
    ///
    /// # Errors
    ///
    /// When the gradients are not one row of `n_inputs` values per neuron; the weights are
    /// then left as they were.
    pub fn try_update_weights(&mut self, weight_grads: &[Vec<f64>], lr: f64) -> Result<(), String> {
        if weight_grads.len() != self.layer.n_neurons {
            return Err(format!(
                "Expected {} grad rows, got {}.",
                self.layer.n_neurons,
                weight_grads.len()
            ));
        }
        if weight_grads
            .iter()
            .any(|row| row.len() != self.layer.n_inputs)
        {
            return Err(format!(
                "Expected each grad row to have length {}.",
                self.layer.n_inputs
            ));
        }

        for (w_row, g_row) in self.layer.weights.iter_mut().zip(weight_grads.iter()) {
            for (w, g) in w_row.iter_mut().zip(g_row.iter()) {
                *w = (*w - lr * *g).clamp(0.0, 1.0);
            }
        }

        self.layer.refresh_packed_weights();
        Ok(())
    }

    /// Apply gradient descent update and clamp weights to `[0, 1]`, as
    /// [`Self::try_update_weights`] does.
    ///
    /// This keeps the signature the method had at 4.1.0, so code written against that release
    /// still compiles; new code should call [`Self::try_update_weights`].
    ///
    /// # Panics
    ///
    /// When the gradients are not one row of `n_inputs` values per neuron, with the words
    /// [`Self::try_update_weights`] returns. At 4.1.0 such gradients were ignored silently.
    pub fn update_weights(&mut self, weight_grads: &[Vec<f64>], lr: f64) {
        if let Err(error) = self.try_update_weights(weight_grads, lr) {
            panic!("{error}");
        }
    }

    /// Clear cached forward tensors.
    pub fn clear_cache(&mut self) {
        self.input_cache.clear();
        self.output_cache.clear();
    }
}
