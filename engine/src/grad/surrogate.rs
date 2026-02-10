//! # Surrogate Gradient Components
//!
//! Differentiable wrappers around SC forward operators.

#[derive(Clone, Debug)]
pub enum SurrogateType {
    /// Fast Sigmoid: d/dx = 1 / (2k * (1 + k|x|)^2)
    FastSigmoid { k: f32 },
    /// SuperSpike: d/dx = 1 / (1 + k|x|)^2
    SuperSpike { k: f32 },
    /// ArcTan: d/dx = 1 / (1 + (kx)^2)
    ArcTan { k: f32 },
    /// Straight-through estimator.
    StraightThrough,
}

impl SurrogateType {
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
        }
    }
}

/// LIF neuron with surrogate gradient support.
pub struct SurrogateLif {
    pub lif: crate::neuron::FixedPointLif,
    pub surrogate: SurrogateType,
    membrane_trace: Vec<(f32, f32)>,
}

impl SurrogateLif {
    /// Construct a surrogate-enabled fixed-point LIF neuron.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: SurrogateType,
    ) -> Self {
        Self {
            lif: crate::neuron::FixedPointLif::new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
            ),
            surrogate,
            membrane_trace: Vec::new(),
        }
    }

    /// Forward LIF step while caching trace for backward pass.
    pub fn forward(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let (spike, v_out) = self.lif.step(leak_k, gain_k, i_t, noise_in);
        let scale = (1_u32 << self.lif.fraction) as f32;
        let v_norm = (v_out as f32 - self.lif.v_threshold as f32) / scale;
        self.membrane_trace.push((v_norm, spike as f32));
        (spike, v_out)
    }

    /// Backward pass through last cached membrane value.
    pub fn backward(&mut self, grad_output: f32) -> f32 {
        let (v_norm, _spike) = self
            .membrane_trace
            .pop()
            .expect("backward() called without forward()");
        grad_output * self.surrogate.grad(v_norm)
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
    pub layer: crate::layer::DenseLayer,
    pub surrogate: SurrogateType,
    input_cache: Vec<f64>,
    output_cache: Vec<f64>,
}

impl DifferentiableDenseLayer {
    /// Construct a differentiable dense SC layer.
    pub fn new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        surrogate: SurrogateType,
    ) -> Self {
        Self {
            layer: crate::layer::DenseLayer::new(n_inputs, n_neurons, length, seed),
            surrogate,
            input_cache: Vec::new(),
            output_cache: Vec::new(),
        }
    }

    /// Forward pass and cache activations for backward pass.
    pub fn forward(&mut self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        let out = self.layer.forward(input_values, seed)?;
        self.input_cache = input_values.to_vec();
        self.output_cache = out.clone();
        Ok(out)
    }

    /// Backward pass producing input and weight gradients.
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

        for j in 0..self.layer.n_neurons {
            let local_grad = grad_output[j];
            for i in 0..self.layer.n_inputs {
                grad_weights[j][i] = local_grad * self.input_cache[i];
                grad_input[i] += local_grad * self.layer.weights[j][i];
            }
        }

        Ok((grad_input, grad_weights))
    }

    /// Apply gradient descent update and clamp weights to `[0, 1]`.
    pub fn update_weights(&mut self, weight_grads: &[Vec<f64>], lr: f64) {
        if weight_grads.len() != self.layer.n_neurons {
            return;
        }
        if weight_grads
            .iter()
            .any(|row| row.len() != self.layer.n_inputs)
        {
            return;
        }

        for (w_row, g_row) in self.layer.weights.iter_mut().zip(weight_grads.iter()) {
            for (w, g) in w_row.iter_mut().zip(g_row.iter()) {
                *w = (*w - lr * *g).clamp(0.0, 1.0);
            }
        }

        self.layer.refresh_packed_weights();
    }

    /// Clear cached forward tensors.
    pub fn clear_cache(&mut self) {
        self.input_cache.clear();
        self.output_cache.clear();
    }
}
