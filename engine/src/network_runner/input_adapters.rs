// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner input adapters

//! Normalise neuron models with non-standard inputs or outputs to the network
//! runner's `step(f64) -> i32` contract.

use crate::neurons::*;

/// Define an adapter for a model whose step accepts one additional input.
macro_rules! wrap_2arg_f64 {
    ($name:ident, $inner:ty, $v:ident, $extra:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self(<$inner>::new())
            }
            pub fn step(&mut self, current: f64) -> i32 {
                self.0.step(current, $extra)
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Define an adapter for a model whose step accepts two additional inputs.
macro_rules! wrap_3arg {
    ($name:ident, $inner:ty, $v:ident, $e2:expr, $e3:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self(<$inner>::new())
            }
            pub fn step(&mut self, current: f64) -> i32 {
                self.0.step(current, $e2, $e3)
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Define an adapter from the runner's floating input to an integer-input model.
macro_rules! wrap_i32_input {
    ($name:ident, $inner:ty, $v:ident, $ctor:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self($ctor)
            }
            pub fn step(&mut self, current: f64) -> i32 {
                self.0.step(current as i32)
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Define an adapter that thresholds a graded output into a spike value.
macro_rules! wrap_graded {
    ($name:ident, $inner:ty, $v:ident, $threshold:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self(<$inner>::new())
            }
            pub fn step(&mut self, current: f64) -> i32 {
                let out = self.0.step(current);
                if out > $threshold {
                    1
                } else {
                    0
                }
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

wrap_2arg_f64!(WrAlpha, AlphaNeuron, v, 0.0_f64);
wrap_3arg!(WrCOBALIF, COBALIFNeuron, v, 0.0_f64, 0.0_f64);
wrap_2arg_f64!(WrCompteWM, CompteWMNeuron, v, false);
wrap_2arg_f64!(WrTsodyksMarkram, TsodyksMarkramNeuron, v, false);
wrap_2arg_f64!(WrPinskyRinzel, PinskyRinzelNeuron, v_s, 0.0_f64);
wrap_2arg_f64!(WrHayL5, HayL5PyramidalNeuron, v_s, 0.0_f64);
/// Adapter for the canonical one-input TC-LIF (Zhang et al. 2024): the
/// runner drive is the single external current entering the dendrite,
/// and the reported voltage is the somatic potential `u_s`.
#[derive(Clone, Debug)]
pub struct WrTwoCompLIF(pub TwoCompartmentLIFNeuron);
impl WrTwoCompLIF {
    pub fn new() -> Self {
        Self(TwoCompartmentLIFNeuron::new())
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.0.step(current)
    }
    pub fn reset(&mut self) {
        self.0.reset();
    }
    pub fn v(&self) -> f64 {
        self.0.u_s
    }
}
impl Default for WrTwoCompLIF {
    fn default() -> Self {
        Self::new()
    }
}

wrap_i32_input!(WrLoihiCUBA, LoihiCUBANeuron, v, LoihiCUBANeuron::new());
wrap_i32_input!(WrLoihi2, Loihi2Neuron, s1, Loihi2Neuron::new());
wrap_i32_input!(WrSpiNNaker2, SpiNNaker2Neuron, v, SpiNNaker2Neuron::new());
wrap_i32_input!(WrTrueNorth, TrueNorthNeuron, v, TrueNorthNeuron::new(256));
wrap_i32_input!(
    WrIntegerQIF,
    IntegerQIFNeuron,
    v,
    IntegerQIFNeuron::default()
);

/// Preserve the McCulloch-Pitts absolute-inhibition marker on signed transport.
#[derive(Clone, Debug)]
pub struct WrMcCullochPitts(pub McCullochPittsNeuron);

impl WrMcCullochPitts {
    pub fn new() -> Self {
        Self(McCullochPittsNeuron::default())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if current == -1.0 {
            return self.0.try_step(0, true).unwrap_or(0);
        }
        if !current.is_finite()
            || current.fract() != 0.0
            || current < 0.0
            || current > f64::from(i32::MAX)
        {
            return 0;
        }
        self.0.try_step(current as i32, false).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        debug_assert!(self.0.validate().is_ok());
    }

    pub fn v(&self) -> f64 {
        0.0
    }
}

impl Default for WrMcCullochPitts {
    fn default() -> Self {
        Self::new()
    }
}

wrap_graded!(WrSigmoidRate, SigmoidRateNeuron, r, 0.5);
wrap_graded!(WrThresholdLinear, ThresholdLinearRateNeuron, r, 0.5);
wrap_graded!(WrAstrocyte, AstrocyteModel, ca, 0.1);
wrap_graded!(WrInnerHairCell, InnerHairCell, v, 0.0);
wrap_graded!(WrOuterHairCell, OuterHairCell, v, 0.0);
wrap_graded!(WrRodPhotoreceptor, RodPhotoreceptor, v, 0.0);
wrap_graded!(WrConePhotoreceptor, ConePhotoreceptor, v, 0.0);
wrap_graded!(WrTasteReceptor, TasteReceptorCell, v, 0.0);
