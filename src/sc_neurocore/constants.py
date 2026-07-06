# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Named physical and model constants for SC-NeuroCore

"""Named physical and model constants for SC-NeuroCore.

This module is the citation ledger for 44 named constants that trace to
textbook references, hardware design choices, or documented conventions.
Current adoption is explicit and tested: 16 maintained source modules import
all 44 named constants from this ledger. See ``docs/api/constants.md`` for the
Python import boundary and the separate Rust parity note.
"""

# ---------------------------------------------------------------------------
# LIF neuron defaults — Gerstner & Kistler, *Spiking Neuron Models*, 2002
# Table 1.1 (normalised units: voltage in [0, 1], time in ms)
# ---------------------------------------------------------------------------
LIF_V_REST = 0.0  # mV (normalised)
LIF_V_RESET = 0.0  # mV (normalised)
LIF_V_THRESHOLD = 1.0  # mV (normalised)
LIF_TAU_MEM = 20.0  # ms — Gerstner & Kistler 2002, Ch. 1
LIF_RESISTANCE = 1.0  # MOhm (normalised, absorbed into current scaling)
LIF_DT = 1.0  # ms — default simulation time step
LIF_NOISE_STD = 0.0  # mV — zero noise by default
LIF_REFRACTORY_PERIOD = 0  # ms — no refractory by default

# Layer-level noise (slightly non-zero for population diversity)
LIF_LAYER_NOISE_STD = 0.02  # mV — small noise for inter-neuron variability

# ---------------------------------------------------------------------------
# Izhikevich neuron — Izhikevich, IEEE TNN 14(6), 2003
# Table 1, "Regular Spiking" parameter set
# ---------------------------------------------------------------------------
IZH_A = 0.02  # 1/ms — recovery time constant
IZH_B = 0.2  # 1/ms — sensitivity of recovery to subthreshold v
IZH_C = -65.0  # mV — post-spike reset potential
IZH_D = 8.0  # mV — post-spike recovery increment
IZH_SPIKE_THRESHOLD = 30.0  # mV — spike-detect peak, not an adaptive membrane threshold

# ---------------------------------------------------------------------------
# Homeostatic threshold adaptation
# Turrigiano, *Cold Spring Harb Perspect Biol* 4:a005736, 2012
# ---------------------------------------------------------------------------
HOMEOSTATIC_TARGET_RATE = 0.1  # spikes/step — target firing probability
HOMEOSTATIC_ADAPTATION_RATE = 0.01  # dimensionless — learning rate for threshold
HOMEOSTATIC_TRACE_DECAY = 0.95  # dimensionless — exponential moving average decay
HOMEOSTATIC_THRESHOLD_FLOOR = 0.1  # mV — prevent threshold collapse to zero
HOMEOSTATIC_THRESHOLD_CEILING_MULT = 10.0  # max threshold = initial * this factor

# ---------------------------------------------------------------------------
# Dendritic XOR neuron
# Koch, *Biophysics of Computation*, 1999, Ch. 12 (shunting inhibition)
# ---------------------------------------------------------------------------
DENDRITIC_THRESHOLD = 0.5  # dimensionless — decision boundary for XOR output

# ---------------------------------------------------------------------------
# Fixed-point hardware parameters — matches sc_lif_neuron.v RTL
# ---------------------------------------------------------------------------
FP_DATA_WIDTH = 16  # bits — Q8.8 format total width
FP_FRACTION = 8  # bits — fractional bits in Q8.8
FP_V_THRESHOLD = 256  # Q8.8 representation of 1.0 (1 << FP_FRACTION)
FP_REFRACTORY_PERIOD = 2  # clock cycles
FP_LFSR_WIDTH = 16  # bits — x^16 + x^14 + x^13 + x^11 + 1
FP_LFSR_SEED = 0xACE1  # non-zero seed for maximal-length LFSR

# ---------------------------------------------------------------------------
# Synapse & plasticity defaults
# Bi & Poo, J. Neurosci. 18(24):10464-10472, 1998, Fig. 1
# ---------------------------------------------------------------------------
STDP_LEARNING_RATE = 0.01  # dimensionless
STDP_WINDOW_SIZE = 5  # timesteps — coincidence detection window
STDP_LTD_RATIO = 0.5  # LTD/LTP asymmetry — Bi & Poo 1998, Fig. 1
SYNAPSE_DEFAULT_LENGTH = 256  # bits — bitstream length for weight encoding
SYNAPSE_DEFAULT_WEIGHT = 0.5  # dimensionless — initial weight probability

# Reward-modulated STDP — Izhikevich, Cerebral Cortex 17(10), 2007
RSTDP_TRACE_DECAY = 0.9  # dimensionless — eligibility trace decay
RSTDP_ANTI_HEBBIAN_SCALE = 0.5  # LTD/LTP asymmetry — Bi & Poo 1998

# ---------------------------------------------------------------------------
# Layer defaults
# ---------------------------------------------------------------------------
LAYER_DEFAULT_LENGTH = 1024  # bits — default bitstream length
LAYER_CONV_LENGTH = 256  # bits — default for convolutional layers
DENSE_LAYER_LENGTH = 2048  # bits — default for dense layers
DENSE_Y_MIN = 0.0  # lower bound of current source output range
DENSE_Y_MAX = 0.1  # upper bound of current source output range

# Recurrent / reservoir — Jaeger, GMD Report 148, 2001
RESERVOIR_FEEDBACK_STRENGTH = 0.5  # dimensionless
RESERVOIR_INPUT_STRENGTH = 0.5  # dimensionless
RESERVOIR_SPECTRAL_RADIUS = 0.9  # echo state property bound — Jaeger 2001

# Memristive crossbar — Prezioso et al., Nature 521:61-64, 2015
MEMRISTIVE_STUCK_RATE = 0.01  # fraction of devices stuck at 0 or 1
MEMRISTIVE_VARIABILITY = 0.05  # sigma of conductance drift noise

# Neuron seed offset (separates neuron RNG streams from synapse streams)
NEURON_SEED_OFFSET = 10_000
