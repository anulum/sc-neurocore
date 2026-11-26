<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Hardware-software co-simulation
-->

# Hardware-Software Co-Simulation

This text was part of the repository README until the README became a short landing page.

The co-sim flow checks bit-exact equivalence between the bit-true fixed-point
Python model (`FixedPointLIFNeuron`) and the hand-written Verilog RTL
`hdl/sc_lif_neuron.v` over 1000 seeded steps. It covers that one neuron and that
stimulus. The Studio co-simulation panel compares a different boundary, the
generated RTL against the generated bit-true C kernel (see
[model readiness](../studio/model-readiness.md)). Neither is a statement about
a model's scientific fidelity.

The checker requires exactly one result row per requested step, with a binary
spike and a signed decimal membrane potential. Missing, empty, truncated, extra
or malformed rows fail with a nonzero exit status. `--steps` must be positive;
use the same step count and seed for generation and checking, and set the
testbench's `NUM_STEPS` to match when changing the default 1000 steps.
The testbench releases reset on a falling edge, applies each input once and
samples the result after the following rising edge.

```bash
# 1. Generate stimuli + expected results (Python golden model)
python scripts/cosim_gen_and_check.py --generate

# 2. Run Verilog simulation (requires Icarus Verilog)
iverilog -o tb_lif hdl/sc_lif_neuron.v hdl/tb_sc_lif_neuron.v
vvp tb_lif

# 3. Compare results
python scripts/cosim_gen_and_check.py --check
```

## Reproducibility

Every GitHub Release includes:

- **wheel + sdist** — Python distribution artifacts (`dist/sc_neurocore-*`)
- **SBOM** — CycloneDX software bill of materials (`sbom.json`)
- **Changelog extract** — release notes from `CHANGELOG.md`

Co-simulation traces are generated deterministically from fixed LFSR seeds.
To reproduce a published benchmark:

```bash
git checkout v4.2.0
pip install -e ".[dev]"
python benchmarks/benchmark_suite.py --markdown > BENCHMARKS.md
```

For Verilog co-sim trace reproduction, see `scripts/cosim_gen_and_check.py`
and the seed constants in `hdl/sc_bitstream_encoder.v`.

## Key Technical Details

- **LFSR**: 16-bit maximal-length, polynomial x^16+x^14+x^13+x^11+1, period 65535
- **Seed strategy**: Input encoders `0xACE1 + i*7`, weight encoders `0xBEEF + i*13`
- **Fixed-point**: Q8.8 (DATA_WIDTH=16, FRACTION=8), signed two's complement
- **Overflow**: Explicit bit-width masking via `_mask()` function
