# Tutorial 52: Formal Equivalence Verification

Prove that your generated Verilog produces identical outputs to the
Python simulation for ALL inputs — not just test vectors.

## What You'll Learn

- What formal equivalence means vs co-simulation
- How the miter circuit works
- How to run a SymbiYosys proof
- How to generate custom equivalence proofs

## Why Formal Verification?

Co-simulation tests N input vectors. Formal verification proves
correctness for ALL 2^(16*4) = 2^64 input combinations per cycle,
across 30 clock cycles. One proof replaces billions of test vectors.

## Pre-Built Proof: LIF Neuron

The repository includes a ready-to-run proof in `hdl/equiv/`:

```bash
# Requires: sby (SymbiYosys), yosys, z3
cd hdl/equiv
sby -f equiv_lif.sby
```

If the proof passes, SymbiYosys prints `PASS` — the DUT (`sc_lif_neuron`)
and reference (`sc_lif_reference`) are functionally identical for all
input sequences up to 30 clock cycles.

## From Python

```python
from sc_neurocore.nas.equiv import check_equivalence

# Generate proof files without running
result = check_equivalence(run=False)
print(result.summary())

# Run the proof (requires SymbiYosys)
result = check_equivalence(run=True, depth=30)
print(result.summary())
```

## How It Works

1. **Reference model**: Minimal Verilog implementing the exact same
   math as the Python FixedPointLIFNeuron (Q8.8 fixed-point).

2. **Miter circuit**: Drives both DUT and reference with identical
   symbolic inputs (SymbiYosys explores ALL values):

   ```
   [symbolic inputs] → DUT     → spike_dut, v_dut
                     → Reference → spike_ref, v_ref

   assert(spike_dut == spike_ref)
   assert(v_dut == v_ref)
   ```

3. **Bounded model checking**: Z3 SMT solver searches for ANY input
   sequence that violates the assertions. If none found up to depth N,
   equivalence is proved.

## Generating Custom Proofs

```python
from sc_neurocore.nas.equiv import generate_miter, generate_sby

# Generate miter for a different module pair
verilog = generate_miter(
    dut_module="sc_lif_neuron",
    ref_module="sc_lif_reference",
    top_name="my_equiv_check",
    data_width=8,
    fraction=4,
)

# Generate SymbiYosys script
sby = generate_sby(
    top_name="my_equiv_check",
    verilog_files=["my_equiv_check.v", "dut.v", "ref.v"],
    depth=50,
)
```

## Proof Scope

The current LIF equivalence proof covers:
- Membrane potential update (leak + input + noise)
- Threshold crossing and spike generation
- Reset after spike
- Q8.8 fixed-point arithmetic

Not covered (future work):
- Refractory period (set to 0 for equivalence)
- Multi-neuron layer interactions
- Synapse weight multiplication
