# Tutorial 71: Neuromorphic Symbolic Reasoning

Turing-complete computation using only spikes. SC-NeuroCore implements
arithmetic (ALU), memory (registers), and sorting as spike-based
circuits. These map directly to FPGA without any floating-point units.

## Why Spike-Based Computation

Traditional neuromorphic systems handle pattern recognition (perception).
But real intelligence also needs symbolic operations: arithmetic,
comparison, sorting, counting. SC-NeuroCore bridges this gap with
spike-domain equivalents of digital logic primitives.

| Operation | Digital | Spike-Domain | FPGA Cost |
|-----------|---------|:------------:|-----------|
| Addition | Ripple-carry adder | Spike rate summation | ~16 LUTs |
| Subtraction | Borrow chain | Inhibitory population | ~16 LUTs |
| Comparison | Magnitude comparator | Winner-take-all | ~8 LUTs |
| Register | D flip-flop | Persistent firing loop | ~8 LUTs |
| Sorting | Comparator network | Race-to-threshold | ~64 LUTs |

## Spike ALU

8-bit arithmetic using population-coded spike rates:

```python
from sc_neurocore.symbolic import SpikeALU, SpikeRegister, spike_sort

alu = SpikeALU(n_bits=8)

# Addition
result, carry = alu.add(42, 58)
print(f"42 + 58 = {result}, carry={carry}")  # 100, 0

# Subtraction
result, borrow = alu.sub(100, 30)
print(f"100 - 30 = {result}, borrow={borrow}")  # 70, 0

# Overflow
result, carry = alu.add(200, 100)
print(f"200 + 100 = {result}, carry={carry}")  # 44, 1 (8-bit overflow)
```

### How Spike Addition Works

Each bit is encoded as a spike rate (0% = 0, 100% = 1). The
carry chain propagates through a sequence of LIF neurons where:
- Two input spikes within one timestep → output spike + carry
- One input spike → output spike, no carry
- Zero input spikes → no output

This is functionally identical to a digital ripple-carry adder, but
implemented with spiking neurons.

## Spike Register

Persistent storage using a self-exciting loop:

```python
reg = SpikeRegister(n_bits=8)

reg.write(42)
assert reg.read() == 42

# The value persists without external input — the register
# maintains its state through recurrent spike loops
reg.write(0)  # clear
assert reg.read() == 0
```

On FPGA, each register bit is a LIF neuron with self-excitation
above threshold — it keeps firing until explicitly reset.

## Spike Sort

Sort a list using a race-to-threshold network. Each element is
encoded as a spike latency (smaller values fire first):

```python
unsorted = [5, 3, 8, 1, 4]
sorted_list = spike_sort(unsorted)
print(f"Sorted: {sorted_list}")  # [1, 3, 4, 5, 8]
```

The mechanism: encode each value as a latency (value → time to first
spike). Feed all into a winner-take-all network. The first neuron to
fire is the smallest value, second is next smallest, etc.

Sorting N values takes O(N) timesteps with O(N) neurons — compared
to O(N log N) for comparison-based digital sorting.

## Applications

| Application | Spike Primitive | Why |
|-------------|----------------|-----|
| Spike count accumulation | SpikeALU.add | Count events on-chip |
| Address computation | SpikeALU.add/sub | AER packet routing |
| Priority encoding | spike_sort | Nearest-event selection |
| State machines | SpikeRegister | On-chip control logic |
| Neural Turing Machine | ALU + Register + Attention | General computation |

## References

- Maass (1996). "Networks of spiking neurons: the third generation of
  neural network models." Neural Networks 10(9):1659-1671.
- Eliasmith (2013). "How to Build a Brain: A Neural Architecture for
  Biological Cognition." Oxford University Press.
