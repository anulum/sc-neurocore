# Symbolic Reasoning — Spike-Based Logic

Turing-complete symbolic computation using only spiking neurons. LIF neurons configured as logic gates compose into half-adder, full-adder, ALU, comparator, and sorting network.

## Architecture

Each gate maps to a specific LIF configuration:

| Gate | LIF Threshold | Weights | Behavior |
|------|--------------|---------|----------|
| AND | 2 | [1, 1] | Both inputs must fire |
| OR | 1 | [1, 1] | Either input fires |
| NOT | 0 | [-1] | Inhibitory inversion |
| NAND | 0 | [-1, -1], bias=2 | AND + NOT |
| XOR | 1 | [1, 1], inhibit_if_both | Odd-parity |

The `SpikeALU` composes these gates into a ripple-carry adder for N-bit integer arithmetic. Addition: sum = a XOR b XOR carry, carry = (a AND b) OR (carry AND (a XOR b)). Subtraction via two's complement.

## Components

- **`SpikeGate`** — Configurable spike logic gate. Supports AND, OR, NOT, NAND, XOR. The `lif_config` property returns LIF neuron parameters that reproduce the gate behavior in simulation.
- **`SpikeRegister`** — N-bit register using SR latch pairs (mutual inhibition). Read/write integer values or raw bit arrays.
- **`SpikeALU`** — N-bit Arithmetic Logic Unit. Operations: `add`, `sub`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `compare`, `shift_left`, `shift_right`.
- **`spike_sort`** — Sort integers using a bubble-sort comparison network built from SpikeALU.compare.

## Usage

```python
from sc_neurocore.symbolic import SpikeGate, SpikeRegister, SpikeALU, spike_sort

# Logic gates
xor = SpikeGate("XOR")
assert xor(1, 0) == 1

# 8-bit ALU
alu = SpikeALU(8)
result, carry = alu.add(100, 50)   # 150, False
result, carry = alu.add(200, 100)  # 44, True (overflow)

# 16-bit ALU for larger values
alu16 = SpikeALU(16)
result, _ = alu16.add(30000, 30000)  # 60000

# Spike-based sorting
sorted_vals = spike_sort([3, 1, 4, 1, 5, 9, 2, 6])
```

**Reference:** Plana et al. 2022 — Spike-based logic gates on SpiNNaker.

See [Tutorial 71: Symbolic Reasoning](../tutorials/71_symbolic_reasoning.md).

::: sc_neurocore.symbolic
    options:
      show_root_heading: true
