# Tutorial 71: Neuromorphic Symbolic Reasoning

Turing-complete computation using only spikes.

```python
from sc_neurocore.symbolic import SpikeALU, SpikeRegister, spike_sort

alu = SpikeALU(n_bits=8)
result, carry = alu.add(42, 58)  # 100
result, borrow = alu.sub(100, 30)  # 70

reg = SpikeRegister(8)
reg.write(42)
assert reg.read() == 42

sorted_list = spike_sort([5, 3, 8, 1, 4])  # [1, 3, 4, 5, 8]
```
