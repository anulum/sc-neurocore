# Tutorial 55: O(1) Memory Online Learning

Train SNNs on long sequences without running out of memory.

## The Problem

BPTT unrolls T timesteps, storing all activations: O(T) memory.
For T=1000 (1 second at 1ms resolution), this fills GPU memory fast.
For T=100,000 (real-time BCI), it's impossible.

## E-prop: Eligibility Propagation

O(1) memory, biologically plausible, works on recurrent SNNs:

```python
from sc_neurocore.online_learning import EpropTrainer

trainer = EpropTrainer(
    n_inputs=64,
    n_neurons=128,
    n_outputs=10,
    tau_mem=20.0,     # membrane time constant (ms)
    tau_trace=20.0,   # eligibility trace decay
    lr=0.001,
)

# Train on a 1000-step sequence — O(1) memory!
loss = trainer.train_sequence(inputs, targets)

# Inference
outputs = trainer.predict_sequence(test_inputs)

# Memory usage doesn't depend on sequence length
print(f"Memory per step: {trainer.memory_per_step} parameters")
```

## OnlineTrainer: Feedforward Stacks

For feedforward architectures (no recurrence):

```python
from sc_neurocore.online_learning import OnlineTrainer

trainer = OnlineTrainer(
    layer_sizes=[784, 256, 128, 10],  # MNIST architecture
    tau_mem=20.0,
    threshold=1.0,
    lr=0.001,
)

# Train online, one timestep at a time
trainer.reset()
for t in range(T):
    result = trainer.step(inputs[t], target=targets[t])
    print(f"t={t}, loss={result['loss']:.4f}")
```

## BPTT vs E-prop Comparison

| Property | BPTT | E-prop |
|----------|------|--------|
| Memory | O(T) | O(1) per step |
| Accuracy | Exact gradients | Approximate |
| Max sequence | ~1000 steps (GPU) | Unlimited |
| Hardware | GPU only | CPU, FPGA, neuromorphic |
| Biological plausibility | No | Yes |

## Caveats

E-prop is an approximation. Accuracy may be 1-5% lower than BPTT on
benchmarks that require long-range credit assignment. For tasks where
temporal locality is strong (most real-world signals), the gap is small.
