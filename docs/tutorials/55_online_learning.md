# Tutorial 55: O(1) Memory Online Learning

Train SNNs on arbitrarily long sequences without running out of memory.
Standard BPTT unrolls all timesteps, storing every intermediate
activation. E-prop and online learning maintain only local eligibility
traces — constant memory regardless of sequence length.

## The Problem

BPTT stores activations for all T timesteps before computing gradients:

- T=25 (typical SNN training): manageable, ~100 MB for a small network
- T=1,000 (1 second at 1 ms): starts filling GPU memory
- T=100,000 (real-time BCI, 100 seconds): impossible with BPTT

Real-world spiking applications (BCI, robotics, always-on sensors)
require continuous learning on potentially infinite streams. BPTT
cannot do this. Online learning can.

## E-prop: Eligibility Propagation

E-prop (Bellec et al. 2020) maintains a per-synapse eligibility trace
that summarises the effect of past inputs on current weights. The trace
decays exponentially, so only recent history matters — O(1) memory.

```python
from sc_neurocore.online_learning import EpropTrainer

trainer = EpropTrainer(
    n_inputs=64,
    n_neurons=128,
    n_outputs=10,
    tau_mem=20.0,     # membrane time constant (ms)
    tau_trace=20.0,   # eligibility trace decay (ms)
    lr=0.001,
)

# Train on a 1000-step sequence — O(1) memory per step
import numpy as np
inputs = np.random.rand(1000, 64).astype(np.float32)
targets = np.zeros((1000, 10), dtype=np.float32)
targets[:, 3] = 1.0  # target class 3

loss = trainer.train_sequence(inputs, targets)
print(f"Sequence loss: {loss:.4f}")

# Memory usage is constant regardless of sequence length
print(f"Memory per step: {trainer.memory_per_step} parameters")
# This doesn't change if you train on 100 steps or 100,000 steps
```

### How E-prop Works

At each timestep t, three quantities are updated locally:

1. **Eligibility trace** e(t): tracks how each synapse contributed to
   the neuron's membrane potential. Decays with time constant tau_trace.

   ```
   e(t) = alpha * e(t-1) + spike_pre(t) * pseudo_derivative(t)
   ```

2. **Learning signal** L(t): error signal broadcast from the output
   layer (analogous to backpropagated error, but local in time).

3. **Weight update**: `dW = lr * L(t) * e(t)` — applied immediately,
   no accumulation across time.

The key insight: the eligibility trace summarises the credit assignment
problem locally. No need to store the full unrolled computation graph.

## OnlineTrainer: Feedforward Stacks

For feedforward architectures without recurrence:

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
total_loss = 0.0
for t in range(1000):
    x_t = np.random.rand(784).astype(np.float32)
    target_t = np.zeros(10, dtype=np.float32)
    target_t[t % 10] = 1.0

    result = trainer.step(x_t, target=target_t)
    total_loss += result["loss"]

    if (t + 1) % 100 == 0:
        print(f"t={t+1}, avg loss={total_loss / (t+1):.4f}")
```

## Comparison: BPTT vs E-prop vs Online

| Property | BPTT | E-prop | OnlineTrainer |
|----------|------|--------|---------------|
| Memory | O(T × N) | O(N) per step | O(N) per step |
| Gradient quality | Exact | Approximate (3-factor) | Approximate (local) |
| Max sequence | ~1000 steps (GPU) | Unlimited | Unlimited |
| Recurrence | Yes | Yes | Feedforward only |
| Hardware target | GPU only | CPU, FPGA, neuromorphic | CPU, FPGA |
| Biological plausibility | No | Yes (3-factor rule) | Partial |

## When to Use Which

**Use BPTT** (standard `train_epoch()`) when:
- Sequence length ≤ 100 timesteps
- GPU memory is available
- Maximum accuracy matters more than deployment constraints

**Use E-prop** when:
- Sequence length > 1000 timesteps or unbounded
- Deploying on neuromorphic hardware (Loihi, BrainScaleS)
- Biological plausibility is a requirement
- Training on streaming data (BCI, robotics)

**Use OnlineTrainer** when:
- Feedforward architecture (no recurrence)
- Continuous online adaptation needed
- Edge deployment with limited memory

## Accuracy Gap: Honest Assessment

E-prop is an approximation. On benchmarks requiring long-range credit
assignment (sequential MNIST, where digit class depends on all 784
pixels presented sequentially), BPTT achieves ~97% while E-prop
typically reaches ~93-95%.

For tasks with temporal locality (most real-world signals — speech,
motor control, sensor processing), the gap narrows to <1%.

The trade-off is memory vs accuracy: E-prop runs on 1 MB of memory
where BPTT would need 100+ MB. For deployment on neuromorphic chips
or FPGAs, E-prop is the only viable option.

## Hardware Deployment

E-prop's local learning rule maps directly to on-chip learning on
neuromorphic hardware:

```python
# Export trained weights for FPGA deployment
weights = trainer.export_weights()

# The learning rule itself can run on-chip:
# each synapse maintains its own trace register
# weight updates happen locally, no global backprop bus
```

In the Studio:

1. Train with E-prop in the Training Monitor
2. Export weights via the Code Generator
3. Compile the network to SystemVerilog
4. Deploy with on-chip learning enabled

## References

- Bellec et al. (2020). "A solution to the learning dilemma for recurrent
  networks of spiking neurons." Nature Communications 11:3625.
- Zenke & Neftci (2021). "Brain-Inspired Learning on Neuromorphic
  Substrates." Proceedings of the IEEE 109(5):935-950.
- Kaiser et al. (2020). "Synaptic Plasticity Dynamics for Deep Continuous
  Local Learning (DECOLLE)." Frontiers in Neuroscience 14:424.
