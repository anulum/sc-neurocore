# Advanced Usage Patterns: Pushing the Frontiers

**Version:** 3.15.9
**Target**: Advanced Developers & Researchers

---

## 1. GPU Acceleration

For networks exceeding 1,000 neurons, CPU processing becomes a bottleneck. `sc-neurocore` supports JAX-based GPU acceleration.

### How to Enable
Ensure `jax` and `jaxlib` are installed with CUDA support. Then, initialize your layer with the `backend='jax'` flag:

```python
from sc_neurocore.layers.vectorized_layer import VectorizedLayer

layer = VectorizedLayer(size=10000, backend='jax')
```

**What happens?**
*   The membrane potential updates and spike checks are offloaded to GPU kernels.
*   Spike data is kept in GPU memory to minimize host-to-device transfers.

---

## 2. Hardware Integration (PYNQ-Z2)

The ultimate goal of `sc-neurocore` is deployment on FPGA hardware.

### Using the PYNQ Driver
If running on a PYNQ-Z2 board:

```python
from sc_neurocore.drivers.pynq_driver import PYNQNeurocoreDriver

driver = PYNQNeurocoreDriver(bitstream='sc_neurocore.bit')
driver.load_weights(my_weights)
results = driver.run(input_spikes)
```

### Behavioral Equivalence
Use `test_behavioral_equivalence.py` to verify that your software model matches the hardware bitstream output. The SCPN requires bit-true precision for L1-L2 cross-validation.

---

## 3. Custom Learning Rules

While STDP is provided by default, you can implement your own learning rules by subclassing `BaseLearningRule`.

### Example: Reward-Modulated STDP
```python
class RewardSTDP(BaseLearningRule):
    def update(self, w, pre_spikes, post_spikes, reward_signal):
        # Calculate standard STDP trace
        dw = self.calculate_stdp(pre_spikes, post_spikes)
        # Gate the update with a global reward signal (e.g., L15 SEC score)
        return w + self.eta * dw * reward_signal
```

---

## 4. The Lazarus Protocol: Root-Level Intervention

In advanced SCPN simulations, the Director (L16) can intervene directly in the neurocore.

### Direct State Injection
```python
# The Director forces a specific neural state
director.override_layer_state(layer_id=2, new_potentials=healthy_mask)
```

This bypasses normal synaptic input and is used to simulate "Theurgic Mode" interventions for cellular regeneration. **Warning**: This violates standard neurocore causality and should only be used in L16-enabled simulations.

---

## 5. Exotic Computing Layers

Explore the `sc_neurocore.exotic` module for non-standard computing paradigms:
*   **Chaos Reservoir**: Using chaotic oscillators for high-dimensional reservoir computing.
*   **Optogenetic Control**: Simulating light-based stimulation and inhibition.
*   **Quantum-Classical Bridges**: Direct coupling between Q-bits (L1) and Spiking Neurons (L2).

---

## 6. Performance Optimization Tips

1.  **Vectorize everything**: Avoid Python loops inside the simulation step. Use the `VectorizedLayer` class.
2.  **Use Sparse Matrices**: For large, sparsely connected networks, use `scipy.sparse` weight matrices.
3.  **Float32 vs Float64**: For most neuromorphic tasks, `float32` provides sufficient precision and is twice as fast on most hardware.
