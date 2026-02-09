# Getting Started

## Installation

```bash
# Core (CPU only)
pip install -e .

# With development tools
pip install -e ".[dev]"

# With GPU acceleration
pip install -e ".[gpu]"

# Full research stack
pip install -e ".[research]"
```

## Requirements

- Python >= 3.9
- NumPy >= 1.22
- SciPy >= 1.7
- Numba >= 0.56
- Matplotlib >= 3.5

## Running Tests

```bash
# Full suite (826 tests, 100% coverage)
pytest tests/ -v --cov=sc_neurocore --cov-report=term

# Quick smoke test
pytest tests/test_integration.py -v
```

## First Steps

### 1. Create a Bitstream Encoder

```python
from sc_neurocore import BitstreamEncoder, bitstream_to_probability

encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024)
bitstream = encoder.encode(0.7)
recovered = bitstream_to_probability(bitstream)
print(f"Encoded 0.7 -> recovered {recovered:.3f}")
```

### 2. Build a Neuron Layer

```python
from sc_neurocore import VectorizedSCLayer

layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=512)
output = layer.forward([0.3, 0.7, 0.5, 0.2])
print(f"Output firing rates: {output}")
```

### 3. Run the SCPN Stack

```python
from sc_neurocore.scpn import create_full_stack, run_integrated_step, get_global_metrics

stack = create_full_stack()
outputs = run_integrated_step(stack, dt=0.01)
metrics = get_global_metrics(stack)
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```
