<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Reservoir Computing with SC Recurrent Layers

Build an Echo State Network (ESN) using stochastic computing recurrent
layers. The reservoir transforms temporal input into a high-dimensional
spike-rate space; a simple linear readout learns the task.

**Prerequisites**: `pip install sc-neurocore scikit-learn matplotlib`

## 1. Why reservoir computing?

Training recurrent SNNs end-to-end is hard — vanishing/exploding
gradients through the spike nonlinearity. Reservoir computing sidesteps
this: the recurrent layer (reservoir) is **fixed** (random weights),
only the readout is trained. This is biologically plausible and maps
naturally to hardware.

SC reservoirs have an additional advantage: the stochastic bitstream
provides built-in noise that prevents the reservoir from collapsing
to fixed-point attractors, improving computational capacity.

## 2. Build the reservoir

```python
import numpy as np
from sc_neurocore import SCRecurrentLayer, VectorizedSCLayer

np.random.seed(42)
N_RES = 200          # reservoir neurons
N_IN = 3             # input channels
L = 256              # bitstream length
SPECTRAL_RADIUS = 0.9

reservoir = SCRecurrentLayer(
    n_inputs=N_IN,
    n_neurons=N_RES,
    length=L,
    spectral_radius=SPECTRAL_RADIUS,
)

print(f"Reservoir: {N_IN} → {N_RES} neurons, L={L}")
print(f"Spectral radius: {SPECTRAL_RADIUS}")
```

The spectral radius controls the reservoir's memory: values close to
1.0 give longer memory but risk instability. SC noise acts as implicit
regularisation, so you can push closer to 1.0 than with float ESNs.

## 3. Generate a Mackey-Glass time series

The Mackey-Glass equation is a standard benchmark for chaotic
time-series prediction (Mackey & Glass, 1977):

```python
def mackey_glass(n_steps, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    """Generate Mackey-Glass chaotic time series."""
    history = np.ones(tau + 1) * 1.2
    series = np.zeros(n_steps)
    for t in range(n_steps):
        x_t = history[-1]
        x_tau = history[-tau - 1] if len(history) > tau else history[0]
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x_t
        x_new = x_t + dt * dx
        history = np.append(history, x_new)
        series[t] = x_new
    return series

mg = mackey_glass(5000)
mg = (mg - mg.min()) / (mg.max() - mg.min())  # normalise to [0, 1]
print(f"Mackey-Glass: {len(mg)} steps, range [{mg.min():.2f}, {mg.max():.2f}]")
```

## 4. Drive the reservoir

Feed the time series into the reservoir and collect output states.
We use delay-embedded inputs (current, t-6, t-12) to give the
reservoir a richer input signal:

```python
WASHOUT = 200  # discard transient
TRAIN_LEN = 3000
TEST_LEN = 1000
DELAYS = [0, 6, 12]

def embed_input(series, t, delays):
    return np.array([series[max(0, t - d)] for d in delays])

states = np.zeros((len(mg), N_RES))
for t in range(len(mg)):
    x = embed_input(mg, t, DELAYS)
    x = np.clip(x, 0.01, 0.99)
    states[t] = reservoir.step(x)

# Split into train/test (after washout)
X_train = states[WASHOUT:WASHOUT + TRAIN_LEN]
y_train = mg[WASHOUT + 1:WASHOUT + TRAIN_LEN + 1]  # 1-step ahead target
X_test = states[WASHOUT + TRAIN_LEN:WASHOUT + TRAIN_LEN + TEST_LEN]
y_test = mg[WASHOUT + TRAIN_LEN + 1:WASHOUT + TRAIN_LEN + TEST_LEN + 1]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

## 5. Train the readout

Ridge regression — the only trained component:

```python
from sklearn.linear_model import Ridge

readout = Ridge(alpha=1e-4)
readout.fit(X_train, y_train)

train_pred = readout.predict(X_train)
test_pred = readout.predict(X_test)

train_nmse = np.mean((train_pred - y_train) ** 2) / np.var(y_train)
test_nmse = np.mean((test_pred - y_test) ** 2) / np.var(y_test)

print(f"Train NMSE: {train_nmse:.4f}")
print(f"Test  NMSE: {test_nmse:.4f}")
```

Expected: NMSE < 0.05 for 1-step ahead prediction with N=200, L=256.

## 6. Plot predictions

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(y_test[:300], label="True", linewidth=0.8)
axes[0].plot(test_pred[:300], label="Predicted", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("Mackey-Glass value")
axes[0].legend()
axes[0].set_title("SC Reservoir: 1-Step Ahead Prediction")

axes[1].plot(np.abs(y_test[:300] - test_pred[:300]), color="tab:red", linewidth=0.5)
axes[1].set_ylabel("Absolute error")
axes[1].set_xlabel("Time step")

plt.tight_layout()
plt.savefig("reservoir_prediction.png", dpi=150)
```

## 7. Reservoir hyperparameter sweep

The key hyperparameters for an SC reservoir:

```python
results = []
for sr in [0.5, 0.7, 0.9, 0.95, 0.99]:
    for length in [64, 128, 256]:
        res = SCRecurrentLayer(n_inputs=N_IN, n_neurons=N_RES,
                               length=length, spectral_radius=sr)
        s = np.zeros((len(mg), N_RES))
        for t in range(len(mg)):
            x = np.clip(embed_input(mg, t, DELAYS), 0.01, 0.99)
            s[t] = res.step(x)

        Xtr = s[WASHOUT:WASHOUT + TRAIN_LEN]
        ytr = mg[WASHOUT + 1:WASHOUT + TRAIN_LEN + 1]
        Xte = s[WASHOUT + TRAIN_LEN:WASHOUT + TRAIN_LEN + TEST_LEN]
        yte = mg[WASHOUT + TRAIN_LEN + 1:WASHOUT + TRAIN_LEN + TEST_LEN + 1]

        rd = Ridge(alpha=1e-4).fit(Xtr, ytr)
        nmse = np.mean((rd.predict(Xte) - yte) ** 2) / np.var(yte)
        results.append((sr, length, nmse))
        print(f"SR={sr:.2f}  L={length:4d}  NMSE={nmse:.4f}")
```

## 8. Memory capacity measurement

Quantify how many past time steps the reservoir can recall:

```python
def memory_capacity(states_matrix, input_series, max_delay=50):
    """Sum of R² for delayed reconstructions (Jaeger, 2001)."""
    mc = 0.0
    n = len(states_matrix)
    for d in range(1, max_delay + 1):
        X = states_matrix[:n - d]
        y = input_series[d:n]
        rd = Ridge(alpha=1e-4).fit(X, y)
        r2 = rd.score(X, y)
        mc += max(0, r2)
    return mc

mc = memory_capacity(
    states[WASHOUT:WASHOUT + TRAIN_LEN],
    mg[WASHOUT:WASHOUT + TRAIN_LEN],
)
print(f"Memory capacity: {mc:.1f} (theoretical max: {N_RES})")
```

## 9. Adding STDP to the reservoir

Standard ESNs use fixed weights. Adding STDP to the reservoir
connections enables online adaptation — the reservoir topology
self-organises to match the input statistics:

```python
from sc_neurocore import StochasticSTDPSynapse

# This is a sketch — full implementation uses SCLearningLayer
# which wraps STDP synapses into the recurrent connectivity.
# The key insight: STDP on reservoir connections improves
# memory capacity for non-stationary inputs.
```

## What you learned

- SC reservoirs combine fixed recurrent weights with a trained linear readout
- Stochastic bitstream noise acts as implicit regularisation
- Spectral radius controls memory length; SC tolerates values closer to 1.0
- Delay embedding enriches the input signal
- NMSE < 0.05 on Mackey-Glass with 200 neurons, L=256
- Memory capacity quantifies the reservoir's temporal information storage
- STDP can adapt reservoir topology online

## Next steps

- Multi-step ahead prediction with iterated readout
- Classification tasks (spoken digit recognition, gesture classification)
- Compare SC reservoir against Brian2 / NEST equivalent
- Deploy reservoir on FPGA with fixed-point weights
