# Neuromorphic Datasets

**Module:** `sc_neurocore.datasets`
**Source:** `src/sc_neurocore/datasets/` — 3 files, 423 LOC
**Status (v3.14.0):** all 5 public symbols wired; 23 tests pass; pure
NumPy I/O — no Rust path needed for the loaders, no synaptic kinetics.
The "Poisson" encoder is actually Bernoulli (§3.1, same wording issue
as `network/stimulus.PoissonInput`).

This page covers the two encoders (`poisson_encode`, `latency_encode`)
and the three event-camera / cochlear loaders (`load_nmnist`,
`load_shd`, `load_dvs_cifar10`), each of which can fall back to
synthetic data when the real archive is not on disk.

---

## 1. Public surface

`sc_neurocore.datasets.__init__` re-exports 5 symbols:

| Symbol | Source file | Role |
|--------|-------------|------|
| `poisson_encode` | `encoding.py` | Per-neuron Bernoulli draw → spike train |
| `latency_encode` | `encoding.py` | Continuous value → first-spike-time |
| `load_nmnist` | `loaders.py` | N-MNIST (Orchard 2015), 34×34 DVS |
| `load_shd` | `loaders.py` | Spiking Heidelberg Digits (Cramer 2022), 700 channels |
| `load_dvs_cifar10` | `loaders.py` | DVS-CIFAR10 (Li 2017), 128×128 DVS |

Each loader accepts `synthetic=True` to bypass disk reads — useful
for unit tests and for CI where the real archives are not stored.

---

## 2. Loaders

### 2.1 `load_nmnist`

```python
def load_nmnist(
    root: str | Path = "data/nmnist",
    train: bool = True,
    dt_ms: float = 1.0,
    T: int = 300,
    synthetic: bool = False,
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
```

Loads the **N-MNIST** dataset:

> Orchard G., Cohen G., Jayawant A., Thakor N. "Converting Static
> Image Datasets to Spiking Neuromorphic Datasets Using Saccades."
> *Front Neurosci* 9:437 (2015).

34×34 ATIS DVS recordings of MNIST digits moved across the sensor by
saccadic eye movements. 10 classes.

Returns `(samples, labels)`:
- `samples`: list of `(N_events, 4)` `float32` arrays with columns
  `[x, y, polarity, timestamp_ms]`
- `labels`: `int64` array of length `len(samples)`

The real-data path expects the directory layout
`root/{Train,Test}/<class_id>/<sample>.bin`. Each `.bin` file is a
sequence of 5-byte events: `[addr_high, addr_low, ts2, ts1, ts0]`,
parsed by the helper `_parse_nmnist_bin` (`loaders.py:150`):

| Bits | Meaning |
|------|---------|
| 0–4 | x coordinate (5-bit, max 31) |
| 5–9 | y coordinate |
| 10 | polarity |
| 16-bit `ts` | timestamp in microseconds |

`dt_ms` scales the parsed timestamps to milliseconds via
`ts_us * (dt_ms / 1000.0)`.

### 2.2 `load_shd`

Loads the **Spiking Heidelberg Digits** dataset:

> Cramer B., Stradmann Y., Schemmel J., Zenke F. "The Heidelberg
> Spiking Data Sets for the Systematic Evaluation of Spiking Neural
> Networks." *IEEE Transactions on Neural Networks and Learning
> Systems* 33(7):2744-2757 (2022).

20-class English/German digit utterances (0–9 in two languages)
spike-encoded through Lauscher's artificial cochlea model. 700 input
channels.

Returns `(samples, labels)`:
- `samples`: list of `(T_per_sample, 700)` `bool` arrays (binned spike
  rasters). `T_per_sample` is `min(ceil(times.max() / (dt_ms/1000)) + 1, T)`.
- `labels`: `int64` array

The real-data path requires `h5py` (declared in extras) and reads
`root/shd_{train,test}.h5`. The H5 layout is the standard SHD release:
`/spikes/times[i]`, `/spikes/units[i]`, `/labels`.

### 2.3 `load_dvs_cifar10`

Loads the **DVS-CIFAR10** dataset:

> Li H., Liu H., Ji X., Li G., Shi L. "CIFAR10-DVS: An Event-Stream
> Dataset for Object Classification." *Front Neurosci* 11:309 (2017).

CIFAR-10 images displayed on a monitor and recorded by a 128×128 DVS
camera. 10 classes.

Returns `(samples, labels)` in the same shape as `load_nmnist`. The
real-data path expects `.npy` files (one per sample) under
`root/{train,test}/<class_id>/`. Each `.npy` must be an array with
columns `[x, y, polarity, timestamp_ms]`. Raw `.aedat` / `.mat`
conversion is left to the caller.

### 2.4 Common contracts

All three loaders:
- Raise `FileNotFoundError` with the dataset's download URL embedded
  in the message when `root` does not exist (`_check_root`,
  `loaders.py:73`).
- Raise `FileNotFoundError` when `root` exists but the train/test
  subdirectory is missing.
- Accept `synthetic=True` to bypass disk reads entirely; the
  synthetic path uses `_synthetic_event_dataset` (event-based
  loaders) or `_synthetic_shd` (binned-raster loader). Both pin
  their RNG to `seed` for reproducibility.

The synthetic generators draw class-conditional rate templates from
`U(0, 0.3)` (event loaders) or `U(0, 0.1)` (SHD), then expand them
through `poisson_encode` to per-sample spike trains. Polarities for
event loaders are `randint(0, 2)`.

---

## 3. Encoders

### 3.1 `poisson_encode` (actually Bernoulli)

```python
def poisson_encode(
    rates: npt.ArrayLike,
    T: int,
    dt_ms: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:  # shape (T, N), bool
```

Returns `(T, N)` boolean spike train: each cell is
`rng.random() < min(rate * dt_ms, 1)`. The function name says
"Poisson" but the per-step sample is **Bernoulli**, not a true
Poisson draw. For low `rate * dt_ms` (< 0.1) the Bernoulli /
Poisson distinction is < 5 % — the two distributions agree to
first order. For high `rate * dt_ms` (> 0.5) Bernoulli under-counts
because it cannot emit more than one spike per timestep; a true
Poisson would.

Same wording issue as
[`PoissonInput` in `network/stimulus.py`](network.md#7-stimulus-sources).
Either rename to `bernoulli_encode` or replace the `< scaled` line
with `rng.poisson(scaled, size)` and accept fractional spike counts.
Tracked as task #26.

### 3.2 `latency_encode` (first-spike-time, FIXED by task #27)

```python
def latency_encode(
    values: npt.ArrayLike,
    T: int,
    tau: float = 5.0,
    strict: bool = True,
) -> np.ndarray:  # shape (T, N), bool
```

Each value `v ∈ [0, 1]` produces exactly one spike at timestep
`int(tau * (1 - v))`, clamped to `[0, T-1]`. Higher value → earlier
spike.

**Input range guard (`strict=True` default):** the function now
raises `ValueError` when any element of `values` is outside
`[0, 1]`. The error message reports the offending min/max and
suggests `strict=False` for the legacy silent-clip behaviour.
This closes the contract gap that the original docstring claimed
but did not enforce.

`strict=False` keeps the v3.14.0 behaviour: `values=1.5` clips to
spike-time 0, `values=-0.5` clips toward T-1.

`tau = 5.0` (default) means the latest possible spike (for v=0) is
at timestep 5. For larger `T`, most timesteps are silent.

`tau = 5.0` (default) means the latest possible spike (for v=0) is at
timestep 5. For larger `T`, most timesteps are silent.

---

## 4. Performance — measured (this workstation)

Hardware: Intel i5-11600K, 32 GB DDR4, Python 3.12.3, NumPy 2.2.6.

### 4.1 Encoder throughput (mean of 20 calls)

| Encoder | N | T | Per-call wall | Spike-cells/s |
|---------|---:|---:|--------------:|--------------:|
| `poisson_encode` | 100 | 300 | 0.37 ms | 81.1 M |
| `poisson_encode` | 1 000 | 300 | 3.33 ms | 90.0 M |
| `poisson_encode` | 10 000 | 300 | 37.38 ms | 80.3 M |
| `latency_encode` | 100 | 300 | 0.06 ms | — |
| `latency_encode` | 1 000 | 300 | 0.05 ms | — |
| `latency_encode` | 10 000 | 300 | 0.63 ms | — |

`poisson_encode` is dominated by the `rng.random((T, N))` call
(uniform draw of `T*N` floats). Throughput is ~80 M spike-cells/s
across all sizes, which matches NumPy's PRNG cost (~10 ns/element).

`latency_encode` is much faster because it draws **no** random
numbers — just one fancy-indexed write per call. The
`(T=300, N=10000)` call still runs in under 1 ms.

### 4.2 Synthetic loader cost

Loading N-MNIST in synthetic mode at `T = 300`, single-threaded:

| `n_samples` | Wall | Total events generated |
|------------:|-----:|-----------------------:|
| 10 | 170.7 ms | 515 780 |
| 100 | 1 739.8 ms | 5 168 651 |
| 500 | 7 566.2 ms | 25 894 833 |

Linear in `n_samples`: ~17 ms per sample, ~50 k events per sample.
The cost is split between `poisson_encode` and the per-event
`np.column_stack` + dtype cast inside `_synthetic_event_dataset`.

### 4.3 No Rust path

These are I/O loaders + per-call NumPy vectorised ops. The hot path
(`rng.random((T, N))`) is already at NumPy/PCG64 speed (~80 M
samples/s). A Rust port would gain little on the encoder side; the
loader side is dominated by file I/O for real datasets and by NumPy
allocation for synthetic data. No Rust path planned.

---

## 5. Pipeline wiring

| Surface | How it's wired | Verifier |
|---------|---------------|----------|
| `from sc_neurocore.datasets import load_nmnist, ...` | `__init__.py:8-9` re-export | `tests/test_datasets.py` |
| Synthetic fallback path | each loader checks `synthetic` first | `TestSyntheticLoaders` |
| Real-data path | `_check_root` raises with download URL | `TestNMNISTRealLoader::test_load_nmnist_real_path`, etc. |
| `_synthetic_event_dataset` calls `poisson_encode` | `loaders.py:56` | covered transitively |
| H5 path imports `h5py` lazily | inside `load_shd` body | works without h5py if `synthetic=True` |

No orphan helpers; `_parse_nmnist_bin` and `_check_root` are private
but reachable from public loaders.

---

## 6. Audit (7-point checklist)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | Pipeline wiring | ✅ PASS | All 5 symbols wired; loaders → encoders → synthetic fallbacks |
| 2 | Multi-angle tests | ✅ PASS | 23 tests across 6 classes (TestCheckRoot, TestSyntheticLoaders, TestEncoding, TestNMNISTRealLoader, TestSHDRealLoader, TestDVSCIFAR10RealLoader); covers shape, reproducibility, file-not-found, real-data parse, encoder rate correlation |
| 3 | Rust path | N/A | I/O + NumPy-vectorised encoders; no compute kernel that would benefit |
| 4 | Benchmarks | ✅ PASS | §4.1 + §4.2 measured this session |
| 5 | Performance docs | ✅ PASS | §4 |
| 6 | Documentation page | ✅ PASS | This page |
| 7 | Rules followed | ⚠️ WARN | SPDX header on every file ✅. **`poisson_encode` is misnamed** — it is Bernoulli, not Poisson (§3.1). **`latency_encode` has an unenforced `[0, 1]` input contract** (§3.2). British English consistent. |

Net: **1 WARN, 0 FAIL.** Both WARN items are naming / contract
issues, not behavioural bugs. Tasks #27 and #28 track them.

---

## 7. Known issues

### 7.1 `poisson_encode` is Bernoulli (task #26)

For low rates this is fine; for high rates it under-counts. Either
rename to `bernoulli_encode` (preferred — the function does not
implement what the name claims) or replace the body with an actual
Poisson draw and accept fractional spike counts (wider behaviour
change).

### 7.2 `latency_encode` silently clips out-of-range input (FIXED by task #27)

The function now raises `ValueError` by default when any value is
outside `[0, 1]`. Pass `strict=False` to keep the legacy silent-clip
behaviour. Regression tests:
`tests/test_datasets.py::TestLatencyEncodeStrict` (5 cases —
above-1 raises, negative raises, strict=False keeps clip, boundary
values 0.0 / 1.0 accepted, interior values correctly ordered).

### 7.3 N-MNIST `_NMNIST_RES` constant is unused on the real path

`loaders.py:24` declares `_NMNIST_RES = 34` but the real-data
parser (`_parse_nmnist_bin`) decodes coordinates straight from the
5-bit address fields without referring to the constant. The constant
is used only on the synthetic path. Either delete it from the
real path, or assert that decoded coordinates fall inside
`[0, _NMNIST_RES)`.

### 7.4 `load_dvs_cifar10` real path requires `.npy` not raw

The docstring says "DVS-CIFAR10 event-camera dataset" but the loader
expects pre-converted `.npy` files, not the raw `.aedat`/`.mat`
released by Li et al. 2017. The error message at line 329-332 makes
this clear, but the docstring at line 271-300 does not. Either
add a one-line "Note: requires `.npy`-converted input" to the
docstring, or ship a `convert_dvs_cifar10_to_npy` utility.

### 7.5 Synthetic SHD differs from real SHD distributionally

`_synthetic_shd` draws class templates from `U(0, 0.1)` independently
per channel, then Poisson-encodes them. Real SHD has rich temporal
structure (cochlear filter banks, formants). The synthetic data
produces correct shapes and labels for unit testing but **trains a
classifier to chance** if used for actual learning. Document this
constraint in the loader docstring.

---

## 8. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_datasets.py -q
# 23 passed in 10.04s (verified 2026-04-17)
```

Coverage breakdown:

- **TestCheckRoot** (2): `_check_root` returns Path on existing dir,
  raises `FileNotFoundError` with URL on missing dir.
- **TestSyntheticLoaders** (7): synthetic-shape correctness for all 3
  loaders, missing-root paths raise even with bad inputs,
  reproducibility across two same-seed calls.
- **TestEncoding** (6): `poisson_encode` shape + rate correlation +
  zero/ones edge cases; `latency_encode` shape + monotonic
  earlier-fire-for-higher-value.
- **TestNMNISTRealLoader** (3): `_parse_nmnist_bin` decodes a
  hand-crafted 5-byte event correctly; `load_nmnist` real path with
  a synthesised directory tree; missing-split raises.
- **TestSHDRealLoader** (2): real-path with synthesised H5 file;
  missing-h5 raises.
- **TestDVSCIFAR10RealLoader** (3): real-path with synthesised npy
  tree; missing-split raises; empty-dir raises.

Not covered:

- **High-rate Poisson distinction** — no test asserts that
  `poisson_encode(rates=1.5, T=10)` saturates at 1 spike/step (the
  Bernoulli ceiling). A test would document the §3.1 issue.
- **Latency input range** — no test asserts behaviour for `value > 1`
  or `value < 0`.
- **Real h5py format** — `TestSHDRealLoader::test_load_shd_real_path`
  uses a synthesised H5 file; the actual SHD release format has not
  been smoke-tested in CI.

---

## 9. References

Datasets (cited by source):

- Orchard G. *et al.* "Converting Static Image Datasets to Spiking
  Neuromorphic Datasets Using Saccades." *Front Neurosci* 9:437
  (2015). N-MNIST.
- Cramer B., Stradmann Y., Schemmel J., Zenke F. "The Heidelberg
  Spiking Data Sets for the Systematic Evaluation of Spiking Neural
  Networks." *IEEE TNNLS* 33(7):2744-2757 (2022). SHD.
- Li H. *et al.* "CIFAR10-DVS: An Event-Stream Dataset for Object
  Classification." *Front Neurosci* 11:309 (2017). DVS-CIFAR10.

Encoders (background):

- Gerstner W., Kistler W. M. *Spiking Neuron Models: Single Neurons,
  Populations, Plasticity*. Cambridge UP (2002). Chapters on rate
  vs latency coding.
- Thorpe S., Fize D., Marlot C. "Speed of processing in the human
  visual system." *Nature* 381:520-522 (1996). The original
  motivation for first-spike-time / latency coding.

Internal:

- Network simulation engine (Poisson stimulus): [`api/network.md`](network.md)
- Monitors & stimulus: [`api/monitor.md`](monitor.md)

---

## 10. Auto-rendered API

::: sc_neurocore.datasets
    options:
      show_root_heading: true
      show_source: true
      members:
        - load_nmnist
        - load_shd
        - load_dvs_cifar10
        - poisson_encode
        - latency_encode
