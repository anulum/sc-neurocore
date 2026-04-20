# Bioware Interface

Biological-hardware interface for cerebral organoids and multi-electrode
array (MEA) systems. Bridges wet-lab experiments with in-silico SC
simulations — detection, transcoding, plasticity, optogenetic feedback,
and a closed-loop session layer.

## Quick Start

```python
from sc_neurocore.bioware.bioware import (
    BioHybridSession,
    BioHybridFrameResult,
    MEAConfig,
    MEALayout,
    SpikeDetector,
    MEAToAERTranscoder,
    AERToSCConverter,
    SCToOptoEncoder,
    CultureHealth,
    HomeostaticPlasticity,
    SpikeSorter,
    mea_fitness_hook,
)
```

---

## Closed-loop session — `BioHybridSession`

`BioHybridSession` orchestrates the wet-lab pipeline:

```
MEA recording → spike detection → AER transcoding → SC processing
              → optogenetic feedback → plasticity update
```

Each call to :meth:`BioHybridSession.process_frame(voltage, …)`
returns a :class:`BioHybridFrameResult` packet summarising the round.

### `BioHybridFrameResult`

Typed dataclass with a dual access surface: attribute access
(``result.round``) **and** read-only mapping access
(``result["round"]``, ``"latency_us" in result``, ``dict(result)``).
Fields:

| Field               | Type                     | Meaning                              |
| ------------------- | ------------------------ | ------------------------------------ |
| `round`             | `int`                    | 1-based round counter                |
| `num_spikes`        | `int`                    | detected spikes this frame           |
| `num_aer_events`    | `int`                    | AER events generated                 |
| `num_bitstreams`    | `int`                    | SC bitstreams emitted                |
| `num_opto_pulses`   | `int`                    | optogenetic pulses produced          |
| `latency_us`        | `float`                  | wall-clock round-trip (microseconds) |
| `health`            | `dict[str, Any]`         | CultureHealth snapshot               |
| `spikes`            | `list[DetectedSpike]`    | individual spike events              |
| `aer_events`        | `list[AEREvent]`         | transcoded address-event packets     |
| `bitstreams`        | `dict[int, np.ndarray]`  | per-channel SC bitstreams            |
| `opto_pulses`       | `list[OptogeneticPulse]` | scheduled opto pulses                |

Both access styles point at the same objects (``result["health"] is
result.health`` holds). Unknown keys raise ``KeyError``; private /
dunder names are hidden from the mapping view.

```python
session = BioHybridSession(mea_config=..., detector=..., ...)
result = session.process_frame(voltage_data)

# Attribute access (recommended for new code):
print(result.round, result.latency_us)

# Dict access (legacy-compatible):
if "opto_pulses" in result:
    for p in result["opto_pulses"]:
        ...
```

---

## Homeostatic plasticity — Q8.8 threshold controller

`HomeostaticPlasticity.update_threshold(current_q88, observed_rate_hz,
dt_ms)` is a proportional negative-feedback controller that drives a
neuron's firing rate toward `target_rate_hz`:

```
error    = observed_rate_hz − target_rate_hz
alpha    = dt_ms / tau_homeo_ms
delta_q88 = int(alpha · error · 256)
new_q88  = clamp(current_q88 + delta_q88, min_threshold_q88, max_threshold_q88)
```

The Q8.8 scaling factor of 256 means a 1 Hz error integrated over one
full time-constant shifts the threshold by exactly 1.0 threshold unit
(256 Q8.8 steps). Output is clamped into
`[min_threshold_q88, max_threshold_q88]`.

```python
hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
new_threshold = hp.update_threshold(current_q88=256, observed_rate_hz=50.0, dt_ms=1000.0)
# firing too fast → threshold raised
```

---

## Spike sorting — PCA + KMeans

`SpikeSorter` clusters detected spikes by **waveform shape** (not by
scalar amplitude). Requires the ``scikit-learn`` optional dependency at
``fit`` time when there are enough waveforms to cluster;
amplitude-only spike lists (``waveform is None``) silently no-op so
downstream code still runs without ``sklearn`` installed.

```python
sorter = SpikeSorter(num_units=3)
sorter.fit(spikes_with_waveforms)
labelled = sorter.assign(spikes_with_waveforms)
# each DetectedSpike now carries its unit_id
```

Fewer than `num_units` waveforms available → `fit` is a no-op,
`assign` returns the input unchanged (all `unit_id = 0`).

---

## Evo Substrate bridge — `mea_fitness_hook`

Plugs :mod:`sc_neurocore.evo_substrate`'s ``ReplicationEngine`` into
MEA response dynamics:

```python
from sc_neurocore.bioware.bioware import mea_fitness_hook
from sc_neurocore.evo_substrate.evo_substrate import ReplicationEngine

engine = ReplicationEngine(metrics_fn=mea_fitness_hook)
```

The hook returns the standard fitness triple the engine scores:

| Key           | Meaning                                                         |
| ------------- | --------------------------------------------------------------- |
| `accuracy`    | `1 − min(1, abs(mean_rate − target_rate) / target_rate)` clamped to `[0.1, 0.99]` |
| `energy_mw`   | `0.5 mW × num_spikes` (proxy cost)                              |
| `latency_ms`  | constant round-trip time budget marker                          |

Empty or zero-target inputs return the floor `{0.1, 0.0, 0.0}` — never
raises. Uses `DetectedSpike.channel` (not `channel_id`) to group spikes
by source.

---

## Reference

- Source: `src/sc_neurocore/bioware/bioware.py`.
- Tests: `tests/test_bioware/test_bioware.py`.
- Demo: `examples/14_bioware_closed_loop_demo.py` (end-to-end 100-frame
  closed-loop run with SpikeSorter + HomeostaticPlasticity +
  ArcaneZenith cognitive core).

::: sc_neurocore.bioware.bioware
    options:
      show_root_heading: true
