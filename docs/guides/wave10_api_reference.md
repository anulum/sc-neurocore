<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->

# Wave 10 API Reference — Security, Sovereignty & Endgame

Complete API reference for Wave 10: 10 hardware profiles across 3 platform
classes, 8 compiler features (§60–§67), and the `from_constraints()` universal
profile constructor.

---

## 1. `HardwareProfile.from_constraints()` — Universal Constructor

```python
@classmethod
def from_constraints(
    cls, name: str, *, vendor: str = "Generic", family: str = "Auto",
    platform_class: str = "custom", data_width: int | None = None,
    fraction: int | None = None, max_freq_mhz: int = 0,
    overflow: OverflowMode = "saturate", rounding: RoundingMode = "nearest",
    min_precision_bits: int = 8, max_power_budget_mw: float | None = None,
    notes: str = "",
) -> HardwareProfile
```

**Auto-selection logic**: `data_width` auto-derived from `max_power_budget_mw`:
- <10 mW → 8-bit | 10–100 mW → 16-bit | >100 mW / None → 16-bit

`fraction` auto-derived: `max(min_precision_bits, data_width // 2)`, clamped to `data_width - 1`.

**Side effect**: Auto-registers in global `_PROFILES` registry.

```python
from sc_neurocore.compiler.hardware_profiles import HardwareProfile, get_profile

p = HardwareProfile.from_constraints("my_chip", vendor="Lab", max_power_budget_mw=5)
assert p.data_width == 8
assert get_profile("my_chip") is p  # Auto-registered
```

---

## 2. Hardware Profiles — 3 New Platform Classes (10 profiles)

### Magnonic (3 profiles)

Skyrmion/spin-wave reservoir computing. Sub-fJ switching energy.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `tum_skyrmion` | TU Munich | SkyANN-v1 | Q4.4 |
| `kaist_spinwave` | KAIST | SpinWave-RC | Q4.4 |
| `imec_mtj_reservoir` | imec | MTJ-Reservoir | Q4.4 |

### Organic Bioelectronic (2 profiles)

OECT/PEDOT:PSS wet computing for in-vivo bioelectronic interfaces.

| Profile | Vendor | Family | Width |
|---------|--------|--------|------:|
| `cambridge_oect` | Cambridge | OECT-Synapse | Q4.4 |
| `linkoping_organic` | Linköping | Organic-NN | Q4.4 |

### RISC-V Sovereign AI (5 profiles)

Open-ISA processors. No ITAR/EAR restrictions. Data-sovereign deployment.

| Profile | Vendor | Family | Width | Freq |
|---------|--------|--------|------:|-----:|
| `sifive_x280_ai` | SiFive | X280-AI | Q8.8 | 2 GHz |
| `esperanto_et_soc` | Esperanto | ET-SoC-1 | Q4.4 | 1 GHz |
| `ventana_veyron_ai` | Ventana | Veyron-V2 | Q8.8 | 3.6 GHz |
| `tenstorrent_ascalon` | Tenstorrent | Ascalon | Q8.8 | 4 GHz |
| `andes_ax45mpv` | Andes | AX45MPV | Q8.8 | 1.5 GHz |

---

## 3. Compiler Features §60–§67

### §60. `lint_hardware_trojans(equations, *, check_dormant=True, check_payload=True) → TrojanLintResult`

Detect suspicious combinational paths (dormant triggers, payload injection).

| Return field | Type | Description |
|---|---|---|
| `suspicious_paths` | `list[str]` | Flagged paths with descriptions |
| `risk_level` | `str` | `LOW` / `MEDIUM` / `HIGH` |
| `total_checks` | `int` | Number of checks performed |

```python
from sc_neurocore.compiler.advanced_features import lint_hardware_trojans
r = lint_hardware_trojans({"v": "-(v)/tau + I"})
assert r.risk_level == "LOW"
```

---

### §61. `generate_sbom(module_name, profile_name, *, dependencies=None, sbom_format="CycloneDX") → SBOM`

Generate CycloneDX/SPDX Bill of Materials. **Required by EU CRA (2026).**

| Return field | Type | Description |
|---|---|---|
| `format` | `str` | `CycloneDX` or `SPDX` |
| `components` | `list[dict]` | Component inventory |
| `total_components` | `int` | Total count |

```python
from sc_neurocore.compiler.advanced_features import generate_sbom
s = generate_sbom("sc_lif", "artix7", dependencies={"numpy": "1.26.0"})
assert s.total_components >= 4
```

---

### §62. `generate_hil_calibration(module_name, equations, *, parameters=None) → HILCalibration`

Generate hardware-in-the-loop calibration protocol for drift compensation.

| Return field | Type | Description |
|---|---|---|
| `protocol_steps` | `list[str]` | Step-by-step procedure |
| `num_parameters` | `int` | Number of sweep parameters |
| `sweep_ranges` | `dict[str, tuple]` | Parameter → (min, max) |

```python
from sc_neurocore.compiler.advanced_features import generate_hil_calibration
cal = generate_hil_calibration("sc_lif", {"v": "expr"}, parameters={"tau": (5.0, 50.0)})
assert cal.num_parameters == 1
```

---

### §63. `generate_digital_twin(module_name, equations, profile_name) → str`

Generate Python class that mirrors deployed hardware state in real-time.
Contains `__init__()`, `step(inputs)`, and `compare(hw_state)` methods.

```python
from sc_neurocore.compiler.advanced_features import generate_digital_twin
code = generate_digital_twin("sc_lif", {"v": "-(v)/tau"}, "artix7")
assert "class ScLifTwin:" in code
assert "def step" in code
assert "def compare" in code
```

---

### §64. `map_ucie_protocol(blocks, *, lane_bandwidth_gbps=32.0, protocol_version="UCIe 2.0") → UCIeMapping`

Map neuron blocks to UCIe die-to-die protocol lanes for chiplet architectures.

| Return field | Type | Description |
|---|---|---|
| `lanes` | `dict[str, int]` | Block → number of UCIe lanes |
| `protocol_version` | `str` | UCIe version |
| `total_bandwidth_gbps` | `float` | Aggregate bandwidth |

```python
from sc_neurocore.compiler.advanced_features import map_ucie_protocol
m = map_ucie_protocol({"cortex": 256, "motor": 128})
assert m.total_bandwidth_gbps > 0
```

---

### §65. `schedule_seu_scrubbing(config_bits, *, orbit_altitude_km=400, shielding_mm_al=3.0, strategy="hybrid") → ScrubSchedule`

Generate scrubbing schedule for space-grade configuration memory using
orbital altitude and shielding to estimate SEU rate.

| Return field | Type | Description |
|---|---|---|
| `interval_ms` | `float` | Scrub interval in milliseconds |
| `strategy` | `str` | `blind` or `hybrid` |
| `frames_per_cycle` | `int` | Configuration frames per scrub cycle |
| `expected_seu_rate` | `float` | Expected upsets/day |

```python
from sc_neurocore.compiler.advanced_features import schedule_seu_scrubbing
s = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=800)
assert s.interval_ms > 0
```

---

### §66. `obfuscate_ip(module_name, equations, *, key_length=64, methods=None) → ObfuscationResult`

Apply logic locking + structural transformation for IP protection.

| Return field | Type | Description |
|---|---|---|
| `techniques_applied` | `list[str]` | Applied obfuscation methods |
| `key_bits` | `int` | Obfuscation key length |
| `original_signals` | `int` | Pre-obfuscation signal count |
| `obfuscated_signals` | `int` | Post-obfuscation signal count |

```python
from sc_neurocore.compiler.advanced_features import obfuscate_ip
r = obfuscate_ip("sc_lif", {"v": "a + b"}, key_length=128)
assert r.obfuscated_signals > r.original_signals
```

---

### §67. `embed_watermark(module_name, equations, *, owner_id="SC-NeuroCore", method="constraint_based") → WatermarkResult`

Embed verifiable watermark into compiled netlist. Survives synthesis optimisation.

| Return field | Type | Description |
|---|---|---|
| `watermark_hash` | `str` | 16-char SHA-256 truncated hash |
| `embedding_method` | `str` | Method used |
| `overhead_percent` | `float` | Logic overhead percentage |
| `verifiable` | `bool` | Always `True` |

```python
from sc_neurocore.compiler.advanced_features import embed_watermark
wm = embed_watermark("sc_lif", {"v": "a"}, owner_id="MyLab")
assert wm.verifiable and len(wm.watermark_hash) == 16
```

---

## 4. Test Suite — test_wave10_features.py (281 lines, 32 tests)

| Test Class | Tests | Coverage |
|------------|------:|----------|
| `TestMagnonicPlatforms` | 3 | All magnonic profiles |
| `TestOrganicBioelectronic` | 2 | All OECT profiles |
| `TestRiscVSovereign` | 5 | All RISC-V profiles |
| `TestTotalCoverage` | 2 | ≥175 profiles, ≥31 classes |
| `TestFromConstraints` | 3 | Basic, low-power, explicit-width |
| `TestTrojanLint` | 2 | Clean + conditional trigger |
| `TestSBOM` | 2 | Basic + with dependencies |
| `TestHILCalibration` | 2 | Basic + custom ranges |
| `TestDigitalTwin` | 1 | Code generation |
| `TestUCIeMapper` | 1 | Lane assignment |
| `TestSEUScrubber` | 2 | LEO + higher orbit comparison |
| `TestIPObfuscation` | 2 | Default + custom key |
| `TestWatermark` | 3 | Basic, deterministic, owner diff |
| `TestWave10Integration` | 2 | E2E pipeline + space pipeline |
| **Total** | **32** | |

```bash
python -m pytest tests/test_wave10_features.py -v
```

---

## Further Reading

- [Wave 9 API Reference](wave9_api_reference.md) — §52–§59 + 4 platform classes
- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features
- [Frontier Platforms Guide](frontier_platforms.md) — 31 platform classes
- [Platform Extensibility Guide](platform_extensibility.md) — 3 extensibility mechanisms
