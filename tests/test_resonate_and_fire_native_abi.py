# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Direct resonate-and-fire Go and Mojo ABI contracts

"""Exercise width, pointer, overlap, status, atomicity, and build boundaries."""

from __future__ import annotations

import importlib.util
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import resonate_and_fire as public_backend

_ROOT = Path(__file__).resolve().parents[1]
_BACKENDS = ("go", "mojo")
_REGION_NAMES = (
    "current",
    "x_trace",
    "y_trace",
    "spikes",
    "x_final",
    "y_final",
    "spike_count",
)
_REGION_LENGTHS = {
    "current": 2,
    "x_trace": 2,
    "y_trace": 2,
    "spikes": 2,
    "x_final": 1,
    "y_final": 1,
    "spike_count": 1,
}
_REGION_PAIRS = tuple(itertools.combinations(_REGION_NAMES, 2))
_PARTIAL_REGION_PAIRS = tuple(
    pair for pair in _REGION_PAIRS if max(_REGION_LENGTHS[pair[0]], _REGION_LENGTHS[pair[1]]) > 1
)
_CONFIG = (0.13, -0.27, -0.8, 7.5, 0.9, 0.006)
_LATE_FAILURE_CONFIG = (0.25, -0.5, 1_000.0, 1.0, 1.0e300, 1.0)
_SENTINEL = -999.0


def _module(backend: str) -> Any:
    module = __import__(
        f"sc_neurocore.accel.{backend}.resonate_and_fire",
        fromlist=["resonate_and_fire"],
    )
    marker = f"_HAS_{backend.upper()}_RESONATE_AND_FIRE"
    assert bool(getattr(module, marker))
    assert module._lib is not None
    return module


def _separate_buffers(
    drive: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    steps = int(drive.size)
    return {
        "current": np.ascontiguousarray(drive, dtype=np.float64),
        "x_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "y_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "spikes": np.full(steps, _SENTINEL, dtype=np.float64),
        "x_final": np.full(1, _SENTINEL, dtype=np.float64),
        "y_final": np.full(1, _SENTINEL, dtype=np.float64),
        "spike_count": np.full(1, _SENTINEL, dtype=np.float64),
    }


def _pointers(buffers: dict[str, npt.NDArray[np.float64]]) -> dict[str, int]:
    return {name: int(buffer.ctypes.data) for name, buffer in buffers.items()}


def _call(
    library: Any,
    steps: int,
    pointers: dict[str, int],
    configuration: tuple[float, ...] = _CONFIG,
) -> int:
    return int(
        library.resonate_and_fire_simulate_c(
            steps,
            *configuration,
            *(pointers[name] for name in _REGION_NAMES),
        )
    )


def _output_snapshot(
    buffers: dict[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    return {name: buffers[name].copy() for name in _REGION_NAMES[1:]}


def _assert_outputs_unchanged(
    buffers: dict[str, npt.NDArray[np.float64]],
    before: dict[str, npt.NDArray[np.float64]],
) -> None:
    for name, expected in before.items():
        np.testing.assert_array_equal(buffers[name], expected)


def _arena_pointers(
    offsets: dict[str, int],
) -> tuple[npt.NDArray[np.float64], dict[str, int]]:
    arena = np.full(64, _SENTINEL, dtype=np.float64)
    current_offset = offsets["current"]
    arena[current_offset : current_offset + 2] = (4.5, 5.0)
    pointers = {
        name: int(arena.ctypes.data + offset * arena.itemsize) for name, offset in offsets.items()
    }
    return arena, pointers


def _default_offsets() -> dict[str, int]:
    return {
        "current": 0,
        "x_trace": 4,
        "y_trace": 8,
        "spikes": 12,
        "x_final": 16,
        "y_final": 18,
        "spike_count": 20,
    }


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(("left", "right"), _REGION_PAIRS)
def test_every_identical_active_region_pair_is_rejected_without_writes(
    backend: str,
    left: str,
    right: str,
) -> None:
    offsets = _default_offsets()
    offsets[right] = offsets[left]
    arena, pointers = _arena_pointers(offsets)
    before = arena.copy()
    assert _call(_module(backend)._lib, 2, pointers) == 1
    np.testing.assert_array_equal(arena, before)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(("left", "right"), _PARTIAL_REGION_PAIRS)
def test_every_multielement_region_rejects_partial_overlap_without_writes(
    backend: str,
    left: str,
    right: str,
) -> None:
    offsets = _default_offsets()
    if _REGION_LENGTHS[left] > 1:
        offsets[right] = offsets[left] + 1
    else:
        offsets[left] = offsets[right] + 1
    arena, pointers = _arena_pointers(offsets)
    before = arena.copy()
    assert _call(_module(backend)._lib, 2, pointers) == 1
    np.testing.assert_array_equal(arena, before)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adjacent_regions_are_accepted(backend: str) -> None:
    offsets = {
        "current": 0,
        "x_trace": 2,
        "y_trace": 4,
        "spikes": 6,
        "x_final": 8,
        "y_final": 9,
        "spike_count": 10,
    }
    arena, pointers = _arena_pointers(offsets)
    expected = public_backend.simulate_python(*_CONFIG, np.asarray([4.5, 5.0]))
    assert _call(_module(backend)._lib, 2, pointers) == 0
    np.testing.assert_allclose(arena[2:4], expected["x"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(arena[4:6], expected["y"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(arena[6:8], expected["spikes"], rtol=0.0, atol=1.0e-10)
    assert arena[8] == pytest.approx(float(expected["x_final"]), abs=1.0e-10)
    assert arena[9] == pytest.approx(float(expected["y_final"]), abs=1.0e-10)
    assert arena[10] == float(expected["spike_count"])


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("missing_pointer", _REGION_NAMES)
def test_null_required_pointer_returns_status_one_without_writes(
    backend: str,
    missing_pointer: str,
) -> None:
    buffers = _separate_buffers(np.asarray([4.5, 5.0]))
    pointers = _pointers(buffers)
    pointers[missing_pointer] = 0
    before = _output_snapshot(buffers)
    assert _call(_module(backend)._lib, 2, pointers) == 1
    _assert_outputs_unchanged(buffers, before)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_negative_step_count_returns_status_one_without_writes(backend: str) -> None:
    buffers = _separate_buffers(np.asarray([4.5, 5.0]))
    before = _output_snapshot(buffers)
    assert _call(_module(backend)._lib, -1, _pointers(buffers)) == 1
    _assert_outputs_unchanged(buffers, before)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_empty_batch_accepts_only_distinct_final_receipts(backend: str) -> None:
    x_final = np.full(1, _SENTINEL, dtype=np.float64)
    y_final = np.full(1, _SENTINEL, dtype=np.float64)
    spike_count = np.full(1, _SENTINEL, dtype=np.float64)
    pointers = {
        "current": 0,
        "x_trace": 0,
        "y_trace": 0,
        "spikes": 0,
        "x_final": int(x_final.ctypes.data),
        "y_final": int(y_final.ctypes.data),
        "spike_count": int(spike_count.ctypes.data),
    }
    assert _call(_module(backend)._lib, 0, pointers) == 0
    assert x_final[0] == _CONFIG[0]
    assert y_final[0] == _CONFIG[1]
    assert spike_count[0] == 0.0

    aliased = np.full(1, _SENTINEL, dtype=np.float64)
    pointers["x_final"] = int(aliased.ctypes.data)
    pointers["y_final"] = int(aliased.ctypes.data)
    assert _call(_module(backend)._lib, 0, pointers) == 1
    assert aliased[0] == _SENTINEL


@pytest.mark.parametrize(
    ("status", "configuration", "drive"),
    (
        (2, (*_CONFIG[:-1], -_CONFIG[-1]), np.asarray([4.5, 5.0])),
        (3, _CONFIG, np.asarray([4.5, np.nan])),
        (4, _LATE_FAILURE_CONFIG, np.zeros(4)),
    ),
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_numerical_statuses_leave_every_output_unchanged(
    backend: str,
    status: int,
    configuration: tuple[float, ...],
    drive: npt.NDArray[np.float64],
) -> None:
    buffers = _separate_buffers(drive)
    before = _output_snapshot(buffers)
    assert (
        _call(
            _module(backend)._lib,
            int(drive.size),
            _pointers(buffers),
            configuration,
        )
        == status
    )
    _assert_outputs_unchanged(buffers, before)


def test_go_header_uses_signed_32_bit_count_and_status() -> None:
    header = (
        _ROOT / "src/sc_neurocore/accel/go/resonate_and_fire/libresonate_and_fire.h"
    ).read_text(encoding="utf-8")
    signature = re.search(
        r"extern\s+int32_t\s+resonate_and_fire_simulate_c\s*\(\s*int32_t\s+n\b",
        header,
    )
    assert signature is not None


def test_mojo_llvm_uses_signed_32_bit_count_and_status(tmp_path: Path, pixi_cli: str) -> None:
    manifest = _ROOT / "src/sc_neurocore/accel/mojo/pixi.toml"
    source = _ROOT / "src/sc_neurocore/accel/mojo/resonate_and_fire/resonate_and_fire.mojo"
    llvm_path = tmp_path / "resonate_and_fire.ll"
    subprocess.run(
        [
            pixi_cli,
            "run",
            "--manifest-path",
            str(manifest),
            "mojo",
            "build",
            "--emit",
            "llvm",
            "--target-cpu",
            "x86-64-v3",
            "-o",
            str(llvm_path),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    llvm = llvm_path.read_text(encoding="utf-8")
    signature = re.search(
        r"define\b[^\n@]*\bi32\s+@resonate_and_fire_simulate_c\s*\(\s*i32\b",
        llvm,
    )
    assert signature is not None


def _load_builder() -> Any:
    path = _ROOT / "tools/build_accel_backends.py"
    spec = importlib.util.spec_from_file_location("model40_build_accel_backends", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_builder_discovers_model40_for_both_native_languages() -> None:
    builder = _load_builder()
    for language in _BACKENDS:
        targets = {target.name: target for target in builder.discover_targets(language)}
        assert "resonate_and_fire" in targets
        assert targets["resonate_and_fire"].output.name == "libresonate_and_fire.so"


def test_ci_requires_model40_for_both_native_builds() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required_line = next(
        line.strip() for line in workflow.splitlines() if line.strip().startswith("REQUIRED=")
    )
    required = set(required_line.removeprefix("REQUIRED=").split(","))
    assert "resonate_and_fire" in required
    assert workflow.count('build_accel_backends.py --language go --require "$REQUIRED"') == 1
    assert workflow.count('build_accel_backends.py --language mojo --require "$REQUIRED"') == 1
