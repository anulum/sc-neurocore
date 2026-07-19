# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Direct alpha-synapse Go and Mojo ABI contracts

"""Exercise width, pointer, overlap, status, atomicity, and build boundaries."""

from __future__ import annotations

import importlib.util
import itertools
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import alpha as public_backend

_ROOT = Path(__file__).resolve().parents[1]
_BACKENDS = ("go", "mojo")
_REGION_NAMES = (
    "exc_current",
    "inh_current",
    "v_trace",
    "a_exc_trace",
    "i_exc_trace",
    "a_inh_trace",
    "i_inh_trace",
    "spikes",
    "v_final",
    "a_exc_final",
    "i_exc_final",
    "a_inh_final",
    "i_inh_final",
    "spike_count",
)
_REGION_LENGTHS = {
    "exc_current": 2,
    "inh_current": 2,
    "v_trace": 2,
    "a_exc_trace": 2,
    "i_exc_trace": 2,
    "a_inh_trace": 2,
    "i_inh_trace": 2,
    "spikes": 2,
    "v_final": 1,
    "a_exc_final": 1,
    "i_exc_final": 1,
    "a_inh_final": 1,
    "i_inh_final": 1,
    "spike_count": 1,
}
_REGION_PAIRS = tuple(itertools.combinations(_REGION_NAMES, 2))
_PARTIAL_REGION_PAIRS = tuple(
    pair for pair in _REGION_PAIRS if max(_REGION_LENGTHS[pair[0]], _REGION_LENGTHS[pair[1]]) > 1
)
_CONFIG = (0.15, 0.08, 0.05, 0.04, 0.03, -0.5, 1.2, 16.0, 4.0, 9.0, 0.5)
_LATE_FAILURE_CONFIG = (-1.0e308, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0)
_SENTINEL = -999.0


def _module(backend: str) -> Any:
    module = __import__(
        f"sc_neurocore.accel.{backend}.alpha",
        fromlist=["alpha"],
    )
    marker = f"_HAS_{backend.upper()}_ALPHA"
    assert bool(getattr(module, marker))
    assert module._lib is not None
    return module


def _separate_buffers(
    exc: npt.NDArray[np.float64],
    inh: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    steps = int(exc.size)
    return {
        "exc_current": np.ascontiguousarray(exc, dtype=np.float64),
        "inh_current": np.ascontiguousarray(inh, dtype=np.float64),
        "v_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "a_exc_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "i_exc_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "a_inh_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "i_inh_trace": np.full(steps, _SENTINEL, dtype=np.float64),
        "spikes": np.full(steps, _SENTINEL, dtype=np.float64),
        "v_final": np.full(1, _SENTINEL, dtype=np.float64),
        "a_exc_final": np.full(1, _SENTINEL, dtype=np.float64),
        "i_exc_final": np.full(1, _SENTINEL, dtype=np.float64),
        "a_inh_final": np.full(1, _SENTINEL, dtype=np.float64),
        "i_inh_final": np.full(1, _SENTINEL, dtype=np.float64),
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
        library.alpha_simulate_c(
            steps,
            *configuration,
            *(pointers[name] for name in _REGION_NAMES),
        )
    )


def _output_snapshot(
    buffers: dict[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    return {name: buffers[name].copy() for name in _REGION_NAMES[2:]}


def _assert_outputs_unchanged(
    buffers: dict[str, npt.NDArray[np.float64]],
    before: dict[str, npt.NDArray[np.float64]],
) -> None:
    for name, expected in before.items():
        np.testing.assert_array_equal(buffers[name], expected)


def _arena_pointers(
    offsets: dict[str, int],
) -> tuple[npt.NDArray[np.float64], dict[str, int]]:
    arena = np.full(96, _SENTINEL, dtype=np.float64)
    exc_offset = offsets["exc_current"]
    inh_offset = offsets["inh_current"]
    arena[exc_offset : exc_offset + 2] = (2.0, 2.5)
    arena[inh_offset : inh_offset + 2] = (0.5, 0.75)
    pointers = {
        name: int(arena.ctypes.data + offset * arena.itemsize) for name, offset in offsets.items()
    }
    return arena, pointers


def _default_offsets() -> dict[str, int]:
    return {
        "exc_current": 0,
        "inh_current": 4,
        "v_trace": 8,
        "a_exc_trace": 12,
        "i_exc_trace": 16,
        "a_inh_trace": 20,
        "i_inh_trace": 24,
        "spikes": 28,
        "v_final": 32,
        "a_exc_final": 34,
        "i_exc_final": 36,
        "a_inh_final": 38,
        "i_inh_final": 40,
        "spike_count": 42,
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
        "exc_current": 0,
        "inh_current": 2,
        "v_trace": 4,
        "a_exc_trace": 6,
        "i_exc_trace": 8,
        "a_inh_trace": 10,
        "i_inh_trace": 12,
        "spikes": 14,
        "v_final": 16,
        "a_exc_final": 17,
        "i_exc_final": 18,
        "a_inh_final": 19,
        "i_inh_final": 20,
        "spike_count": 21,
    }
    arena, pointers = _arena_pointers(offsets)
    expected = public_backend.simulate_python(
        *_CONFIG, np.asarray([2.0, 2.5]), np.asarray([0.5, 0.75])
    )
    assert _call(_module(backend)._lib, 2, pointers) == 0
    np.testing.assert_allclose(arena[4:6], expected["v"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(arena[6:8], expected["a_exc"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(arena[8:10], expected["i_exc"], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(arena[14:16], expected["spikes"], rtol=0.0, atol=1.0e-10)
    assert arena[16] == pytest.approx(float(expected["v_final"]), abs=1.0e-10)
    assert arena[17] == pytest.approx(float(expected["a_exc_final"]), abs=1.0e-10)
    assert arena[21] == float(expected["spike_count"])


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("missing_pointer", _REGION_NAMES)
def test_null_required_pointer_returns_status_one_without_writes(
    backend: str,
    missing_pointer: str,
) -> None:
    buffers = _separate_buffers(np.asarray([2.0, 2.5]), np.asarray([0.5, 0.75]))
    pointers = _pointers(buffers)
    pointers[missing_pointer] = 0
    before = _output_snapshot(buffers)
    assert _call(_module(backend)._lib, 2, pointers) == 1
    _assert_outputs_unchanged(buffers, before)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_negative_step_count_returns_status_one_without_writes(backend: str) -> None:
    buffers = _separate_buffers(np.asarray([2.0, 2.5]), np.asarray([0.5, 0.75]))
    before = _output_snapshot(buffers)
    assert _call(_module(backend)._lib, -1, _pointers(buffers)) == 1
    _assert_outputs_unchanged(buffers, before)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_empty_batch_accepts_only_distinct_final_receipts(backend: str) -> None:
    finals = {name: np.full(1, _SENTINEL, dtype=np.float64) for name in _REGION_NAMES[8:]}
    pointers = {
        **{name: 0 for name in _REGION_NAMES[:8]},
        **{name: int(buffer.ctypes.data) for name, buffer in finals.items()},
    }
    assert _call(_module(backend)._lib, 0, pointers) == 0
    assert finals["v_final"][0] == _CONFIG[0]
    assert finals["a_exc_final"][0] == _CONFIG[1]
    assert finals["spike_count"][0] == 0.0

    aliased = np.full(1, _SENTINEL, dtype=np.float64)
    pointers["v_final"] = int(aliased.ctypes.data)
    pointers["a_exc_final"] = int(aliased.ctypes.data)
    assert _call(_module(backend)._lib, 0, pointers) == 1
    assert aliased[0] == _SENTINEL


@pytest.mark.parametrize(
    ("status", "configuration", "exc", "inh"),
    (
        (2, (*_CONFIG[:-1], -_CONFIG[-1]), np.asarray([2.0, 2.5]), np.asarray([0.5, 0.75])),
        (3, _CONFIG, np.asarray([2.0, np.nan]), np.asarray([0.5, 0.75])),
        (4, _LATE_FAILURE_CONFIG, np.full(4, 1.0e308), np.zeros(4)),
    ),
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_numerical_statuses_leave_every_output_unchanged(
    backend: str,
    status: int,
    configuration: tuple[float, ...],
    exc: npt.NDArray[np.float64],
    inh: npt.NDArray[np.float64],
) -> None:
    buffers = _separate_buffers(exc, inh)
    before = _output_snapshot(buffers)
    assert (
        _call(
            _module(backend)._lib,
            int(exc.size),
            _pointers(buffers),
            configuration,
        )
        == status
    )
    _assert_outputs_unchanged(buffers, before)


def test_go_header_uses_signed_32_bit_count_and_status() -> None:
    header = (_ROOT / "src/sc_neurocore/accel/go/alpha/libalpha.h").read_text(encoding="utf-8")
    signature = re.search(
        r"extern\s+int32_t\s+alpha_simulate_c\s*\(\s*int32_t\s+n\b",
        header,
    )
    assert signature is not None


def test_mojo_llvm_uses_signed_32_bit_count_and_status(tmp_path: Path) -> None:
    assert shutil.which("pixi") is not None
    manifest = _ROOT / "src/sc_neurocore/accel/mojo/pixi.toml"
    source = _ROOT / "src/sc_neurocore/accel/mojo/alpha/alpha.mojo"
    llvm_path = tmp_path / "alpha.ll"
    subprocess.run(
        [
            "pixi",
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
        r"define\b[^\n@]*\bi32\s+@alpha_simulate_c\s*\(\s*i32\b",
        llvm,
    )
    assert signature is not None


def _load_builder() -> Any:
    path = _ROOT / "tools/build_accel_backends.py"
    spec = importlib.util.spec_from_file_location("model42_build_accel_backends", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_builder_discovers_model42_for_both_native_languages() -> None:
    builder = _load_builder()
    for language in _BACKENDS:
        targets = {target.name: target for target in builder.discover_targets(language)}
        assert "alpha" in targets
        assert targets["alpha"].output.name == "libalpha.so"


def test_ci_requires_model42_for_both_native_builds() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required_line = next(
        line.strip() for line in workflow.splitlines() if line.strip().startswith("REQUIRED=")
    )
    required = set(required_line.removeprefix("REQUIRED=").split(","))
    assert "alpha" in required
    assert workflow.count('build_accel_backends.py --language go --require "$REQUIRED"') == 1
