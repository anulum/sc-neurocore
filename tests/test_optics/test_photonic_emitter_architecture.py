# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic responsibility and fidelity contracts

"""Architecture, byte-fidelity, validation, and compatibility regression tests."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

import sc_neurocore.optics.photonic_emitter as facade
from sc_neurocore.optics.photonic_emitter import (
    BitstreamToOptical,
    CrosstalkModel,
    FDTD2DSolver,
    FDTDSolver,
    PhotonicCompiler,
    PhotonicEmitter,
    PhotonicTarget,
    WaveguidePair,
)

_REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass
class _Node:
    type: str
    id: str
    inputs: list[str]
    output: str


@dataclass
class _Graph:
    nodes: list[_Node]


def _json_digest(value: object) -> str:
    """Return a canonical SHA-256 digest for JSON-compatible evidence."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_digest(value: np.ndarray[Any, Any]) -> str:
    """Hash dtype, shape, and byte-exact array content."""
    payload = value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_facade_preserves_exact_public_surface_signatures_and_pickle_identity() -> None:
    """Keep historical imports and serialised qualified names stable."""
    expected = {
        "BitstreamToOptical",
        "CompilationResult",
        "CrosstalkModel",
        "FDTD2DSolver",
        "FDTDSolver",
        "MeepAdapter",
        "OpticalModulation",
        "OpticalPulse",
        "PhotonicCompiler",
        "PhotonicEmitter",
        "PhotonicTarget",
        "WaveguidePair",
    }
    assert set(facade.__all__) == expected
    assert not hasattr(facade, "_public_class")
    for name in expected:
        public_class = getattr(facade, name)
        assert public_class.__module__ == "sc_neurocore.optics.photonic_emitter"

    restored = pickle.loads(pickle.dumps(PhotonicTarget.lightmatter()))
    assert type(restored) is PhotonicTarget
    assert restored == PhotonicTarget.lightmatter()
    assert list(inspect.signature(PhotonicCompiler.compile_bitstream).parameters) == [
        "self",
        "bitstream",
        "run_fdtd",
        "fdtd_steps",
    ]
    assert list(inspect.signature(CrosstalkModel.analyze_pairs).parameters) == [
        "self",
        "pair_indices",
        "gaps_nm",
        "coupling_lengths_um",
        "wavelength_nm",
        "core_index",
        "cladding_index",
    ]


def test_responsibility_modules_have_bounded_size_and_one_way_imports() -> None:
    """Prevent the compatibility facade or any private responsibility becoming a GodFile."""
    directory = _REPOSITORY / "src/sc_neurocore/optics"
    allowed = {
        "_photonic_types.py": set(),
        "_photonic_conversion.py": {"_photonic_types"},
        "_photonic_fdtd.py": {"_photonic_types"},
        "_photonic_emitter.py": set(),
        "_photonic_compiler.py": {
            "_photonic_conversion",
            "_photonic_emitter",
            "_photonic_fdtd",
            "_photonic_types",
        },
        "_photonic_meep.py": {"_photonic_types"},
        "_photonic_crosstalk.py": {"_photonic_types"},
    }
    for filename, expected_imports in allowed.items():
        source = (directory / filename).read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 350
        tree = ast.parse(source)
        relative = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }
        assert relative == expected_imports

    facade_source = (directory / "photonic_emitter.py").read_text(encoding="utf-8")
    assert len(facade_source.splitlines()) <= 100
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef))
        for node in ast.walk(ast.parse(facade_source))
    )


def test_valid_outputs_remain_byte_exact_after_responsibility_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin every deterministic valid-output family to its pre-split payload."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    expected_conversions = {
        "lightmatter": "2cfb8a45a94838cc1c7c76dc760d54f85f918b615ca86cfbb28f49b0407d64fc",
        "silicon_photonics": "b135e883506a4cced5f477e0634e3320d31e0ac8a1aea0ec069e1ab8dba76396",
        "two_d_waveguide": "93ddd92e0a4cb569bbe0fcb7c8d8c495c56feb8d866fb2df04c9a15a7fad657f",
    }
    for target_name, expected_digest in expected_conversions.items():
        target = getattr(PhotonicTarget, target_name)()
        pulses = [asdict(pulse) for pulse in BitstreamToOptical(target).convert(bits)]
        assert _json_digest(pulses) == expected_digest

    for target, expected_digest in (
        (
            PhotonicTarget.lightmatter(),
            "c1562fd358685618e1df84ccb1ec75d822b356f3d134e1f853247fb3cdc47cfa",
        ),
        (
            PhotonicTarget.silicon_photonics(),
            "22b65aaf3cf5b491e4caa83de8a86680f698b0eee40cbedda946a32c1da063b9",
        ),
    ):
        result = PhotonicCompiler(target).compile_bitstream(bits)
        assert _json_digest(asdict(result)) == expected_digest

    graph = _Graph(
        [
            _Node("LIF_MEMBRANE", "n1", ["bus"], "spike"),
            _Node("SC_AND", "m1", ["a", "b"], "bus"),
        ]
    )
    assert _json_digest(PhotonicEmitter("imec").emit_lumerical_netlist(graph)) == (
        "a1866a2fac4824fe8cc4fb3af0a92d3616da908a58a754d23de0def8cfb7b7ce"
    )

    one_d = FDTDSolver(grid_size=96, boundary_cells=8)
    one_d.inject_pulse(24, amplitude=0.7, phase=0.2)
    one_d.set_loss(1.5)
    one_d.step(7)
    ez, hy = one_d.snapshot()
    assert _array_digest(np.concatenate((ez, hy))) == (
        "54f718a7fde7c14ecc1a0c8aac682f8e76bf63705b904d0548d74f2e69840241"
    )
    assert one_d.field_energy() == 14.952394987886558

    two_d = FDTD2DSolver(nx=24, ny=18, pml_layers=3)
    two_d.set_waveguide(9, 4)
    two_d.inject_source(5, 9, sigma_cells=2)
    two_d.step(3)
    ez2, hx2, hy2 = two_d.snapshot()
    assert _array_digest(np.concatenate((ez2.ravel(), hx2.ravel(), hy2.ravel()))) == (
        "e47fe8d88913ec7727999852306b6b664363dd558049945f2bc782e6bdf197ee"
    )
    assert two_d.field_energy() == 0.01806380172177048

    monkeypatch.setattr(facade, "_HAS_RUST_PH", False)
    pair = WaveguidePair(gap_nm=220.0, coupling_length_um=15.0)
    model = CrosstalkModel()
    model.add_pair(pair)
    crosstalk = {
        "props": [
            pair.effective_index_diff,
            pair.coupling_coefficient,
            pair.coupling_ratio,
            pair.isolation_db,
        ],
        "bank": model.analyze_bank(5, 220.0, 15.0),
        "pairs": model.analyze_pairs([(0, 1), (1, 3)], [220.0, 440.0], [15.0, 20.0]),
    }
    assert _json_digest(crosstalk) == (
        "b7fa4265c39d04ee7e54ab2f3972f3c519c6761ca5348e66a9736dc5968e9180"
    )
    compiler = PhotonicCompiler(PhotonicTarget.lightmatter())
    assert _json_digest(compiler.generate_mzi_verilog(12)) == (
        "dcd838cfa22ca3e6175038747a3c4d3d3b6d66147b38e382ff85c4196ac8c372"
    )
    assert _json_digest(compiler.generate_microring_verilog(12)) == (
        "0ed414ad90074959561c10263490e68a7c250696582af55edf6091dc6218aaf5"
    )
