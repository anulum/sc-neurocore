# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic fail-closed validation contracts

"""Malformed-input, optional-backend, and simulator-boundary regression tests."""

from __future__ import annotations

import builtins
import copy
from dataclasses import dataclass
import sys
import types
from typing import Any, Callable, cast

import numpy as np
import pytest

import sc_neurocore.optics._photonic_crosstalk as crosstalk_impl
import sc_neurocore.optics.photonic_emitter as facade
from sc_neurocore.optics.photonic_emitter import (
    BitstreamToOptical,
    CompilationResult,
    CrosstalkModel,
    FDTD2DSolver,
    FDTDSolver,
    MeepAdapter,
    OpticalPulse,
    PhotonicCompiler,
    PhotonicEmitter,
    PhotonicTarget,
    WaveguidePair,
)


@dataclass
class _Node:
    type: str
    id: str
    inputs: list[str]
    output: str


@dataclass
class _Graph:
    nodes: list[_Node]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PhotonicTarget(""),
        lambda: PhotonicTarget("x", modulator_type=""),
        lambda: PhotonicTarget("x", modulation=cast(Any, "phase")),
        lambda: PhotonicTarget("x", wavelength_nm=0.0),
        lambda: PhotonicTarget("x", q_factor=0.0),
        lambda: PhotonicTarget("x", insertion_loss_db=-1.0),
        lambda: PhotonicTarget("x", thermo_optic_coeff=float("nan")),
        lambda: OpticalPulse(float("nan"), 0.5, 1550.0, 10.0),
        lambda: OpticalPulse(0.0, 1.1, 1550.0, 10.0),
        lambda: OpticalPulse(0.0, 0.5, 0.0, 10.0),
        lambda: OpticalPulse(0.0, 0.5, 1550.0, 0.0),
    ],
)
def test_value_contracts_reject_invalid_physical_inputs(factory: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_conversion_and_compiler_fail_closed_on_malformed_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        BitstreamToOptical(cast(Any, object()))
    converter = BitstreamToOptical(PhotonicTarget.lightmatter())
    invalid_bits = (
        np.zeros((1, 1)),
        np.array([object()], dtype=object),
        np.array([float("nan")]),
        np.array([2.0]),
    )
    for bits in invalid_bits:
        with pytest.raises((TypeError, ValueError)):
            converter.convert(bits)
    with pytest.raises(ValueError):
        converter.convert(np.array([1]), pulse_duration_ps=0.0)
    with pytest.raises(ValueError):
        converter.optical_power_profile(np.array([1]), input_power_mw=-1.0)

    with pytest.raises(TypeError):
        PhotonicCompiler(cast(Any, object()))
    compiler = PhotonicCompiler(PhotonicTarget.lightmatter())
    with pytest.raises(ValueError, match="empty"):
        compiler.compile_bitstream(cast(Any, None))
    with pytest.raises(ValueError, match="one-dimensional"):
        compiler.compile_bitstream(np.array(1))
    with pytest.raises(TypeError, match="Boolean"):
        compiler.compile_bitstream(np.array([1]), run_fdtd=cast(Any, 1))
    with pytest.raises(TypeError, match="integer"):
        compiler.compile_bitstream(np.array([1]), fdtd_steps=True)
    with pytest.raises(ValueError):
        compiler.generate_mzi_verilog(1)
    with pytest.raises(ValueError):
        compiler.generate_microring_verilog(1)

    with pytest.raises((TypeError, ValueError)):
        CompilationResult("", 1, 1.0, 1.0, "")
    with pytest.raises(TypeError):
        CompilationResult("x", True, 1.0, 1.0, "")
    with pytest.raises(ValueError):
        CompilationResult("x", 1, -1.0, 1.0, "")
    with pytest.raises(ValueError):
        CompilationResult("x", 1, 1.0, -1.0, "")
    with pytest.raises(ValueError):
        CompilationResult("x", 1, 1.0, 1.0, "", -1.0)
    with pytest.raises(TypeError):
        CompilationResult("x", 1, 1.0, 1.0, cast(Any, 1))

    result = CompilationResult("x", 0, 1.0, 1.0, "")
    with pytest.raises(NotImplementedError):
        result.to_gdsii("x.gds")
    result.num_modulators = 1
    for filename, length, pitch in (("", 1.0, 1.0), ("x.gds", 0.0, 1.0), ("x.gds", 1.0, 0.0)):
        with pytest.raises(ValueError):
            result.to_gdsii(filename, length, pitch)

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "gdsfactory":
            raise ImportError("blocked for contract test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="gdsfactory is not installed"):
        result.to_gdsii("x.gds")


def test_fdtd_and_emitter_validation_rejects_malformed_state() -> None:
    with pytest.raises(TypeError):
        FDTDSolver(grid_size=True)
    with pytest.raises(ValueError):
        FDTDSolver(grid_size=4, boundary_cells=5)
    with pytest.raises(ValueError):
        FDTDSolver(dt_factor=1.1)
    one_d = FDTDSolver(grid_size=12, boundary_cells=2)
    with pytest.raises(ValueError):
        one_d.inject_pulse(12)
    with pytest.raises(ValueError):
        one_d.inject_pulse(2, phase=float("nan"))
    with pytest.raises(TypeError):
        one_d.step(True)

    with pytest.raises(ValueError):
        FDTD2DSolver(nx=5, ny=5, pml_layers=5)
    with pytest.raises(ValueError):
        FDTD2DSolver(dt_factor=1.1)
    two_d = FDTD2DSolver(nx=8, ny=8, pml_layers=1)
    with pytest.raises(ValueError):
        two_d.set_waveguide(8, 2)
    with pytest.raises(ValueError):
        two_d.set_waveguide(4, 2, refractive_index=0.9)
    with pytest.raises(ValueError):
        two_d.set_waveguide(4, 2, x_end=0)
    two_d.set_waveguide(4, 1)
    with pytest.raises(ValueError):
        two_d.inject_source(8, 4)
    with pytest.raises(ValueError):
        two_d.inject_source(4, 8)
    two_d.n_map[0, 0] = float("nan")
    with pytest.raises(ValueError):
        two_d.step()
    two_d.n_map[0, 0] = 0.0
    with pytest.raises(ValueError):
        two_d.step()
    with pytest.raises(ValueError):
        two_d.field_at_point(8, 0)
    with pytest.raises(ValueError):
        two_d.cross_section(8)

    with pytest.raises(ValueError):
        PhotonicEmitter("")
    emitter = PhotonicEmitter()
    with pytest.raises(TypeError):
        emitter.emit_lumerical_netlist(object())
    with pytest.raises(ValueError, match="identifiers"):
        emitter._topological_sort([_Node("x", "a", [], "x"), _Node("x", "a", [], "y")])
    with pytest.raises(ValueError, match="outputs"):
        emitter._topological_sort([_Node("x", "a", [], "x"), _Node("x", "b", [], "x")])
    with pytest.raises(ValueError, match="cycle"):
        emitter._topological_sort(
            [_Node("x", "a", ["b_out"], "a_out"), _Node("x", "b", ["a_out"], "b_out")]
        )
    ordered = emitter._topological_sort(
        [
            _Node("x", "a", [], "a_out"),
            _Node("x", "b", [], "b_out"),
            _Node("x", "c", ["a_out", "b_out"], "c_out"),
        ]
    )
    assert [node.id for node in ordered] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="two inputs"):
        emitter.emit_lumerical_netlist(_Graph([_Node("SC_AND", "a", ["x"], "y")]))
    with pytest.raises(ValueError, match="requires an input"):
        emitter.emit_lumerical_netlist(_Graph([_Node("LIF_MEMBRANE", "a", [], "y")]))
    assert "ADD" not in emitter.emit_lumerical_netlist(_Graph([_Node("UNKNOWN", "a", [], "y")]))


def test_crosstalk_validation_and_dynamic_backend_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ImportError):
        crosstalk_impl._unavailable_analyzer()
    sentinel = object()
    monkeypatch.delitem(sys.modules, "sc_neurocore.optics.photonic_emitter")
    assert crosstalk_impl._facade_binding("missing", sentinel) is sentinel
    monkeypatch.setitem(sys.modules, "sc_neurocore.optics.photonic_emitter", facade)

    for value in (True, 1.5):
        with pytest.raises(TypeError):
            CrosstalkModel().analyze_bank(cast(Any, value), 200.0, 10.0)
    with pytest.raises(ValueError):
        CrosstalkModel().analyze_bank(-1, 200.0, 10.0)
    with pytest.raises(ValueError):
        WaveguidePair(core_index=1.45, cladding_index=1.45)
    model = CrosstalkModel()
    with pytest.raises(TypeError):
        model.add_pair(cast(Any, object()))
    with pytest.raises(TypeError):
        model.transfer_matrix(cast(Any, object()))
    pair = WaveguidePair()
    with pytest.raises(ValueError):
        model.compute_crosstalk(pair, cast(Any, (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        model.compute_crosstalk(pair, (float("nan"), 0.0))
    with pytest.raises(ValueError, match="distinct"):
        model.analyze_pairs([(1, 1)], [200.0], [10.0])

    bank_calls: list[dict[str, object]] = []

    def fake_bank(**kwargs: object) -> dict[str, object]:
        bank_calls.append(kwargs)
        return {"backend": "rust", "num_waveguides": kwargs["num_waveguides"]}

    monkeypatch.setattr(facade, "_HAS_RUST_PH", True)
    monkeypatch.setattr(facade, "py_ph_analyze_crosstalk_bank", fake_bank)
    assert model.analyze_bank(3, 200.0, 10.0) == {"backend": "rust", "num_waveguides": 3}
    assert bank_calls[0]["gap_nm"] == 200.0

    def fake_pairs(**kwargs: object) -> dict[str, object]:
        pairs_a = cast(list[int], kwargs["pairs_a"])
        return {"backend": "rust", "num_pairs": len(pairs_a)}

    monkeypatch.setattr(facade, "py_ph_analyze_crosstalk_pairs", fake_pairs)
    assert model.analyze_pairs([(0, 1)], [200.0], [10.0]) == {
        "backend": "rust",
        "num_pairs": 1,
    }
    assert model.analyze_pairs([], [], [])["backend"] == "python"


def test_meep_geometry_validation_and_gaussian_empty_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        MeepAdapter.build_waveguide_geometry(cast(Any, object()))
    with pytest.raises(ValueError, match="at least one"):
        MeepAdapter.build_waveguide_geometry(PhotonicTarget.lightmatter(), substrate_index=0.9)
    geometry = MeepAdapter.build_waveguide_geometry(PhotonicTarget.silicon_photonics())
    with pytest.raises(TypeError):
        MeepAdapter.run_simulation(cast(Any, object()))

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.pop("sources"),
        lambda value: value.__setitem__("cell_size", [1.0, 2.0]),
        lambda value: value.__setitem__("cell_size", [True, 2.0, 0.0]),
        lambda value: value.__setitem__("cell_size", [float("nan"), 2.0, 0.0]),
        lambda value: value.__setitem__("cell_size", [-1.0, 2.0, 0.0]),
        lambda value: value.__setitem__("cell_size", [0.0, 2.0, 0.0]),
        lambda value: value.__setitem__("resolution", 0),
        lambda value: value.__setitem__("pml_layers", 0.0),
        lambda value: value.__setitem__("sources", []),
        lambda value: value["sources"][0].pop("size"),
        lambda value: value["sources"][0].__setitem__("type", "Unknown"),
        lambda value: value["sources"][0].__setitem__("frequency", 0.0),
        lambda value: value.__setitem__("geometry", []),
        lambda value: value["geometry"][0].pop("size"),
        lambda value: value["geometry"][0].__setitem__("type", "Sphere"),
        lambda value: value["geometry"][0].__setitem__("material_index", 0.0),
        lambda value: value["geometry"][0].__setitem__("size", [1.0, -1.0, "Infinity"]),
        lambda value: value["geometry"][0].__setitem__("size", [0.0, 1.0, "Infinity"]),
        lambda value: value.__setitem__("wavelength_nm", 0.0),
    )
    for mutate in mutations:
        malformed = copy.deepcopy(geometry)
        mutate(malformed)
        with pytest.raises((TypeError, ValueError)):
            MeepAdapter.run_simulation(malformed)

    calls: dict[str, object] = {}

    class Vector:
        def __init__(self, *coordinates: object):
            self.coordinates = coordinates

    class SourceFactory:
        def __init__(self, *, frequency: float):
            calls["source_factory"] = type(self).__name__
            self.frequency = frequency

    class ContinuousSource(SourceFactory):
        pass

    class GaussianSource(SourceFactory):
        pass

    class Simulation:
        def __init__(self, **kwargs: object):
            calls["simulation"] = kwargs

        def add_flux(self, *_args: object) -> str:
            return "handle"

        def run(self, *, until: float) -> None:
            calls["until"] = until

    fake_meep = types.SimpleNamespace(
        Vector3=Vector,
        ContinuousSource=ContinuousSource,
        GaussianSource=GaussianSource,
        Source=lambda *args, **kwargs: (args, kwargs),
        Medium=lambda **kwargs: kwargs,
        Block=lambda **kwargs: kwargs,
        PML=lambda thickness: thickness,
        FluxRegion=lambda **kwargs: kwargs,
        Simulation=Simulation,
        Ez="Ez",
        get_fluxes=lambda _handle: [],
    )
    monkeypatch.setitem(sys.modules, "meep", fake_meep)
    geometry.pop("wavelength_nm")
    result = MeepAdapter.run_simulation(geometry, run_time=4.0)
    assert calls["source_factory"] == "GaussianSource"
    assert result["transmission"] == 0.0
    assert result["wavelength_nm"] == 1550.0
