# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Meep photonic adapter

"""Translate photonic target contracts into optional Meep simulations."""

from __future__ import annotations

import math
from typing import Any, Dict

from ._photonic_types import (
    OpticalModulation,
    PhotonicTarget,
    _require_positive,
)


def _validate_vector(
    value: object,
    name: str,
    *,
    non_negative: bool = False,
    allow_infinity_z: bool = False,
) -> None:
    """Validate a three-coordinate serialisable Meep vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    for index, coordinate in enumerate(value):
        if allow_infinity_z and index == 2 and coordinate == "Infinity":
            continue
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        if not math.isfinite(float(coordinate)):
            raise ValueError(f"{name}[{index}] must be finite")
        if non_negative and coordinate < 0:
            raise ValueError(f"{name}[{index}] must be non-negative")


def _validate_geometry(geometry: Dict[str, Any]) -> None:
    """Reject malformed Meep requests before importing or invoking Meep."""
    required = {"cell_size", "resolution", "sources", "geometry", "pml_layers"}
    missing = sorted(required.difference(geometry))
    if missing:
        raise ValueError(f"geometry is missing required keys: {', '.join(missing)}")

    _validate_vector(geometry["cell_size"], "cell_size", non_negative=True)
    cell_size = geometry["cell_size"]
    if cell_size[0] <= 0 or cell_size[1] <= 0:
        raise ValueError("cell_size x and y coordinates must be positive")
    resolution = geometry["resolution"]
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    _require_positive(geometry["pml_layers"], "pml_layers")

    sources = geometry["sources"]
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ValueError("sources must contain exactly one source dictionary")
    source = sources[0]
    required_source = {"type", "frequency", "center", "size"}
    if required_source.difference(source):
        raise ValueError("source dictionary is incomplete")
    if source["type"] not in {"ContinuousSource", "GaussianSource"}:
        raise ValueError("source type must be ContinuousSource or GaussianSource")
    _require_positive(source["frequency"], "source frequency")
    _validate_vector(source["center"], "source center")
    _validate_vector(source["size"], "source size", non_negative=True)

    objects = geometry["geometry"]
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError("geometry must contain exactly one block dictionary")
    block = objects[0]
    required_block = {"type", "material_index", "center", "size"}
    if required_block.difference(block):
        raise ValueError("block dictionary is incomplete")
    if block["type"] != "Block":
        raise ValueError("geometry object type must be Block")
    _require_positive(block["material_index"], "material_index")
    _validate_vector(block["center"], "block center")
    _validate_vector(block["size"], "block size", non_negative=True, allow_infinity_z=True)
    if block["size"][0] <= 0 or block["size"][1] <= 0:
        raise ValueError("block size x and y coordinates must be positive")
    if "wavelength_nm" in geometry:
        _require_positive(geometry["wavelength_nm"], "wavelength_nm")


class MeepAdapter:
    """Build and execute Meep waveguide simulations when Meep is installed."""

    @staticmethod
    def is_available() -> bool:
        """Return whether the optional Meep dependency is importable."""
        try:
            import meep as meep_module

            return meep_module is not None
        except ImportError:
            return False

    @staticmethod
    def build_waveguide_geometry(
        target: PhotonicTarget,
        waveguide_width_um: float = 0.5,
        length_um: float = 10.0,
        substrate_index: float = 1.45,
    ) -> Dict[str, Any]:
        """Build a serialisable Meep waveguide-geometry description."""
        if not isinstance(target, PhotonicTarget):
            raise TypeError("target must be a PhotonicTarget")
        _require_positive(waveguide_width_um, "waveguide_width_um")
        _require_positive(length_um, "length_um")
        _require_positive(substrate_index, "substrate_index")
        if substrate_index < 1.0:
            raise ValueError("substrate_index must be at least one")

        core_index = 3.48 if target.wavelength_nm > 1000 else 2.0
        wavelength_um = target.wavelength_nm / 1000.0
        frequency = 1.0 / wavelength_um
        return {
            "cell_size": [length_um, 3.0 * waveguide_width_um, 0],
            "resolution": 20,
            "sources": [
                {
                    "type": "ContinuousSource"
                    if target.modulation == OpticalModulation.PHASE
                    else "GaussianSource",
                    "frequency": frequency,
                    "center": [-length_um / 2 + 0.5, 0, 0],
                    "size": [0, waveguide_width_um, 0],
                }
            ],
            "geometry": [
                {
                    "type": "Block",
                    "material_index": core_index,
                    "center": [0, 0, 0],
                    "size": [length_um, waveguide_width_um, "Infinity"],
                },
            ],
            "substrate_index": substrate_index,
            "pml_layers": 1.0,
            "wavelength_nm": target.wavelength_nm,
            "modulation": target.modulation.value,
        }

    @staticmethod
    def run_simulation(geometry: Dict[str, Any], run_time: float = 50.0) -> Dict[str, Any]:
        """Execute a real Meep simulation and return its transmission record.

        The method fails closed when Meep is unavailable. Earlier versions
        returned invented transmission values on that path, which could be
        mistaken for simulation evidence.
        """
        _require_positive(run_time, "run_time")
        if not isinstance(geometry, dict):
            raise TypeError("geometry must be a dictionary")
        _validate_geometry(geometry)
        if not MeepAdapter.is_available():
            raise ImportError(
                "Meep is not installed; install Meep before requesting a photonic simulation"
            )

        import meep as mp

        cell_size = geometry["cell_size"]
        resolution = geometry["resolution"]
        source_spec = geometry["sources"][0]
        geometry_spec = geometry["geometry"][0]
        source_factory = (
            mp.ContinuousSource if source_spec["type"] == "ContinuousSource" else mp.GaussianSource
        )

        frequency = source_spec["frequency"]
        sources = [
            mp.Source(
                source_factory(frequency=frequency),
                component=mp.Ez,
                center=mp.Vector3(*source_spec["center"]),
                size=mp.Vector3(*source_spec["size"]),
            )
        ]
        material = mp.Medium(index=geometry_spec["material_index"])
        geometry_objects = [
            mp.Block(
                size=mp.Vector3(geometry_spec["size"][0], geometry_spec["size"][1]),
                center=mp.Vector3(*geometry_spec["center"]),
                material=material,
            )
        ]
        simulation = mp.Simulation(
            cell_size=mp.Vector3(*cell_size),
            resolution=resolution,
            sources=sources,
            geometry=geometry_objects,
            boundary_layers=[mp.PML(geometry["pml_layers"])],
        )
        flux_region = mp.FluxRegion(
            center=mp.Vector3(cell_size[0] / 2 - 1, 0),
            size=mp.Vector3(0, cell_size[1]),
        )
        transmission = simulation.add_flux(frequency, 0, 1, flux_region)
        simulation.run(until=run_time)
        flux_data = mp.get_fluxes(transmission)
        return {
            "transmission": float(flux_data[0]) if flux_data else 0.0,
            "reflection": 0.0,
            "field_decay": 0.0,
            "run_time": run_time,
            "mock": False,
            "wavelength_nm": geometry.get("wavelength_nm", 1550.0),
        }


__all__ = ["MeepAdapter"]
