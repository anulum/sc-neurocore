# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Five-runtime controlled-uniform dispatch for SC stochastic adaptation."""

from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any, Protocol, cast
import numpy as np
import numpy.typing as npt
from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

KERNEL = "sc_stochastic_rate_adaptation_rk4_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2e-12, "julia": 2e-12, "go": 2e-12, "mojo": 2e-10}
_AUTO_BACKENDS = with_floor("python")


class _Runner(Protocol):
    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _rust() -> Any | None:
    try:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return None


def _native(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.sc_stochastic_rate_adaptation")


def backend_available(backend: str) -> bool:
    if backend == "python":
        return True
    if backend == "rust":
        module = _rust()
        return module is not None and hasattr(module, "py_sc_stochastic_rate_adaptation_simulate")
    if backend == "julia":
        try:
            importlib.import_module(
                "sc_neurocore.accel.julia.neurons"
            )._ensure_sc_stochastic_rate_adaptation_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_SC_STOCHASTIC_RATE_ADAPTATION", False))
    return False


def auto_backend() -> str:
    return next(
        (b for b in select_backend_order(KERNEL, static=_AUTO_BACKENDS) if backend_available(b)),
        "python",
    )


def _python(
    config: tuple[float, ...], drive: npt.NDArray[np.float64], uniforms: npt.NDArray[np.float64]
) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
        SCStochasticRateAdaptationNeuron,
    )

    n = SCStochasticRateAdaptationNeuron(
        a=config[0],
        f_max=config[1],
        beta=config[2],
        i_half=config[3],
        tau_a=config[4],
        delta_a=config[5],
        dt=config[6],
        seed=0,
    )
    adaptation = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    for index, (current, uniform) in enumerate(zip(drive, uniforms, strict=True)):
        events[index] = n.step_with_uniform(float(current), float(uniform))
        adaptation[index] = n.a
    return {"adaptation": adaptation, "events": events, "a_final": n.a}


def simulate_sc_stochastic_rate_adaptation(
    currents: npt.ArrayLike,
    uniforms: npt.ArrayLike,
    *,
    a: float = 0.0,
    f_max: float = 200.0,
    beta: float = 0.1,
    i_half: float = 5.0,
    tau_a: float = 100.0,
    delta_a: float = 0.5,
    dt: float = 1.0,
    backend: str = "auto",
) -> dict[str, object]:
    config = tuple(float(x) for x in (a, f_max, beta, i_half, tau_a, delta_a, dt))
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    randoms = np.ascontiguousarray(uniforms, dtype=np.float64)
    if (
        drive.ndim != 1
        or drive.shape != randoms.shape
        or not np.isfinite(drive).all()
        or not np.isfinite(randoms).all()
        or not ((randoms >= 0) & (randoms < 1)).all()
    ):
        raise ValueError("SC stochastic currents/uniforms must be finite matched vectors")
    _python(config, drive[:0], randoms[:0])
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown SC stochastic backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} SC stochastic backend is unavailable")
    if selected == "python":
        result = _python(config, drive, randoms)
    elif selected == "rust":
        module = _rust()
        if module is None:
            raise RuntimeError("rust SC stochastic backend is unavailable")
        result = cast(_Runner, module.py_sc_stochastic_rate_adaptation_simulate)(
            *config, drive, randoms
        )
    elif selected == "julia":
        result = importlib.import_module(
            "sc_neurocore.accel.julia.neurons"
        ).simulate_sc_stochastic_rate_adaptation(
            drive,
            randoms,
            **dict(
                zip(("a", "f_max", "beta", "i_half", "tau_a", "delta_a", "dt"), config, strict=True)
            ),
        )
    else:
        result = cast(_Runner, _native(selected).simulate_sc_stochastic_rate_adaptation)(
            *config, drive, randoms
        )
    adaptation = np.ascontiguousarray(result["adaptation"], dtype=np.float64)
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if (
        adaptation.shape != (drive.size,)
        or not np.isfinite(adaptation).all()
        or events.shape != (drive.size,)
        or not np.isin(events, (0, 1)).all()
    ):
        raise FloatingPointError("SC stochastic backend returned malformed trace")
    final = float(cast(float, result["a_final"]))
    expected = config[0] if not drive.size else float(adaptation[-1])
    if final != expected:
        raise FloatingPointError("SC stochastic final adaptation disagrees with trace")
    return {"adaptation": adaptation, "events": events, "a_final": final}


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_sc_stochastic_rate_adaptation",
]
