# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Managed Julia QCFS lifetime and admission acceptance

"""Exercise the Julia QCFS module across real shutdown, import order and a skewed package."""

import shutil
from pathlib import Path

from tests.test_conversion_qcfs_native import ROOT, qcfs_environment, run_probe
from tests.test_conversion_qcfs_native import _MEASURE as MEASURE

__all__ = ["qcfs_environment"]

_SHUTDOWN = r"""
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False)
    measurement.start()
import atexit
import numpy as np
x, output = np.array([0.5]), np.zeros(1)
inputs, thresholds = np.zeros(1), np.zeros(1)
def at_exit():
    for call in (
        lambda: api.forward(8, 1.0, x.ctypes.data, 1, output.ctypes.data),
        lambda: api.backward(8, 1.0, x.ctypes.data, x.ctypes.data, 1, inputs.ctypes.data, thresholds.ctypes.data),
        load_qcfs_julia,
    ):
        try:
            call()
        except RuntimeError as error:
            assert "closing" in str(error), str(error)
        else:
            raise AssertionError("stored Julia QCFS callback entered a closed runtime")
    if measurement is not None:
        measurement.stop()
        measurement.save()
    print("stored Julia QCFS callbacks refused after shutdown", flush=True)
atexit.register(at_exit)
from sc_neurocore.conversion.qcfs_native import load_qcfs_julia
api = load_qcfs_julia()
assert api.forward(8, 1.0, x.ctypes.data, 1, output.ctypes.data) == 0 and output.tolist() == [0.5]
print("Julia QCFS ran before shutdown", flush=True)
"""


def test_stored_julia_callbacks_refuse_after_shutdown(qcfs_environment: dict[str, str]) -> None:
    """Callbacks kept past JuliaCall's exit hook refuse instead of entering a closed runtime."""
    output = run_probe(_SHUTDOWN, qcfs_environment, "julia-shutdown")
    assert "Julia QCFS ran before shutdown" in output
    assert "stored Julia QCFS callbacks refused after shutdown" in output


_TORCH_FIRST = (
    MEASURE
    + r"""
import torch
from sc_neurocore.conversion import qcfs_forward
try:
    qcfs_forward([0.5], backend="julia")
except RuntimeError as error:
    assert "unavailable or incompatible" not in str(error), str(error)
    assert "before importing PyTorch" in str(error), str(error)
else:
    raise AssertionError("Julia QCFS started after PyTorch")
print("PyTorch-first start refused", flush=True)
"""
)


def test_julia_must_start_before_pytorch(qcfs_environment: dict[str, str]) -> None:
    """Starting JuliaCall after PyTorch is refused with the reason, not a generic failure."""
    assert "PyTorch-first start refused" in run_probe(_TORCH_FIRST, qcfs_environment, "torch-first")


_SKEWED = (
    MEASURE
    + r"""
from sc_neurocore.conversion import qcfs_forward
try:
    qcfs_forward([0.5], backend="julia")
except RuntimeError as error:
    assert "Julia QCFS managed runtime unavailable or incompatible" in str(error), str(error)
    assert "incompatible array ABI" in str(error.__cause__), repr(error.__cause__)
else:
    raise AssertionError("a Julia QCFS module reporting another ABI was accepted")
print("skewed Julia QCFS module refused", flush=True)
"""
)


def test_julia_module_from_another_abi_is_refused(
    qcfs_environment: dict[str, str], tmp_path: Path
) -> None:
    """A package whose Julia QCFS source reports ABI two is refused before any call.

    The copy is a real version-skewed package: its Julia source says ABI two
    while its Python admits one.
    """
    skewed = tmp_path / "abi-mismatch"
    shutil.copytree(
        ROOT / "src/sc_neurocore",
        skewed / "sc_neurocore",
        ignore=shutil.ignore_patterns("__pycache__", "target"),
    )
    module = skewed / "sc_neurocore/accel/julia/conversion/qcfs.jl"
    text = module.read_text()
    assert text.count("sc_qcfs_abi_version()::UInt32 = 1") == 1
    module.write_text(
        text.replace("sc_qcfs_abi_version()::UInt32 = 1", "sc_qcfs_abi_version()::UInt32 = 2")
    )
    settings = dict(qcfs_environment, PYTHONPATH=str(skewed))
    output = run_probe(_SKEWED, settings, "julia-abi", source=skewed / "sc_neurocore/conversion")
    assert "skewed Julia QCFS module refused" in output
