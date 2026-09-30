# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Five-runtime QCFS quantisation and derivative acceptance

"""Require bit-identical QCFS values and derivatives from NumPy, Rust, Go, Mojo and Julia.

The Rust, Go and Mojo libraries are compiled from the maintained sources and
Julia runs in an owned offline project; every call goes through the public
``qcfs_forward``/``qcfs_backward`` API. The NumPy reference is checked against
``QCFSActivation`` autograd itself.
"""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sc_neurocore.conversion import qcfs_backward, qcfs_forward
from sc_neurocore.conversion.qcfs import QCFSActivation
from sc_neurocore.conversion.qcfs_native import load_qcfs_library
from tests.julia_runtimes import require_julia_runtime
from sc_neurocore.accel.mojo.isa_baseline import pin_isa

ROOT = Path(__file__).resolve().parents[1]
ACCEL = ROOT / "src/sc_neurocore/accel"
LIBRARIES = ("SC_NEUROCORE_QCFS_RUST_LIB", "SC_NEUROCORE_QCFS_GO_LIB", "SC_NEUROCORE_QCFS_MOJO_LIB")


def qcfs_libraries(accel: Path, workspace: Path) -> dict[str, Path]:
    """Build the Rust, Go and Mojo QCFS libraries from one ``accel`` tree.

    Parameters
    ----------
    accel:
        The ``sc_neurocore/accel`` directory whose sources are built.
    workspace:
        Directory receiving the build outputs.

    Returns
    -------
    dict of str to pathlib.Path
        Configuration variable name to built shared library.
    """
    subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--release",
            "--manifest-path",
            str(accel / "rust/safety/qcfs_native/Cargo.toml"),
            "--target-dir",
            str(workspace / "rust"),
        ],
        capture_output=True,
        check=True,
        timeout=300,
    )
    subprocess.run(
        [
            "go",
            "build",
            "-buildvcs=false",
            "-buildmode=c-shared",
            "-o",
            str(workspace / "go.so"),
            "./conversion/qcfscshared",
        ],
        cwd=accel / "go",
        env=dict(os.environ, GOEXPERIMENT="cgocheck2"),
        capture_output=True,
        check=True,
        timeout=300,
    )
    kernels = accel / "mojo/kernels"
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                "--fp-mode",
                "contract=off",
                "--diagnose-missing-doc-strings",
                "--Werror",
                "-I",
                str(kernels),
                "--emit",
                "shared-lib",
                "-o",
                str(workspace / "mojo.so"),
                str(kernels / "qcfs.mojo"),
            ]
        ),
        capture_output=True,
        check=True,
        timeout=300,
    )
    return {
        "SC_NEUROCORE_QCFS_RUST_LIB": workspace / "rust/release/libsc_neurocore_qcfs.so",
        "SC_NEUROCORE_QCFS_GO_LIB": workspace / "go.so",
        "SC_NEUROCORE_QCFS_MOJO_LIB": workspace / "mojo.so",
    }


@pytest.fixture(scope="module")
def qcfs_environment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Configure the three compiled libraries and an owned offline Julia project."""
    workspace = tmp_path_factory.mktemp("qcfs-native")
    libraries = qcfs_libraries(ACCEL, workspace)
    executables = [require_julia_runtime("1.11")]
    assert executables, "installed Julia required"
    project = workspace / "julia"
    project.mkdir()
    version = importlib.metadata.version("juliacall")
    (project / "Project.toml").write_text(
        '[deps]\nPythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"\n'
        f'[compat]\nPythonCall = "={version}"\n'
    )
    settings = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        SC_NEUROCORE_QCFS_JULIA_ENABLED="1",
        PYTHON_JULIACALL_EXE=str(executables[-1]),
        PYTHON_JULIACALL_PROJECT=str(project),
        PYTHON_JULIACALL_THREADS="1",
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        JULIA_CONDAPKG_BACKEND="Null",
        JULIA_PKG_OFFLINE="true",
        **{name: str(path) for name, path in libraries.items()},
    )
    subprocess.run(
        [
            str(executables[-1]),
            f"--project={project}",
            "--startup-file=no",
            "-e",
            "using Pkg; Pkg.offline(true); Pkg.resolve()",
        ],
        env=settings,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return settings


def run_probe(
    program: str,
    settings: dict[str, str],
    label: str,
    *arguments: str,
    source: Path = ROOT / "src/sc_neurocore/conversion",
) -> str:
    """Run one probe in a fresh process, optionally measured, and return its stdout.

    Parameters
    ----------
    program:
        Python source; ``sys.argv[1]`` is the coverage data file or empty and
        ``sys.argv[2]`` the measured source directory.
    settings:
        Process environment.
    label:
        Coverage data suffix.
    arguments:
        Further probe arguments.
    source:
        Conversion package directory the probe imports and measures.

    Returns
    -------
    str
        The probe's standard output.
    """
    parent_data = os.environ.get("COVERAGE_FILE", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            f"{parent_data}-qcfs-{label}" if parent_data else "",
            str(source),
            *arguments,
        ],
        cwd=ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


_MEASURE = r"""
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False)
    measurement.start()
import atexit
def _save():
    if measurement is not None:
        measurement.stop()
        measurement.save()
atexit.register(_save)
"""

_PARITY = (
    _MEASURE
    + r"""
import numpy as np
from sc_neurocore.conversion import qcfs_backward, qcfs_forward
rng = np.random.default_rng(29)
special = [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0, 1e308, -1e308, 5e-324, -5e-324]
cases = 0
for steps in (1, 4, 8, 1000, 2**32 - 1):
    for theta in (0.25, 1.0, 3.7, 1e-300, 1e300):
        grid = np.arange(-3, min(steps, 50) + 4) * theta / steps
        x = np.concatenate([rng.normal(0, theta, 400), grid, grid + theta / (2 * steps), special])
        upstream = np.concatenate([rng.normal(0, 1, x.size - len(special)), special[::-1]])
        x, upstream = x.reshape(-1, 2), upstream.reshape(-1, 2)
        expected = qcfs_forward(x, steps, theta, backend="numpy")
        derivatives = qcfs_backward(x, upstream, steps, theta, backend="numpy")
        for backend in ("rust", "go", "mojo", "julia"):
            actual = qcfs_forward(x, steps, theta, backend=backend)
            assert actual.shape == x.shape and actual.tobytes() == expected.tobytes(), (backend, steps, theta)
            inputs, thresholds = qcfs_backward(x, upstream, steps, theta, backend=backend)
            assert inputs.tobytes() == derivatives[0].tobytes(), (backend, steps, theta)
            assert thresholds.tobytes() == derivatives[1].tobytes(), (backend, steps, theta)
            cases += x.size
for backend in ("numpy", "rust", "go", "mojo", "julia"):
    assert qcfs_forward([], backend=backend).shape == (0,)
    empty = qcfs_backward(np.empty((0, 3)), np.empty((0, 3)), backend=backend)
    assert empty[0].shape == empty[1].shape == (0, 3)
    assert qcfs_forward(0.3, backend=backend).shape == ()
    assert qcfs_forward(np.int64(1), 4, 2, backend=backend).tolist() == 1.0
from sc_neurocore.conversion.qcfs_native import load_qcfs_julia
julia = load_qcfs_julia()
x = np.array([0.5, 1.5]); upstream = np.ones(2)
output, inputs, thresholds = np.full(2, 7.0), np.full(2, 7.0), np.full(2, 7.0)
for steps, theta, source in (
    (0, 1.0, x.ctypes.data), (2**32, 1.0, x.ctypes.data), (8, 0.0, x.ctypes.data),
    (8, float("nan"), x.ctypes.data), (8, 1.0, 0), (8, 1.0, x.ctypes.data + 1),
):
    assert julia.forward(steps, theta, source, 2, output.ctypes.data) == -1
    assert julia.backward(steps, theta, source, upstream.ctypes.data, 2, inputs.ctypes.data, thresholds.ctypes.data) == -1
assert output.tolist() == inputs.tolist() == thresholds.tolist() == [7.0, 7.0]
print("five runtimes bit-identical over", cases, "elements", flush=True)
"""
)


def test_five_runtimes_return_identical_bits(qcfs_environment: dict[str, str]) -> None:
    """Every native runtime reproduces the NumPy bits, including NaN payloads and zero signs."""
    output = run_probe(_PARITY, qcfs_environment, "parity")
    assert "five runtimes bit-identical over" in output


def _same(actual: float, expected: float) -> bool:
    return bool(np.isnan(actual) and np.isnan(expected)) or (
        np.float64(actual).tobytes() == np.float64(expected).tobytes()
    )


@pytest.mark.parametrize("steps", [1, 4, 8, 1000, 2**32 - 1])
def test_reference_matches_activation_autograd(steps: int) -> None:
    """The NumPy reference equals ``QCFSActivation`` forward and autograd element by element.

    PyTorch holds the threshold as float32 before ``double()``; the reference
    receives that held value. NaN elements are compared as NaN.
    """
    rng = np.random.default_rng(steps)
    for theta in (0.25, 1.0, 3.7):
        layer = QCFSActivation(T=steps, theta=theta, learn_theta=True).double()
        held = float(layer.theta.item())
        x = np.concatenate(
            [rng.normal(0, held, 80), np.arange(-2, min(steps, 20) + 3) * held / steps]
        )
        x = np.concatenate([x, [np.inf, -np.inf, np.nan, 0.0, -0.0, 1e308, -1e308]])
        upstream = rng.normal(0, 1, x.size)
        upstream[:5] = [np.inf, -np.inf, np.nan, 0.0, -0.0]
        values = qcfs_forward(x, steps, held, backend="numpy")
        inputs, thresholds = qcfs_backward(x, upstream, steps, held, backend="numpy")
        for index in range(x.size):
            layer = QCFSActivation(T=steps, theta=theta, learn_theta=True).double()
            element = torch.tensor([x[index]], dtype=torch.float64, requires_grad=True)
            output = layer(element)
            output.backward(torch.tensor([upstream[index]], dtype=torch.float64))
            assert element.grad is not None and layer.theta.grad is not None
            assert _same(output.item(), values[index])
            assert _same(element.grad.item(), inputs[index])
            assert _same(layer.theta.grad.item(), thresholds[index])


_SELECTION = (
    _MEASURE
    + r"""
from sc_neurocore.conversion.qcfs_dispatch import select_qcfs_native
from sc_neurocore.conversion.qcfs_native import load_qcfs_julia, load_qcfs_library
import os
expected = sys.argv[3]
chosen = select_qcfs_native("auto")
if expected == "numpy":
    assert chosen is None
elif expected == "julia":
    assert chosen is load_qcfs_julia()
else:
    assert chosen is load_qcfs_library(os.environ["SC_NEUROCORE_QCFS_" + expected.upper() + "_LIB"])
print("auto selected", expected, flush=True)
"""
)


@pytest.mark.parametrize(
    "configured,expected",
    [
        (("rust", "go", "mojo", "julia"), "mojo"),
        (("rust", "go", "julia"), "rust"),
        (("go", "julia"), "go"),
        (("julia",), "julia"),
        ((), "numpy"),
    ],
)
def test_auto_takes_the_first_configured_runtime(
    qcfs_environment: dict[str, str], configured: tuple[str, ...], expected: str
) -> None:
    """Auto follows Mojo, Rust, Go, Julia over what is configured, then NumPy."""
    settings = dict(qcfs_environment)
    for name in ("rust", "go", "mojo"):
        if name not in configured:
            settings.pop(f"SC_NEUROCORE_QCFS_{name.upper()}_LIB")
    if "julia" not in configured:
        settings["SC_NEUROCORE_QCFS_JULIA_ENABLED"] = "0"
    assert f"auto selected {expected}" in run_probe(
        _SELECTION, settings, f"auto-{expected}", expected
    )


_REFUSALS = (
    _MEASURE
    + r"""
import numpy as np
from sc_neurocore.conversion import qcfs_backward, qcfs_forward
for backend, message in (
    ("rust", "requires SC_NEUROCORE_QCFS_RUST_LIB"),
    ("go", "requires SC_NEUROCORE_QCFS_GO_LIB"),
    ("mojo", "requires SC_NEUROCORE_QCFS_MOJO_LIB"),
    ("julia", "requires SC_NEUROCORE_QCFS_JULIA_ENABLED=1"),
):
    try:
        qcfs_forward([0.5], backend=backend)
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(backend + " ran without configuration")
import os
os.environ["SC_NEUROCORE_QCFS_JULIA_ENABLED"] = "2"
try:
    qcfs_forward([0.5])
except RuntimeError as error:
    assert "opt-in must be 0 or 1" in str(error)
else:
    raise AssertionError("invalid Julia opt-in accepted")
assert qcfs_forward([0.5], backend="numpy").tolist() == [0.5]
print("unconfigured runtimes refused", flush=True)
"""
)


def test_unconfigured_runtimes_are_refused_not_replaced(qcfs_environment: dict[str, str]) -> None:
    """An explicit runtime without configuration fails instead of falling back to NumPy."""
    settings = {key: value for key, value in qcfs_environment.items() if key not in LIBRARIES}
    settings["SC_NEUROCORE_QCFS_JULIA_ENABLED"] = "0"
    assert "unconfigured runtimes refused" in run_probe(_REFUSALS, settings, "unconfigured")


@pytest.mark.parametrize(
    "arguments,error,message",
    [
        ({"steps": 0}, ValueError, "positive integer no larger than 2\\*\\*32 - 1"),
        ({"steps": 2**32}, ValueError, "positive integer no larger than 2\\*\\*32 - 1"),
        ({"steps": True}, ValueError, "positive integer"),
        ({"steps": 8.0}, ValueError, "positive integer"),
        ({"theta": 0.0}, ValueError, "finite and positive"),
        ({"theta": -1.0}, ValueError, "finite and positive"),
        ({"theta": float("nan")}, ValueError, "finite and positive"),
        ({"theta": float("inf")}, ValueError, "finite and positive"),
        ({"theta": True}, ValueError, "finite and positive"),
        ({"theta": "1.0"}, ValueError, "finite and positive"),
        ({"theta": 10**400}, ValueError, "finite and positive"),
        ({"backend": "cuda"}, ValueError, "unsupported QCFS backend"),
    ],
)
def test_parameters_are_admitted_before_any_runtime(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    """Grids, thresholds and backend names outside the shared domain are refused."""
    with pytest.raises(error, match=message):
        qcfs_forward([0.5], **arguments)


@pytest.mark.parametrize(
    "values", [[True, False], [1 + 2j], np.array(["a"], dtype=object), ["0.5"]]
)
def test_non_real_activations_are_refused(values: Any) -> None:
    """Boolean, complex, object and text activations are refused, not coerced."""
    with pytest.raises(TypeError, match="real integer or floating"):
        qcfs_forward(values)
    with pytest.raises(TypeError, match="real integer or floating"):
        qcfs_backward([0.5], values)


def test_upstream_shape_must_equal_input_shape() -> None:
    """Derivatives are never broadcast across a mismatched upstream gradient."""
    with pytest.raises(ValueError, match="shape of x"):
        qcfs_backward(np.zeros((2, 2)), np.zeros(4))


def test_native_array_contract_refuses_without_writing(qcfs_environment: dict[str, str]) -> None:
    """Each C library refuses invalid grids, thresholds and spans and leaves outputs untouched."""
    for variable in LIBRARIES:
        api = load_qcfs_library(qcfs_environment[variable])
        x = np.array([0.5, 1.5])
        upstream = np.ones(2)
        output = np.full(2, 7.0)
        inputs, thresholds = np.full(2, 7.0), np.full(2, 7.0)
        misaligned = x.ctypes.data + 1
        for steps, theta, source in (
            (0, 1.0, x.ctypes.data),
            (8, 0.0, x.ctypes.data),
            (8, float("nan"), x.ctypes.data),
            (8, float("inf"), x.ctypes.data),
            (8, 1.0, 0),
            (8, 1.0, misaligned),
        ):
            assert api.forward(steps, theta, source, 2, output.ctypes.data) == -1, variable
            assert (
                api.backward(
                    steps,
                    theta,
                    source,
                    upstream.ctypes.data,
                    2,
                    inputs.ctypes.data,
                    thresholds.ctypes.data,
                )
                == -1
            ), variable
        assert api.forward(8, 1.0, x.ctypes.data, 2**62, output.ctypes.data) == -1, variable
        assert (
            api.backward(8, 1.0, x.ctypes.data, 0, 2, inputs.ctypes.data, thresholds.ctypes.data)
            == -1
        )
        assert output.tolist() == inputs.tolist() == thresholds.tolist() == [7.0, 7.0], variable
        assert api.forward(8, 1.0, 0, 0, 0) == 0, variable
        assert api.forward(8, 1.0, x.ctypes.data, 2, x.ctypes.data) == 0, variable
        assert x.tolist() == [0.5, 1.0], variable


_FOREIGN = r"""
#include <stddef.h>
#include <stdint.h>
uint32_t sc_qcfs_abi_version(void) { return SC_ABI; }
int32_t sc_qcfs_forward(uint32_t s, double t, const double *x, size_t n, double *o) {
    (void)s; (void)t; (void)x; (void)n; (void)o; return SC_STATUS;
}
#if SC_BACKWARD
int32_t sc_qcfs_backward(uint32_t s, double t, const double *x, const double *g, size_t n,
                         double *i, double *h) {
    (void)s; (void)t; (void)x; (void)g; (void)n; (void)i; (void)h; return SC_STATUS;
}
#endif
"""


@pytest.mark.parametrize(
    "variant,defines,message",
    [
        (
            "abi-two",
            {"SC_ABI": "2", "SC_STATUS": "0", "SC_BACKWARD": "1"},
            "incompatible array ABI",
        ),
        (
            "no-backward",
            {"SC_ABI": "1", "SC_STATUS": "0", "SC_BACKWARD": "0"},
            "entry points unavailable",
        ),
        (
            "refusing",
            {"SC_ABI": "1", "SC_STATUS": "-1", "SC_BACKWARD": "1"},
            "refused admitted input",
        ),
    ],
)
def test_public_calls_refuse_a_foreign_library(
    tmp_path: Path, variant: str, defines: dict[str, str], message: str
) -> None:
    """A real library breaking the QCFS contract is refused through the public API.

    The maintained libraries cannot break the contract by construction, so a C
    library compiled here supplies each breach; nothing in Python is replaced.
    """
    source = tmp_path / "foreign.c"
    source.write_text(_FOREIGN)
    library = tmp_path / f"{variant}.so"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Werror",
            *[f"-D{k}={v}" for k, v in defines.items()],
            "-o",
            str(library),
            str(source),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    program = (
        _MEASURE
        + r"""
from sc_neurocore.conversion import qcfs_backward, qcfs_forward
for call in (lambda: qcfs_forward([0.5], backend="rust"), lambda: qcfs_backward([0.5], [1.0], backend="rust")):
    try:
        call()
    except RuntimeError as error:
        print("refused:", error, flush=True)
    else:
        raise AssertionError("foreign QCFS library accepted")
"""
    )
    settings = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), SC_NEUROCORE_QCFS_RUST_LIB=str(library)
    )
    output = run_probe(program, settings, f"foreign-{variant}")
    assert output.count(message) == 2, output
