# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense IF comparison distribution acceptance

"""Build real distributions, then capture and admit measured ordering from the installed wheel.

The installed wheel alone must be enough: its shipped Rust, Go and Mojo IF sources
build the native libraries, its bundled comparison command measures all five
runtimes, and its public dispatch admits the report it produced as well as a
report captured in the source checkout from byte-identical sources.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from sc_neurocore.conversion.if_benchmark_record import validated_timing_order
from tests.test_bench_ann_to_snn_replay import (
    ROOT,
    environment,
    go_library,
    mojo_library,
    rust_library,
)
from tests.test_conversion_measured_order import comparison
from tests.test_studio_distribution import (
    CONVERSION_BENCHMARK,
    build_distribution,
    distribution_source,
    installation_environment,
)

__all__ = [
    "comparison",
    "distribution_source",
    "environment",
    "go_library",
    "mojo_library",
    "rust_library",
]

RESOURCES = "sc_neurocore/conversion/benchmark_resources"
ENTRY_POINT = "sc-neurocore-if-benchmark = sc_neurocore.conversion.if_benchmark_cli:main"

_CAPTURE = r"""
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False
    )
    measurement.start()
from importlib.metadata import entry_points
(command,) = entry_points(group="console_scripts", name="sc-neurocore-if-benchmark")
try:
    status = command.load()(sys.argv[3:])
finally:
    if measurement is not None:
        measurement.stop()
        measurement.save()
raise SystemExit(status)
"""

_ADMIT = r"""
import json
import os
import sys
from pathlib import Path
measurement = None
if sys.argv[2]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[2], branch=True, source=[sys.argv[3]], config_file=False
    )
    measurement.start()
import sc_neurocore
assert Path(sc_neurocore.__file__).is_relative_to(Path(sys.argv[1])), sc_neurocore.__file__
from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_benchmark_order import measured_order
from sc_neurocore.conversion.if_dispatch import select_native
from sc_neurocore.conversion.if_native import load_native
expected = tuple(json.loads(sys.argv[4]))
assert measured_order() == expected, (measured_order(), expected)
assert ConvertedSNN([[[1.0]]], [None], [1.0], T=7).run([1.0]).tolist() == [7.0]
if expected[0] == "julia":
    from sc_neurocore.conversion.if_julia import load_julia
    wanted = load_julia()
else:
    wanted = load_native(os.environ["SC_NEUROCORE_IF_" + expected[0].upper() + "_LIB"])
assert select_native("auto").library is wanted.library
assert "torch" not in sys.modules
if measurement is not None:
    measurement.stop()
    measurement.save()
print("installed measured admission passed", flush=True)
"""


def built_wheel(source: Path, workspace: Path, *, through_sdist: bool = False) -> Path:
    """Build one wheel from a disposable source tree, optionally through its sdist.

    Parameters
    ----------
    source:
        Source tree containing the real build inputs.
    workspace:
        Directory receiving the distributions and any unpacked sdist.
    through_sdist:
        Build the sdist first and the wheel from its unpacked contents.

    Returns
    -------
    pathlib.Path
        The single built wheel.
    """
    if through_sdist:
        result = build_distribution(source, "build_sdist", workspace / "sdist")
        assert result.returncode == 0, result.stdout + result.stderr
        (archive_path,) = (workspace / "sdist").glob("*.tar.gz")
        with tarfile.open(archive_path) as archive:
            archive.extractall(workspace / "unpacked", filter="data")
        (source,) = (workspace / "unpacked").iterdir()
    result = build_distribution(source, "build_wheel", workspace / "wheel")
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = (workspace / "wheel").glob("*.whl")
    return wheel_path


def native_libraries(accel: Path, workspace: Path) -> dict[str, Path]:
    """Build the Rust, Go and Mojo IF libraries from one ``accel`` tree's own sources.

    Parameters
    ----------
    accel:
        The ``sc_neurocore/accel`` directory whose shipped sources are built.
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
            str(accel / "rust/safety/if_native/Cargo.toml"),
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
            # An installed package is not a module checkout; its VCS stamp would
            # describe whatever repository happens to enclose the environment.
            "-buildvcs=false",
            "-buildmode=c-shared",
            "-o",
            str(workspace / "go.so"),
            "./conversion/cshared",
        ],
        cwd=accel / "go",
        env=dict(os.environ, GOEXPERIMENT="cgocheck2"),
        capture_output=True,
        check=True,
        timeout=300,
    )
    kernels = accel / "mojo/kernels"
    subprocess.run(
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
            str(kernels / "ann_to_snn_native.mojo"),
        ],
        capture_output=True,
        check=True,
        timeout=300,
    )
    return {
        "SC_NEUROCORE_IF_RUST_LIB": workspace / "rust/release/libsc_neurocore_if_replay.so",
        "SC_NEUROCORE_IF_GO_LIB": workspace / "go.so",
        "SC_NEUROCORE_IF_MOJO_LIB": workspace / "mojo.so",
    }


def installed_settings(environment: dict[str, str], libraries: dict[str, Path]) -> dict[str, str]:
    """Return the configured runtime environment without checkout import paths or tracing.

    Parameters
    ----------
    environment:
        Actual configured provider and offline Julia settings.
    libraries:
        Native libraries to configure in place of the checkout builds.

    Returns
    -------
    dict of str to str
        Settings under which only the installed ``sc_neurocore`` is importable.
    """
    settings = dict(environment)
    for key in (
        "PYTHONPATH",
        "COVERAGE_PROCESS_START",
        "COVERAGE_FILE",
        "SC_NEUROCORE_IF_BENCHMARK",
    ):
        settings.pop(key, None)
    settings.update({name: str(path) for name, path in libraries.items()})
    return settings


@pytest.fixture(scope="module")
def installed(distribution_source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract a wheel built from the real source tree into a package-only directory."""
    workspace = tmp_path_factory.mktemp("if-distribution")
    wheel_path = built_wheel(distribution_source, workspace)
    target = workspace / "installed"
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(target)
    return target


@pytest.mark.parametrize("through_sdist", [False, True], ids=["wheel", "sdist-wheel"])
def test_both_build_paths_ship_the_comparison_command(
    distribution_source: Path, tmp_path: Path, through_sdist: bool
) -> None:
    """The wheel carries the exact comparison scripts, build declaration and console command."""
    wheel_path = built_wheel(distribution_source, tmp_path, through_sdist=through_sdist)
    with zipfile.ZipFile(wheel_path) as wheel:
        shipped = sorted(name for name in wheel.namelist() if name.startswith(RESOURCES + "/"))
        expected = [*CONVERSION_BENCHMARK, "pyproject.toml"]
        assert shipped == sorted(f"{RESOURCES}/{Path(name).name}" for name in expected)
        for name in expected:
            source = distribution_source / name
            assert wheel.read(f"{RESOURCES}/{source.name}") == source.read_bytes()
        (entry_points,) = [name for name in wheel.namelist() if name.endswith("entry_points.txt")]
        assert ENTRY_POINT in wheel.read(entry_points).decode().splitlines()


def test_build_refuses_a_source_without_the_comparison_script(
    distribution_source: Path, tmp_path: Path
) -> None:
    """A source tree lacking a comparison script fails instead of shipping a broken command."""
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    (source / CONVERSION_BENCHMARK[2]).unlink()
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode != 0
    assert "Missing conversion benchmark source" in result.stdout + result.stderr


def test_rebuild_replaces_changed_and_stray_comparison_resources(
    distribution_source: Path, tmp_path: Path
) -> None:
    """A reused build directory never carries a stale script or a file no longer shipped."""
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    first = build_distribution(source, "build_wheel", tmp_path / "first")
    assert first.returncode == 0, first.stdout + first.stderr
    stray = source / "build/lib" / RESOURCES / "withdrawn.py"
    stray.write_text("raise SystemExit(1)\n", encoding="utf-8")
    script = source / CONVERSION_BENCHMARK[0]
    script.write_bytes(script.read_bytes() + b"\n# revised\n")
    os.utime(script, (1, 1))
    second = build_distribution(source, "build_wheel", tmp_path / "second")
    assert second.returncode == 0, second.stdout + second.stderr
    (wheel_path,) = (tmp_path / "second").glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        assert f"{RESOURCES}/withdrawn.py" not in wheel.namelist()
        assert wheel.read(f"{RESOURCES}/{script.name}") == script.read_bytes()


def test_installed_wheel_reproduces_and_admits_measured_order(
    installed: Path,
    environment: dict[str, str],
    comparison: Path,
    tmp_path: Path,
) -> None:
    """Build, measure and admit from the wheel alone; admit the checkout's report as well."""
    python = installation_environment(installed, tmp_path / "environment")
    libraries = native_libraries(installed / "sc_neurocore/accel", tmp_path / "native")
    settings = installed_settings(environment, libraries)
    report = tmp_path / "installed-comparison.json"
    parent_data = os.environ.get("COVERAGE_FILE", "")
    result = subprocess.run(
        [
            str(python),
            "-c",
            _CAPTURE,
            f"{parent_data}-installed-capture" if parent_data else "",
            str(installed / "sc_neurocore/conversion"),
            "--output",
            str(report),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
            "--samples",
            "3",
            "--warmup",
            "1",
        ],
        cwd=tmp_path,
        env=settings,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(report.read_text())
    checkout = json.loads(comparison.read_text())
    assert record["meta"]["coverage_process_start_requested"] is False
    assert record["source_sha256"] == checkout["source_sha256"]
    for name, path in libraries.items():
        assert record["artifact_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
    julia = {key: value for key, value in record["artifact_sha256"].items() if "JULIACALL" in key}
    assert julia and all(checkout["artifact_sha256"][key] == value for key, value in julia.items())
    for label, measured, configured in (
        ("installed", record, settings),
        ("checkout", checkout, installed_settings(environment, {})),
    ):
        path = report if label == "installed" else comparison
        admission = subprocess.run(
            [
                str(python),
                "-c",
                _ADMIT,
                str(installed),
                f"{parent_data}-installed-{label}" if parent_data else "",
                str(installed / "sc_neurocore/conversion"),
                json.dumps(validated_timing_order(measured)),
            ],
            cwd=tmp_path,
            env=dict(configured, SC_NEUROCORE_IF_BENCHMARK=str(path)),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert admission.returncode == 0, label + admission.stdout + admission.stderr
        assert "installed measured admission passed" in admission.stdout


def test_installed_command_refuses_a_package_without_its_resources(
    installed: Path, tmp_path: Path
) -> None:
    """A package whose comparison resources are absent refuses before running anything."""
    stripped = tmp_path / "stripped"
    shutil.copytree(installed, stripped, ignore=shutil.ignore_patterns("benchmark_resources"))
    python = installation_environment(stripped, tmp_path / "environment")
    parent_data = os.environ.get("COVERAGE_FILE", "")
    result = subprocess.run(
        [
            str(python),
            "-c",
            _CAPTURE,
            f"{parent_data}-installed-stripped" if parent_data else "",
            str(stripped / "sc_neurocore/conversion"),
            "--output",
            str(tmp_path / "report.json"),
            "--cpu",
            "0",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "the complete dense IF comparison resource is absent" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_checkout_command_runs_the_owning_script(tmp_path: Path) -> None:
    """From a source checkout the command runs the owning script and returns its status."""
    output = tmp_path / "report.json"
    output.write_text("prior owner report")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sc_neurocore.conversion.if_benchmark_cli",
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
            "--samples",
            "2",
        ],
        cwd=ROOT,
        env=dict(
            os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "at least three samples and one warmup required" in result.stderr
    assert output.read_text() == "prior owner report"


_STRICT_CONSUMER = """
import numpy as np
import numpy.typing as npt

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_benchmark_cli import main
from sc_neurocore.conversion.if_benchmark_order import measured_order


def rates(model: ConvertedSNN, drive: list[float]) -> npt.NDArray[np.float64]:
    return model.run(drive, input_mode="constant", backend="numpy")


def compare(arguments: list[str]) -> int:
    return main(arguments)


order: tuple[str, ...] = measured_order()
"""

_MISTYPED_CONSUMER = """
from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_benchmark_order import measured_order

fastest: int = measured_order()
ConvertedSNN([[[1.0]]], [None], [1.0], T=7).run([1.0], backend="cuda")
"""


@pytest.mark.parametrize("consumer", ["typed", "mistyped"])
def test_installed_wheel_types_a_strict_consumer(
    installed: Path, tmp_path: Path, consumer: str
) -> None:
    """A downstream module checks under strict mypy against the wheel's own inline types."""
    python = installation_environment(installed, tmp_path / "environment")
    source = tmp_path / "consumer.py"
    source.write_text(_STRICT_CONSUMER if consumer == "typed" else _MISTYPED_CONSUMER)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--python-executable",
            str(python),
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "--config-file",
            os.devnull,
            str(source),
        ],
        cwd=tmp_path,
        env={
            key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "MYPYPATH")
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    if consumer == "typed":
        assert result.returncode == 0, result.stdout + result.stderr
        return
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        'Incompatible types in assignment (expression has type "tuple[str, ...]"' in result.stdout
    )
    assert 'Argument "backend"' in result.stdout and "Literal['cuda']" in result.stdout
    assert "sc_neurocore" not in result.stdout.replace("sc_neurocore.conversion", "")


_SKEWED_JULIA = r"""
import sys
from pathlib import Path
measurement = None
if sys.argv[2]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[2], branch=True, source=[sys.argv[3]], config_file=False
    )
    measurement.start()
import sc_neurocore
assert Path(sc_neurocore.__file__).is_relative_to(Path(sys.argv[1])), sc_neurocore.__file__
from sc_neurocore.conversion import ConvertedSNN
try:
    ConvertedSNN([[[1.0]]], [None], [1.0], T=7).run([1.0], backend="julia")
except RuntimeError as error:
    assert "Julia IF managed runtime unavailable or incompatible" in str(error), str(error)
    assert "incompatible ownership ABI" in str(error.__cause__), repr(error.__cause__)
else:
    raise AssertionError("a Julia boundary reporting another ABI was accepted")
finally:
    if measurement is not None:
        measurement.stop()
        measurement.save()
print("skewed Julia boundary refused", flush=True)
"""


def test_installed_julia_boundary_from_another_abi_is_refused(
    installed: Path, environment: dict[str, str], tmp_path: Path
) -> None:
    """A package whose shipped Julia boundary reports another ABI is refused, not replayed.

    The wheel now ships the Julia boundary, so a version-skewed installation is a
    real state: the copy's boundary source says ABI two while its Python admits one.
    """
    skewed = tmp_path / "abi-mismatch"
    shutil.copytree(installed, skewed)
    boundary = skewed / "sc_neurocore/accel/julia/conversion/ann_to_snn_native.jl"
    text = boundary.read_text()
    assert text.count("sc_if_abi_version()::UInt32 = 1") == 1
    boundary.write_text(
        text.replace("sc_if_abi_version()::UInt32 = 1", "sc_if_abi_version()::UInt32 = 2")
    )
    python = installation_environment(skewed, tmp_path / "environment")
    parent_data = os.environ.get("COVERAGE_FILE", "")
    result = subprocess.run(
        [
            str(python),
            "-c",
            _SKEWED_JULIA,
            str(skewed),
            f"{parent_data}-julia-abi-mismatch" if parent_data else "",
            str(skewed / "sc_neurocore/conversion"),
        ],
        cwd=tmp_path,
        env=installed_settings(environment, {}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skewed Julia boundary refused" in result.stdout
