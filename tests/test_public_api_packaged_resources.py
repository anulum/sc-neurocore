# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Packaged HDL and schema resource contracts

"""Verify the public wheel resource boundary and HDL resource helpers."""

from fnmatch import fnmatchcase
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from tests.public_api_support import _package_root, _project_metadata

# Installed dataset, conversion and adaptive-threshold adapters own these
# runtime resources. Other neuron/research native trees stay source-only.
RUNTIME_ASSETS = (
    "accel/julia/datasets/*.jl",
    "accel/julia/conversion/*.jl",
    "accel/julia/neurons/adaptive_threshold_if.jl",
    "accel/go/go.mod",
    "accel/go/adaptive_threshold_if/*.go",
    "accel/go/adaptive_threshold_if/abi.h",
    "accel/go/conversion/*.go",
    "accel/go/conversion/cshared/*.go",
    "accel/go/conversion/cshared/abi.h",
    "accel/go/conversion/qcfscshared/*.go",
    "accel/mojo/kernels/ann_to_snn*.mojo",
    "accel/mojo/kernels/qcfs.mojo",
    "accel/mojo/kernels/adaptive_threshold_if.mojo",
    "accel/mojo/adaptive_threshold_if/*.mojo",
)
REQUIRED_NATIVE_ASSETS = {
    "accel/julia/datasets/shd_cli.jl",
    "accel/julia/conversion/ann_to_snn_native.jl",
    "accel/julia/neurons/adaptive_threshold_if.jl",
    "accel/go/go.mod",
    "accel/go/adaptive_threshold_if/adaptive_threshold_if.go",
    "accel/go/conversion/cshared/main.go",
    "accel/go/conversion/qcfscshared/main.go",
    "accel/mojo/kernels/ann_to_snn_native.mojo",
    "accel/mojo/kernels/qcfs.mojo",
    "accel/mojo/kernels/adaptive_threshold_if.mojo",
    "accel/mojo/adaptive_threshold_if/adaptive_threshold_if.mojo",
}


def _tree_state(root: Path) -> dict[str, tuple[bool, int, int]]:
    """Map every path under ``root`` to whether it is a folder, its size and its modification time."""
    state = {}
    for path in root.rglob("*"):
        info = path.lstat()
        state[path.relative_to(root).as_posix()] = (path.is_dir(), info.st_size, info.st_mtime_ns)
    return state


def _config_value(path: Path) -> str:
    """Write ``path`` as a config value; setuptools' parser reads ``%`` as interpolation."""
    return str(path).replace("%", "%%")


def _outside_build_config(work: Path) -> Path:
    """Write a setuptools config that keeps the build and egg-info folders under ``work``.

    In place, setuptools reuses ``build/lib`` and copies into the wheel every
    file a former build left there, even one no longer in the sources; it also
    reads an existing ``SOURCES.txt``. Folders of their own under ``work`` give
    the build a clean start, as on a fresh checkout, and leave the project
    untouched.
    """
    egg_base = work / "egg-info"
    egg_base.mkdir()
    config = work / "build_outside_the_project.cfg"
    config.write_text(
        f"[build]\nbuild_base = {_config_value(work / 'build')}\n\n"
        f"[egg_info]\negg_base = {_config_value(egg_base)}\n",
        encoding="utf-8",
    )
    return config


def _build_wheel(project: Path, work: Path) -> Path:
    """Build ``project``'s wheel in the empty folder ``work`` and return its one wheel file."""
    wheel_dir = work / "wheel"
    wheel_dir.mkdir()
    environment = {**os.environ, "DIST_EXTRA_CONFIG": str(_outside_build_config(work))}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(project),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=environment,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_hdl_install_profile_packages_source_artefacts() -> None:
    """The HDL install profile ships the baseline RTL source artefacts."""
    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]["sc_neurocore"]

    assert "hardware/*.v" in package_data
    assert "hdl/primitives/*.v" in package_data
    assert "hdl_gen/safety/*.sv" in package_data
    assert "hdl_gen/openroad_flow/*.sh" in package_data

    package_root = _package_root()
    matched = {
        str(path.relative_to(package_root))
        for pattern in package_data
        for path in package_root.glob(pattern)
    }

    assert "hardware/microtubule_neuron.v" in matched
    assert "hdl/primitives/sc_bitstream_encoder.v" in matched
    assert "hdl/primitives/sc_bitstream_synapse.v" in matched
    assert "hdl/primitives/sc_dense_layer_core.v" in matched
    assert "hdl/primitives/sc_dotproduct_to_current.v" in matched
    assert "hdl/primitives/sc_firing_rate_bank.v" in matched
    assert "hdl/primitives/sc_lif_neuron.v" in matched
    assert "hdl_gen/safety/safety_monitor.sv" in matched
    assert "hdl_gen/openroad_flow/run_asic_flow.sh" in matched


def test_dsl_schema_assets_are_packaged() -> None:
    """The wheel ships UniversalNeuron schema assets used by bare-name loading."""
    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]["sc_neurocore"]

    assert "neurons/model_schemas/*.json" in package_data
    assert "neurons/model_schemas/*.toml" in package_data

    package_root = _package_root()
    matched = {
        str(path.relative_to(package_root))
        for pattern in package_data
        for path in package_root.glob(pattern)
    }

    assert "neurons/model_schemas/perfect_integrator.json" in matched
    assert "neurons/model_schemas/perfect_integrator.toml" in matched
    assert "neurons/model_schemas/sc_perfect_integrator.json" in matched
    assert "neurons/model_schemas/sc_perfect_integrator.toml" in matched
    assert "neurons/model_schemas/lif.toml" in matched


def test_hdl_resource_helper_rejects_unknown_primitive_names() -> None:
    """The HDL resource helper rejects traversal and unknown primitive names."""
    from sc_neurocore.hdl.resources import baseline_primitive_text, list_baseline_primitive_rtl

    names = list_baseline_primitive_rtl()

    assert names == (
        "sc_bitstream_encoder.v",
        "sc_bitstream_synapse.v",
        "sc_dense_layer_core.v",
        "sc_dense_matrix_layer.v",
        "sc_dotproduct_to_current.v",
        "sc_firing_rate_bank.v",
        "sc_lif_neuron.v",
        "sc_dcls_layer_core.v",
        "sc_dcls_tent_kernel.v",
        "sc_dcls_axonal_delay.v",
        "sc_graph_forward.v",
        "sc_softmax_attention.v",
        "sc_kuramoto_step.v",
    )
    assert "module sc_bitstream_encoder" in baseline_primitive_text("sc_bitstream_encoder.v")
    try:
        baseline_primitive_text("../sc_bitstream_encoder.v")
    except ValueError as exc:
        assert "Unknown baseline HDL primitive" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("path traversal primitive name was accepted")


def test_hdl_resource_helper_contract_and_missing_packaged_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HDL resource helper reports missing packaged primitive files."""
    from sc_neurocore.hdl import resources as hdl_resources

    names = hdl_resources.list_baseline_primitive_rtl()
    assert isinstance(names, tuple)
    assert len(set(names)) == len(names)

    class _MissingResource:
        """Minimal resource object that simulates a missing packaged file."""

        class _File:
            """Minimal file-like resource with a negative file predicate."""

            def is_file(self) -> bool:
                """Report that the resource does not exist as a file."""
                return False

        def joinpath(self, _name: str) -> _File:
            """Return a missing child resource for any primitive name."""
            return self._File()

    monkeypatch.setattr(hdl_resources, "files", lambda _pkg: _MissingResource())
    with pytest.raises(FileNotFoundError, match="Missing packaged HDL primitive"):
        hdl_resources.baseline_primitive_path("sc_lif_neuron.v")


def test_packaged_primitives_match_the_canonical_emitter_dependencies() -> None:
    """The public resource catalogue supplies the exact maintained source of every core."""
    from sc_neurocore.hdl.resources import baseline_primitive_text, list_baseline_primitive_rtl

    root = Path(__file__).resolve().parents[1]
    for name in list_baseline_primitive_rtl():
        assert baseline_primitive_text(name) == (root / "hdl" / name).read_text(encoding="utf-8")


def test_base_wheel_does_not_package_polyglot_research_sources(tmp_path: Path) -> None:
    """Build the public wheel and confine native assets to installed runtime dependencies."""
    root = Path(__file__).resolve().parents[1]
    built = _build_wheel(root, tmp_path)
    assert built.name.startswith("sc_neurocore-")
    with ZipFile(built) as wheel:
        from sc_neurocore.hdl.resources import list_baseline_primitive_rtl

        for name in list_baseline_primitive_rtl():
            assert (
                wheel.read("sc_neurocore/hdl/primitives/" + name)
                == (root / "hdl" / name).read_bytes()
            )
        native = {
            name.removeprefix("sc_neurocore/")
            for name in wheel.namelist()
            if name.startswith(
                ("sc_neurocore/accel/julia/", "sc_neurocore/accel/go/", "sc_neurocore/accel/mojo/")
            )
            and not name.endswith(".py")
        }
        assert REQUIRED_NATIVE_ASSETS.issubset(native)
        for name in native:
            assert any(fnmatchcase(name, pattern) for pattern in RUNTIME_ASSETS), name
            assert (
                wheel.read("sc_neurocore/" + name)
                == (root / "src/sc_neurocore" / name).read_bytes()
            )


def test_wheel_build_leaves_the_project_and_its_stale_build_folder_alone(tmp_path: Path) -> None:
    """A file left in the project's ``build/lib`` stays out of the wheel, and the build writes nothing there."""
    project = tmp_path / "project"
    package = project / "src" / "stale_probe"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "stale-probe"\nversion = "0"\n\n'
        '[tool.setuptools.package-data]\nstale_probe = ["*.txt"]\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text('"""A package for the build probe."""\n', encoding="utf-8")
    (package / "kept.txt").write_text("in the sources\n", encoding="utf-8")
    stale = project / "build" / "lib" / "stale_probe" / "removed.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("removed from the sources, left by an earlier build\n", encoding="utf-8")
    before = _tree_state(project)
    # A space and a percent sign: the config must carry any folder name.
    work = tmp_path / "work at 100%"
    work.mkdir()

    with ZipFile(_build_wheel(project, work)) as wheel:
        names = set(wheel.namelist())

    assert "stale_probe/kept.txt" in names
    assert "stale_probe/removed.txt" not in names
    assert _tree_state(project) == before
