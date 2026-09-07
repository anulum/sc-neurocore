# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A record names the compiler that built the lane

"""A benchmark record must name the compiler that built the lane it measures.

Two probes were asking the wrong question. `GOTOOLCHAIN` resolves a Go
toolchain per module, so `go version` at the repository root reported the
installed Go — 1.24.0 on this host — while a build inside `accel/go` uses the
1.26.7 its `go.mod` requires and hosted CI pins. And `mojo` on `PATH` is
whatever is installed there, a prerelease here, while the build and CI both use
the version `accel/mojo/pixi.toml` pins.

Seven committed records were published naming compilers that did not build
their lanes. The cases below hold each probe to the pin the repository declares,
so a record cannot quietly name the wrong one again.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks import bench_brunel_wang, bench_compte_wm
from benchmarks import _non_resetting_lif_benchmark as shared

REPOSITORY = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"
GO_MOD = REPOSITORY / "src" / "sc_neurocore" / "accel" / "go" / "go.mod"
PIXI_MANIFEST = REPOSITORY / "src" / "sc_neurocore" / "accel" / "mojo" / "pixi.toml"

#: Every module carrying a toolchain probe this unit corrected.
PROBES = (shared, bench_compte_wm, bench_brunel_wang)


def _declared_go_version() -> str:
    """Return the Go version the accelerator module requires."""
    match = re.search(r"^go (\S+)$", GO_MOD.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    return match.group(1)


def _pinned_mojo_version() -> str:
    """Return the Mojo version the pixi manifest pins."""
    manifest = tomllib.loads(PIXI_MANIFEST.read_text(encoding="utf-8"))
    for section in manifest.values():
        if isinstance(section, dict) and "mojo" in section:
            return str(section["mojo"]).lstrip("=")
    raise AssertionError("pixi manifest declares no mojo dependency")


class TestTheProbesPointAtTheDeclaredToolchain:
    @pytest.mark.parametrize("module", PROBES, ids=lambda m: m.__name__)
    def test_the_go_probe_asks_inside_the_accelerator_module(self, module: ModuleType) -> None:
        """At the repository root the answer is the installed Go, not the build's."""
        go_module = module.GO_MODULE
        assert go_module == GO_MOD.parent
        assert (go_module / "go.mod").is_file()

    @pytest.mark.parametrize("module", PROBES, ids=lambda m: m.__name__)
    def test_the_mojo_probe_prefers_the_pinned_manifest(self, module: ModuleType) -> None:
        """`mojo` on PATH is whatever is installed; the pin is what builds."""
        command = module._mojo_command()
        if "pixi" in command[0]:
            assert str(PIXI_MANIFEST) in command
        else:
            assert command[0].endswith("mojo")


class TestTheDeclaredToolchainMatchesHostedCI:
    def test_the_go_module_pin_is_the_one_ci_installs(self) -> None:
        """A record is only comparable with CI if both name one compiler."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        assert f'go-version: "{_declared_go_version()}"' in workflow

    def test_the_mojo_pin_is_the_manifest_ci_uses(self) -> None:
        """CI exposes mojo through this manifest; the probe must use the same."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "src/sc_neurocore/accel/mojo/pixi.toml" in workflow
        assert _pinned_mojo_version()


class TestTheProbesAnswerWithTheDeclaredVersions:
    def test_go_reports_the_version_the_module_requires(self) -> None:
        """The measurement this unit exists for: root said 1.24.0, module says 1.26.7."""
        reported = shared._version(shared._toolchain_command("go", "version"), cwd=shared.GO_MODULE)
        if reported == "unavailable":
            pytest.skip("no Go toolchain on this host")
        assert _declared_go_version() in reported

    def test_mojo_reports_the_pinned_version(self) -> None:
        """PATH carried a prerelease; the pin is what the lane was built with."""
        reported = shared._version(shared._mojo_command())
        if reported == "unavailable":
            pytest.skip("no Mojo toolchain on this host")
        assert _pinned_mojo_version() in reported
