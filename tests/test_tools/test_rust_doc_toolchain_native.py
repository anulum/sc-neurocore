# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the Rust documentation measurement runs the compiler its crate pins

"""Bind the Rust documentation measurement to the toolchain the measured crate pins.

The measurement starts Cargo from the repository root, where rustup finds no
toolchain file for a member crate, and it compiles the original Git revision in a
temporary directory, which no rustup directory override reaches. Both used to
run whatever compiler the host defaulted to. A crate that pins its toolchain is
now measured with that toolchain in every compiler call, whatever the caller's
environment says, and a pin that cannot run is refused by name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.test_tools.test_rust_doc_measurement_native import rust_project
from tools.rust_doc_history import LEGACY_SCHEMA, protect_rust_debt
from tools.rust_doc_measurement import RustMeasurementError, measure_rust_findings
from tools.rust_doc_symbols import NativeParser, build_parser
from tools.rust_toolchain_pin import (
    RustToolchainPinError,
    ToolchainPin,
    compiler_environment,
    pinned_toolchain,
    reported_version_matches,
    toolchain_environment,
)

REPO = Path(__file__).resolve().parents[2]
#: A toolchain name that rustup will not find installed on any host.
ABSENT = "0.0.1"


@pytest.fixture(autouse=True)
def local_git_fixture_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the fixture repositories' baseline local, not the hosting job's CI event.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Removes the CI event identity for the test.
    """
    for name in (
        "GITHUB_ACTIONS",
        "GITHUB_SHA",
        "GITHUB_EVENT_PATH",
        "GITHUB_EVENT_NAME",
        "SC_NEUROCORE_POLICY_BASELINE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def parser() -> NativeParser:
    """Build the actual Rust source identity producer used with real Cargo findings."""
    return build_parser(REPO)


@pytest.fixture(scope="module")
def engine_channel() -> str:
    """Return the toolchain the repository's own engine pins, which every Rust host installs."""
    pin = pinned_toolchain(REPO, "engine/Cargo.toml")
    assert pin is not None and pin.source == "engine/rust-toolchain.toml"
    return pin.channel


def _commit(root: Path, message: str) -> None:
    """Commit the whole fixture tree with a local identity."""
    for argv in (
        ["git", "add", "--all"],
        [
            "git",
            "-c",
            "user.name=Native contract",
            "-c",
            "user.email=native@example.invalid",
            "commit",
            "-qm",
            message,
        ],
    ):
        subprocess.run(argv, cwd=root, capture_output=True, check=True, timeout=30)


def _pinned_project(root: Path, channel: str) -> Path:
    """Create a real Cargo library whose root pins ``channel`` in rust-toolchain.toml."""
    root = rust_project(root, "//! Native crate.\npub fn old() {}\n")
    (root / "rust-toolchain.toml").write_text(f'[toolchain]\nchannel = "{channel}"\n')
    _commit(root, "pin the toolchain")
    return root


def _write(root: Path, files: dict[str, str]) -> Path:
    """Write a manifest and the given files below ``root`` and return ``root``."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


class TestThePin:
    """The pin is read from the files rustup would read, nearest the manifest first."""

    def test_a_crate_without_a_toolchain_file_has_no_pin(self, tmp_path: Path) -> None:
        """No file means the caller's toolchain, as before."""
        _write(tmp_path, {"engine/Cargo.toml": ""})
        assert pinned_toolchain(tmp_path, "engine/Cargo.toml") is None

    def test_the_manifest_directory_pin_is_read(self, tmp_path: Path) -> None:
        """A member crate's own rust-toolchain.toml names the toolchain and its file."""
        _write(
            tmp_path,
            {
                "engine/Cargo.toml": "",
                "engine/rust-toolchain.toml": '[toolchain]\nchannel = "1.98.1"\ncomponents = ["clippy"]\n',
            },
        )
        assert pinned_toolchain(tmp_path, "engine/Cargo.toml") == ToolchainPin(
            "1.98.1", "engine/rust-toolchain.toml"
        )

    def test_a_root_pin_applies_to_a_member_without_its_own(self, tmp_path: Path) -> None:
        """Directories are searched from the manifest's up to the repository root."""
        _write(
            tmp_path,
            {"engine/Cargo.toml": "", "rust-toolchain.toml": '[toolchain]\nchannel = "stable"\n'},
        )
        assert pinned_toolchain(tmp_path, "engine/Cargo.toml") == ToolchainPin(
            "stable", "rust-toolchain.toml"
        )

    def test_the_nearest_pin_wins(self, tmp_path: Path) -> None:
        """The member's own pin is taken before the root's."""
        _write(
            tmp_path,
            {
                "engine/Cargo.toml": "",
                "engine/rust-toolchain.toml": '[toolchain]\nchannel = "1.98.1"\n',
                "rust-toolchain.toml": '[toolchain]\nchannel = "1.99.1"\n',
            },
        )
        assert pinned_toolchain(tmp_path, "engine/Cargo.toml") == ToolchainPin(
            "1.98.1", "engine/rust-toolchain.toml"
        )

    @pytest.mark.parametrize(
        ("text", "channel"),
        [
            ("nightly-2026-08-18\n", "nightly-2026-08-18"),
            ("1.98.1-x86_64-unknown-linux-gnu", "1.98.1-x86_64-unknown-linux-gnu"),
            ('[toolchain]\nchannel = "1.98.1"\n', "1.98.1"),
        ],
    )
    def test_the_legacy_file_holds_a_name_or_toml(
        self, tmp_path: Path, text: str, channel: str
    ) -> None:
        """A legacy rust-toolchain file holds a bare toolchain name or the TOML form.

        Parameters
        ----------
        tmp_path : Path
            Fixture repository root.
        text : str
            Contents of the legacy file.
        channel : str
            The toolchain the file names.
        """
        _write(tmp_path, {"Cargo.toml": "", "rust-toolchain": text})
        assert pinned_toolchain(tmp_path, "Cargo.toml") == ToolchainPin(channel, "rust-toolchain")

    @pytest.mark.parametrize(
        ("files", "message"),
        [
            (
                {
                    "rust-toolchain": "1.98.1\n",
                    "rust-toolchain.toml": '[toolchain]\nchannel = "1.98.1"\n',
                },
                "Two Rust toolchain files in the repository root pin the measurement; keep one.",
            ),
            (
                {"rust-toolchain.toml": '[toolchain]\npath = "/opt/rust"\n'},
                "rust-toolchain.toml names a toolchain by path; the measurement needs a named "
                "channel.",
            ),
            (
                {"rust-toolchain.toml": '[toolchain]\ncomponents = ["clippy"]\n'},
                "rust-toolchain.toml names no toolchain channel.",
            ),
            (
                {"rust-toolchain.toml": "[toolchain]\nchannel = 1.98\n"},
                "rust-toolchain.toml names no toolchain channel.",
            ),
            (
                {"rust-toolchain.toml": 'channel = "1.98.1"\n'},
                "rust-toolchain.toml names no toolchain channel.",
            ),
            (
                {"rust-toolchain.toml": '[toolchain\nchannel = "1.98.1"\n'},
                "rust-toolchain.toml is not valid TOML.",
            ),
            (
                {"rust-toolchain.toml": '[toolchain]\nchannel = "1.98.1; rm -rf /"\n'},
                "rust-toolchain.toml names the toolchain '1.98.1; rm -rf /', which is not a "
                "toolchain name.",
            ),
            (
                {"rust-toolchain": "1.98.1\nstable\n"},
                "rust-toolchain must hold one toolchain name.",
            ),
            (
                {"rust-toolchain": "\n"},
                "rust-toolchain must hold one toolchain name.",
            ),
            (
                {"rust-toolchain.toml": b"\xff\xfe".decode("latin-1")},
                "rust-toolchain.toml is not valid TOML.",
            ),
        ],
    )
    def test_a_pin_that_names_no_usable_toolchain_is_refused(
        self, tmp_path: Path, files: dict[str, str], message: str
    ) -> None:
        """Every unreadable or ambiguous pin is refused with its whole reason.

        Parameters
        ----------
        tmp_path : Path
            Fixture repository root.
        files : dict
            Toolchain files written beside a root manifest.
        message : str
            The refusal expected, whole.
        """
        _write(tmp_path, {"Cargo.toml": "", **files})
        with pytest.raises(RustToolchainPinError) as caught:
            pinned_toolchain(tmp_path, "Cargo.toml")
        assert str(caught.value) == message

    def test_a_pin_that_is_not_utf8_is_refused(self, tmp_path: Path) -> None:
        """A legacy file whose bytes are not text names nothing."""
        _write(tmp_path, {"Cargo.toml": ""})
        (tmp_path / "rust-toolchain").write_bytes(b"\xff\xfe1.98.1\n")
        with pytest.raises(RustToolchainPinError) as caught:
            pinned_toolchain(tmp_path, "Cargo.toml")
        assert str(caught.value) == "rust-toolchain is not UTF-8 text."


class TestTheEnvironment:
    """A pin overrides whatever toolchain the caller's environment names."""

    def test_a_pin_names_the_toolchain_and_forbids_installing_one(self) -> None:
        """The caller's toolchain, compiler, wrappers and auto-install choice do not reach a pin."""
        caller = {
            "PATH": "/usr/bin",
            "RUSTUP_TOOLCHAIN": "1.99.1",
            "RUSTUP_AUTO_INSTALL": "1",
            "RUSTC": "/opt/other/rustc",
            "RUSTC_WRAPPER": "sccache",
            "RUSTC_WORKSPACE_WRAPPER": "clippy-driver",
            "CARGO_BUILD_RUSTC": "/opt/other/rustc",
            "CARGO_BUILD_RUSTC_WRAPPER": "sccache",
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER": "clippy-driver",
        }
        environment = toolchain_environment(ToolchainPin("1.98.1", "rust-toolchain.toml"), caller)
        assert environment == {
            "PATH": "/usr/bin",
            "RUSTUP_TOOLCHAIN": "1.98.1",
            "RUSTUP_AUTO_INSTALL": "0",
            "RUSTC_WRAPPER": "",
            "RUSTC_WORKSPACE_WRAPPER": "",
        }
        assert caller["RUSTUP_TOOLCHAIN"] == "1.99.1" and caller["RUSTC"] == "/opt/other/rustc"

    def test_the_compiler_executable_is_named_in_a_copy(self) -> None:
        """Cargo is told which compiler executable to run, over any build.rustc setting."""
        caller = {"PATH": "/usr/bin"}
        environment = compiler_environment(caller, Path("/toolchains/1.98.1/bin/rustc"))
        assert environment == {"PATH": "/usr/bin", "RUSTC": "/toolchains/1.98.1/bin/rustc"}
        assert caller == {"PATH": "/usr/bin"}

    def test_no_pin_leaves_the_caller_environment_as_it_is(self) -> None:
        """Without a pin the measurement runs as the caller set it up, in a copy."""
        caller = {"PATH": "/usr/bin", "RUSTUP_TOOLCHAIN": "1.99.1"}
        environment = toolchain_environment(None, caller)
        assert environment == caller and environment is not caller


class TestTheReportedVersion:
    """A numbered pin must be the compiler that actually ran."""

    @pytest.mark.parametrize(
        ("channel", "reported", "matches"),
        [
            ("1.98.1", "rustc 1.98.1 (48a229cea 2026-09-01)", True),
            ("1.98.1", "rustc 1.99.1 (58cf8f9c3 2026-10-13)", False),
            ("1.98.1", "rustc 1.98.10 (0 2027-01-01)", False),
            ("1.98", "rustc 1.98.1 (48a229cea 2026-09-01)", True),
            ("1.98", "rustc 1.99.0 (0 2026-10-01)", False),
            ("1.98.1-x86_64-unknown-linux-gnu", "rustc 1.98.1 (48a229cea 2026-09-01)", True),
            ("stable", "rustc 1.99.1 (58cf8f9c3 2026-10-13)", True),
            ("nightly-2026-08-18", "rustc 1.100.0-nightly (0 2026-08-17)", True),
            ("1.99.0", "rustc 1.99.0-beta.3 (0 2026-09-20)", False),
            ("1.98", "rustc 1.98.0-beta.1 (0 2026-07-20)", False),
            ("1.98.1", "", False),
        ],
    )
    def test_numbered_channels_are_compared_and_named_ones_are_not(
        self, channel: str, reported: str, matches: bool
    ) -> None:
        """A named channel is resolved by rustup; a number is checked against rustc itself.

        Parameters
        ----------
        channel : str
            The pinned toolchain.
        reported : str
            What ``rustc --version`` printed.
        matches : bool
            Whether the measurement may continue.
        """
        assert (
            reported_version_matches(ToolchainPin(channel, "rust-toolchain.toml"), reported)
            is matches
        )


class TestTheMeasurement:
    """Real Cargo and rustup run the pinned toolchain in every call of a measurement."""

    def test_the_pin_wins_over_a_caller_toolchain_that_cannot_run(
        self,
        parser: NativeParser,
        engine_channel: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller environment naming an absent toolchain does not reach a pinned crate.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        engine_channel : str
            The toolchain the repository's engine pins.
        tmp_path : Path
            Fixture directory.
        monkeypatch : pytest.MonkeyPatch
            Sets the hostile caller environment.
        """
        root = _pinned_project(tmp_path / "repo", engine_channel)
        monkeypatch.setenv("RUSTUP_TOOLCHAIN", ABSENT)
        monkeypatch.setenv("RUSTUP_AUTO_INSTALL", "0")
        measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert measured.toolchain_pin == ToolchainPin(engine_channel, "rust-toolchain.toml")
        assert reported_version_matches(measured.toolchain_pin, measured.rustc_version)
        assert measured.cargo_version.startswith("cargo ")
        assert measured.undocumented == 1
        assert measured.to_public_dict()["toolchain_pin"] == {
            "channel": engine_channel,
            "source": "rust-toolchain.toml",
        }
        measured.verify(root)

    def test_a_caller_compiler_or_wrapper_does_not_reach_a_pinned_compilation(
        self,
        parser: NativeParser,
        engine_channel: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cargo runs the pinned toolchain's own compiler, whatever compiler the caller names.

        Every variable through which a caller can make Cargo run another program is
        set to a program that always fails; the measurement still completes.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        engine_channel : str
            The toolchain the fixture pins.
        tmp_path : Path
            Fixture directory.
        monkeypatch : pytest.MonkeyPatch
            Sets the caller environment.
        """
        failing = shutil.which("false")
        assert failing is not None
        root = _pinned_project(tmp_path / "repo", engine_channel)
        for name in (
            "RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
        ):
            monkeypatch.setenv(name, failing)
        measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert measured.toolchain_pin == ToolchainPin(engine_channel, "rust-toolchain.toml")
        assert reported_version_matches(measured.toolchain_pin, measured.rustc_version)
        assert measured.undocumented == 1
        measured.verify(root)

    def test_a_cargo_configuration_compiler_does_not_reach_a_pinned_compilation(
        self, parser: NativeParser, engine_channel: str, tmp_path: Path
    ) -> None:
        """A build.rustc or wrapper in the crate's Cargo configuration cannot replace the pin.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        engine_channel : str
            The toolchain the fixture pins.
        tmp_path : Path
            Fixture directory.
        """
        failing = shutil.which("false")
        assert failing is not None
        root = _pinned_project(tmp_path / "repo", engine_channel)
        (root / ".cargo").mkdir()
        (root / ".cargo/config.toml").write_text(
            f'[build]\nrustc = "{failing}"\nrustc-wrapper = "{failing}"\n'
            f'rustc-workspace-wrapper = "{failing}"\n'
        )
        _commit(root, "a configuration that names another compiler")
        measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert measured.toolchain_pin == ToolchainPin(engine_channel, "rust-toolchain.toml")
        assert reported_version_matches(measured.toolchain_pin, measured.rustc_version)
        assert measured.undocumented == 1

    def test_a_crate_without_a_pin_runs_the_caller_compiler(
        self, parser: NativeParser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a pin the caller's environment reaches Cargo unchanged, as before.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        tmp_path : Path
            Fixture directory.
        monkeypatch : pytest.MonkeyPatch
            Names a compiler that always fails.
        """
        failing = shutil.which("false")
        assert failing is not None
        root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
        monkeypatch.setenv("RUSTC", failing)
        with pytest.raises(RustMeasurementError, match="measurement failed"):
            measure_rust_findings(root, "Cargo.toml", parser=parser)

    def test_a_crate_without_a_pin_records_none(self, parser: NativeParser, tmp_path: Path) -> None:
        """An unpinned crate is measured with the caller's toolchain and says so."""
        root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
        measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert measured.toolchain_pin is None
        assert measured.to_public_dict()["toolchain_pin"] is None

    def test_a_pinned_toolchain_that_is_not_installed_is_refused_by_name(
        self, parser: NativeParser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An absent pinned toolchain is named, and rustup is not allowed to fetch it.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        tmp_path : Path
            Fixture directory.
        monkeypatch : pytest.MonkeyPatch
            Asks rustup to install missing toolchains, which the measurement overrides.
        """
        root = _pinned_project(tmp_path / "repo", ABSENT)
        monkeypatch.setenv("RUSTUP_AUTO_INSTALL", "1")
        with pytest.raises(RustMeasurementError) as caught:
            measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert str(caught.value) == (
            f"The Rust toolchain {ABSENT} pinned by rust-toolchain.toml does not run here; "
            f"install it with: rustup toolchain install {ABSENT}"
        )

    def test_a_compiler_that_ignores_the_pin_is_refused(
        self,
        parser: NativeParser,
        engine_channel: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A compiler on the path that is not rustup's proxy cannot stand in for the pinned one.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        engine_channel : str
            The toolchain whose own binaries are put first on the path.
        tmp_path : Path
            Fixture directory.
        monkeypatch : pytest.MonkeyPatch
            Replaces the search path.
        """
        sysroot = subprocess.run(
            ["rustc", "--print", "sysroot"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env={**os.environ, "RUSTUP_TOOLCHAIN": engine_channel},
        ).stdout.strip()
        reported = subprocess.run(
            [str(Path(sysroot) / "bin" / "rustc"), "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
        root = _pinned_project(tmp_path / "repo", "1.0.0")
        git = shutil.which("git")
        assert git is not None
        commands = tmp_path / "git-only"
        commands.mkdir()
        (commands / "git").symlink_to(git)
        monkeypatch.setenv("PATH", os.pathsep.join([str(Path(sysroot) / "bin"), str(commands)]))
        with pytest.raises(RustMeasurementError) as caught:
            measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert str(caught.value) == (
            "rust-toolchain.toml pins Rust 1.0.0, but the compiler that ran reports "
            f"'{reported}'; the measurement runs only with the pinned compiler."
        )

    def test_a_changed_pin_invalidates_a_measurement(
        self, parser: NativeParser, engine_channel: str, tmp_path: Path
    ) -> None:
        """The pin file is a measured input; editing it makes an earlier measurement stale.

        Parameters
        ----------
        parser : NativeParser
            The compiled syntax producer.
        engine_channel : str
            The toolchain the fixture pins.
        tmp_path : Path
            Fixture directory.
        """
        root = _pinned_project(tmp_path / "repo", engine_channel)
        measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
        (root / "rust-toolchain.toml").write_text(
            f'[toolchain]\nchannel = "{engine_channel}"\ncomponents = ["clippy"]\n'
        )
        with pytest.raises(RustMeasurementError, match="inputs changed"):
            measured.verify(root)

    def test_an_unreadable_pin_stops_the_measurement(
        self, parser: NativeParser, tmp_path: Path
    ) -> None:
        """A pin the measurement cannot read is a measurement error, not the caller's toolchain."""
        root = _pinned_project(tmp_path / "repo", "1.98.1")
        (root / "rust-toolchain.toml").write_text('[toolchain]\npath = "/opt/rust"\n')
        _commit(root, "a path toolchain")
        with pytest.raises(RustMeasurementError) as caught:
            measure_rust_findings(root, "Cargo.toml", parser=parser)
        assert str(caught.value) == (
            "rust-toolchain.toml names a toolchain by path; the measurement needs a named channel."
        )


def test_the_original_revision_is_measured_with_its_own_pin(
    parser: NativeParser, engine_channel: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original Git image in a temporary directory compiles with the revision's pinned toolchain.

    This is the case that refused pushes: no rustup directory override reaches the
    temporary image, so the original was compiled with the host's default compiler
    and its version did not reproduce the ceiling's producer.

    Parameters
    ----------
    parser : NativeParser
        The compiled syntax producer.
    engine_channel : str
        The toolchain the fixture pins.
    tmp_path : Path
        Fixture directory.
    monkeypatch : pytest.MonkeyPatch
        Sets a caller environment whose toolchain cannot run.
    """
    root = _pinned_project(tmp_path / "repo", engine_channel)
    produced = measure_rust_findings(root, "Cargo.toml", parser=parser)
    ceiling = root / "ceiling.json"
    ceiling.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_SCHEMA,
                "undocumented": produced.undocumented,
                "undocumented_files": produced.files,
                "provenance": {"rustc": produced.rustc_version},
            }
        )
    )
    _commit(root, "original measured ceiling")
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", ABSENT)
    monkeypatch.setenv("RUSTUP_AUTO_INSTALL", "0")
    current = measure_rust_findings(root, "Cargo.toml", parser=parser)
    protected = protect_rust_debt(root, current, ceiling, parser=parser)
    assert protected.original.rustc_version == produced.rustc_version
    assert protected.original.toolchain_pin == ToolchainPin(engine_channel, "rust-toolchain.toml")
    assert protected.original.cases == current.cases
