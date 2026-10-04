# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The Rust documentation ratchet only turns one way

"""A ratchet that can be raised is not a ratchet.

The owner directive of 2026-09-06 asks that documentation debt be enforced so
it cannot grow. The failure mode is not subtle and it is not rare: a ceiling
that the tooling updates in whichever direction the measurement moved records
whatever happened, and enforces nothing.

So the one property these cases exist for is that the ceiling falls and never
rises by machine. The rest hold the boundary a check must respect when it
cannot run at all: a missing ceiling record, a malformed one and an absent
toolchain each stop the check rather than pass it, because a check that reports
success when it measured nothing is worse than no check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_tools.test_rust_doc_measurement_native import rust_project
from tools.rust_doc_ratchet import (
    RUST_DOC_CEILING_SCHEMA_VERSION,
    RatchetError,
    compare,
    main,
    measure,
    read_ceiling,
    write_ceiling,
)

PROVENANCE = {
    "argv": "cargo rustc --lib --manifest-path engine/Cargo.toml -- -W missing_docs",
    "rustc": "rustc 1.98.1",
    "source_sha256": "0" * 40,
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def ceiling_at(path: Path, value: int) -> Path:
    """Write a ceiling record holding one figure."""
    write_ceiling(path, undocumented=value, files=1, provenance=PROVENANCE)
    return path


class TestTheRatchetOnlyTurnsOneWay:
    """The shared ceiling accepts equal or lower debt and rejects automatic increases."""

    def test_it_lowers_the_ceiling_when_debt_falls(self, tmp_path: Path) -> None:
        """Verify a lower measurement replaces the existing private ceiling."""
        path = ceiling_at(tmp_path / "ceiling.json", 100)

        write_ceiling(path, undocumented=90, files=1, provenance=PROVENANCE)

        assert read_ceiling(path) == 90

    def test_it_refuses_to_raise_the_ceiling(self, tmp_path: Path) -> None:
        """A rise is a decision, and it belongs in a diff somebody signed."""
        path = ceiling_at(tmp_path / "ceiling.json", 100)

        with pytest.raises(RatchetError, match="refusing to raise"):
            write_ceiling(path, undocumented=101, files=1, provenance=PROVENANCE)

        assert read_ceiling(path) == 100

    def test_an_unchanged_figure_is_accepted(self, tmp_path: Path) -> None:
        """Verify an equal measurement preserves the existing ceiling value."""
        path = ceiling_at(tmp_path / "ceiling.json", 100)

        write_ceiling(path, undocumented=100, files=1, provenance=PROVENANCE)

        assert read_ceiling(path) == 100

    def test_a_first_ceiling_may_be_any_figure(self, tmp_path: Path) -> None:
        """There is nothing to ratchet against until one exists."""
        path = tmp_path / "ceiling.json"

        write_ceiling(path, undocumented=3873, files=297, provenance=PROVENANCE)

        assert read_ceiling(path) == 3873


class TestTheRecord:
    """Ceiling records retain the contract, counts, producer metadata and direction."""

    def test_it_carries_its_contract_and_what_produced_it(self, tmp_path: Path) -> None:
        """Verify serialized counts and producer metadata retain their contract fields."""
        path = ceiling_at(tmp_path / "ceiling.json", 7)

        document = json.loads(path.read_text(encoding="utf-8"))

        assert document["schema_version"] == RUST_DOC_CEILING_SCHEMA_VERSION
        assert document["undocumented"] == 7
        assert document["undocumented_files"] == 1
        assert document["provenance"]["rustc"] == "rustc 1.98.1"
        assert "missing_docs" in document["provenance"]["argv"]

    def test_it_says_which_way_the_ratchet_turns(self, tmp_path: Path) -> None:
        """Someone will read this file before they read the tool."""
        path = ceiling_at(tmp_path / "ceiling.json", 7)

        assert "may not rise" in json.loads(path.read_text(encoding="utf-8"))["note"]


class TestTheVerdict:
    """Comparison results explain unchanged, falling and increasing documentation debt."""

    def test_debt_below_the_ceiling_passes_and_asks_for_an_update(self) -> None:
        """Verify falling debt passes and explains the available ceiling update."""
        verdict = compare(90, 100)

        assert verdict.ok is True
        assert "fell" in verdict.summary()
        assert "--update" in verdict.summary()

    def test_debt_above_the_ceiling_fails_and_says_by_how_much(self) -> None:
        """Verify growing debt fails with the exact increase in its verdict."""
        verdict = compare(105, 100)

        assert verdict.ok is False
        assert "+5" in verdict.summary()

    def test_debt_at_the_ceiling_passes_without_asking_for_anything(self) -> None:
        """Verify equal debt passes with an unchanged verdict."""
        verdict = compare(100, 100)

        assert verdict.ok is True
        assert "unchanged" in verdict.summary()


class TestACheckThatCannotRun:
    """Unavailable or invalid ceiling inputs prevent a usable acceptance verdict."""

    def test_a_missing_ceiling_record_stops_the_check(self, tmp_path: Path) -> None:
        """Verify an absent ceiling prevents a usable debt comparison."""
        with pytest.raises(RatchetError, match="no ceiling record"):
            read_ceiling(tmp_path / "absent.json")

    def test_a_record_without_a_figure_stops_the_check(self, tmp_path: Path) -> None:
        """Verify a record without a whole-number ceiling is refused."""
        path = tmp_path / "ceiling.json"
        path.write_text(json.dumps({"note": "nothing here"}), encoding="utf-8")

        with pytest.raises(RatchetError, match="no whole-number ceiling"):
            read_ceiling(path)

    def test_a_negative_figure_stops_the_check(self, tmp_path: Path) -> None:
        """Verify negative stored debt is refused as an invalid ceiling."""
        path = tmp_path / "ceiling.json"
        path.write_text(json.dumps({"undocumented": -1}), encoding="utf-8")

        with pytest.raises(RatchetError, match="no whole-number ceiling"):
            read_ceiling(path)

    def test_an_absent_toolchain_reports_an_error_rather_than_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A genuinely empty executable search path cannot produce a native measurement."""
        monkeypatch.setenv("PATH", "")
        code = main(["--repo", str(tmp_path), "--ceiling", str(tmp_path / "c.json")])
        assert code == 2
        assert "cargo is not installed" in capsys.readouterr().out


def _native_project(root: Path, debt: int) -> Path:
    """Commit a dependency-free real Rust crate with the requested original public debt."""
    source = "//! Native original public crate.\n" + "\n".join(
        f"pub fn item_{index}() {{}}" for index in range(debt)
    )
    return rust_project(root, source)


@pytest.fixture(autouse=True)
def local_git_fixture_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Separate actual fixture Git commits from the outer hosted CI checkout metadata."""
    for name in ("GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_EVENT_PATH", "GITHUB_EVENT_NAME"):
        monkeypatch.delenv(name, raising=False)


class TestTheCommandLine:
    """CLI dispatch preserves actual debt comparison, safe updates and input refusal."""

    def test_it_passes_when_debt_is_at_the_ceiling(self, tmp_path: Path) -> None:
        """Compare actual native original debt with an equal private scalar ceiling."""
        root = _native_project(tmp_path / "repo", 10)
        path = ceiling_at(tmp_path / "ceiling.json", 10)
        assert main(["--repo", str(root), "--manifest", "Cargo.toml", "--ceiling", str(path)]) == 0

    def test_it_fails_when_debt_rose(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Eleven actual compiler cases exceed a private ten-case scalar ceiling."""
        root = _native_project(tmp_path / "repo", 11)
        path = ceiling_at(tmp_path / "ceiling.json", 10)
        assert main(["--repo", str(root), "--manifest", "Cargo.toml", "--ceiling", str(path)]) == 1
        assert "rose" in capsys.readouterr().out

    def test_update_lowers_the_ceiling(self, tmp_path: Path) -> None:
        """Document six original functions and lower the native allowance from ten to four."""
        root = _native_project(tmp_path / "repo", 10)
        source = root / "src/lib.rs"
        text = source.read_text()
        for index in range(4, 10):
            text = text.replace(
                f"pub fn item_{index}",
                f"/// Return from original item {index}.\npub fn item_{index}",
            )
        source.write_text(text)
        path = ceiling_at(tmp_path / "ceiling.json", 10)
        assert (
            main(
                [
                    "--repo",
                    str(root),
                    "--manifest",
                    "Cargo.toml",
                    "--ceiling",
                    str(path),
                    "--update",
                ]
            )
            == 0
        )
        assert read_ceiling(path) == 4

    def test_update_refuses_to_raise_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Actual twelve-case debt cannot raise the recorded ten-case private ceiling."""
        root = _native_project(tmp_path / "repo", 12)
        path = ceiling_at(tmp_path / "ceiling.json", 10)
        assert (
            main(
                [
                    "--repo",
                    str(root),
                    "--manifest",
                    "Cargo.toml",
                    "--ceiling",
                    str(path),
                    "--update",
                ]
            )
            == 2
        )
        assert "refusing to raise" in capsys.readouterr().out
        assert read_ceiling(path) == 10

    def test_a_malformed_record_fails_the_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A successful actual compiler measurement cannot authorize malformed ceiling JSON."""
        root = _native_project(tmp_path / "repo", 1)
        path = tmp_path / "ceiling.json"
        path.write_text("{not json", encoding="utf-8")
        assert main(["--repo", str(root), "--manifest", "Cargo.toml", "--ceiling", str(path)]) == 2
        assert "error:" in capsys.readouterr().out


class TestRunningTheLint:
    """Require the public measurement API to use the actual compiler and syntax producer."""

    def test_it_returns_the_count_the_reader_found_and_the_toolchain_version(
        self, tmp_path: Path
    ) -> None:
        """Resolve actual missing function and struct docs across two compiled Rust files."""
        root = rust_project(
            tmp_path / "repo",
            "//! Native crate.\n/// Native module.\npub mod other;\npub fn entry() {}\n",
            extra={"other.rs": "pub struct Item;\n"},
        )
        undocumented, files, version = measure(root, "Cargo.toml")
        assert (undocumented, files) == (2, 2)
        assert version.startswith("rustc 1.98.1")

    def test_it_refuses_when_cargo_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An actual empty executable search path makes Cargo unavailable to the public API."""
        monkeypatch.setenv("PATH", "")
        with pytest.raises(RatchetError, match="cargo is not installed"):
            measure(tmp_path, "Cargo.toml")
