# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (main_cli) from former test_build_accel_backends.py

from __future__ import annotations

from build_accel_backends_support import *  # noqa: F403


def test_main_all_ok(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: _stub_targets(MOD, ["theta"]))
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "go"]) == 0
    assert "built 1/1" in capsys.readouterr().out


def test_main_required_failure_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: _stub_targets(MOD, ["theta"]))
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, False, "exit 1"))
    assert MOD.main(["--language", "go", "--require", "theta"]) == 1


def test_main_required_never_discovered_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: _stub_targets(MOD, ["theta"]))
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "go", "--require", "adex"]) == 1


def test_main_accepts_shared_name_for_ermentrout_mojo_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MOD,
        "discover_targets",
        lambda lang, **k: _stub_targets(MOD, ["ermentrout_kopell_map_neuron"]),
    )
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "mojo", "--require", "ermentrout_kopell_map"]) == 0


def test_main_all_languages_and_mojo_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_discover(language: str, **kwargs: Any) -> list[Any]:
        seen.append(language)
        return _stub_targets(MOD, [f"{language}_only"])

    monkeypatch.setattr(MOD, "discover_targets", fake_discover)
    monkeypatch.setattr(MOD, "build_target", lambda t, **k: MOD.BuildResult(t, True, "ok"))
    assert MOD.main(["--language", "all", "--mojo-command", "pixi run mojo"]) == 0
    assert seen == ["go", "mojo"]


def test_main_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_accel_backends.py", "--language", "go"])
    monkeypatch.setattr(MOD, "discover_targets", lambda lang, **k: [])
    assert MOD.main() == 0
