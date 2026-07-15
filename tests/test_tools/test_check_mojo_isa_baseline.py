# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for the Mojo ISA-baseline drift gate
"""Exercise the AVX-512 drift gate over clean and drifted disassembly."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_mojo_isa_baseline.py"
_spec = importlib.util.spec_from_file_location("check_mojo_isa_baseline", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
MOD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MOD)

# A portable x86-64-v3 kernel: AVX2 ``ymm`` only, no AVX-512.
_CLEAN = """
0000000000001b90 <coba_lif_simulate_c>:
    1b90:  c5 fd 10 06    vmovupd (%rsi),%ymm0
    1b94:  c5 fd 58 c1    vaddpd  %ymm1,%ymm0,%ymm0
    1b98:  c3             ret
"""

# A drifted kernel: AVX-512 opmask + zmm — SIGILLs on runners without AVX-512F.
_DRIFTED = """
0000000000001b90 <coba_lif_simulate_c>:
    1701:  c5 fb 93 c0    kmovd   %k0,%eax
    1710:  62 f1 fd 48 10 06  vmovupd (%rsi),%zmm0
    1720:  c3             ret
"""


def test_scan_disassembly_clean_is_empty() -> None:
    assert MOD.scan_disassembly(_CLEAN) == []


def test_scan_disassembly_flags_opmask_and_zmm() -> None:
    hits = MOD.scan_disassembly(_DRIFTED)
    assert any("%k0" in line for line in hits)
    assert any("%zmm0" in line for line in hits)
    assert len(hits) == 2


def test_scan_disassembly_ignores_avx2_ymm() -> None:
    assert MOD.scan_disassembly("    vaddpd %ymm1,%ymm0,%ymm0") == []


def test_discover_libraries_recurses_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "libnested.so").write_bytes(b"")
    (tmp_path / "liba.so").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    found = MOD.discover_libraries(tmp_path)
    assert {p.name for p in found} == {
        "liba.so",
        "libnested.so",
    }  # rglob recursion + lib*.so filter
    assert found == sorted(found)  # stable, deterministic order


def test_discover_libraries_excludes_pixi_toolchain(tmp_path: Path) -> None:
    (tmp_path / ".pixi" / "envs" / "default" / "lib").mkdir(parents=True)
    (tmp_path / ".pixi" / "envs" / "default" / "lib" / "libMojoLLDB.so").write_bytes(b"")
    (tmp_path / "libcoba_lif.so").write_bytes(b"")
    found = MOD.discover_libraries(tmp_path)
    assert [p.name for p in found] == ["libcoba_lif.so"]  # our kernel only, not the pixi toolchain


def test_scan_library_skips_unreadable(tmp_path: Path) -> None:
    not_an_object = tmp_path / "libbroken.so"
    not_an_object.write_text("not an ELF object")
    assert MOD.scan_library(not_an_object) == []  # objdump failure -> skip, no crash


def test_check_maps_only_offenders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "libclean.so").write_bytes(b"")
    (tmp_path / "libdrift.so").write_bytes(b"")

    def fake_scan(library: Path) -> list[str]:
        return ["kmovd %k0,%eax"] if library.name == "libdrift.so" else []

    monkeypatch.setattr(MOD, "scan_library", fake_scan)
    offenders = MOD.check(tmp_path)
    assert list(offenders) == [tmp_path / "libdrift.so"]


def test_main_clean_tree_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "libclean.so").write_bytes(b"")
    monkeypatch.setattr(MOD, "scan_library", lambda _library: [])
    assert MOD.main(["--root", str(tmp_path)]) == 0


def test_main_drift_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "libdrift.so").write_bytes(b"")
    monkeypatch.setattr(MOD, "scan_library", lambda _library: ["kmovd %k0,%eax"])
    assert MOD.main(["--root", str(tmp_path)]) == 1


def test_main_missing_tree_is_noop() -> None:
    assert MOD.main(["--root", "/nonexistent/mojo/tree"]) == 0


def test_main_empty_tree_is_noop(tmp_path: Path) -> None:
    assert MOD.main(["--root", str(tmp_path)]) == 0
