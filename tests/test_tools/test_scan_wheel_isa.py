# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for the third-party wheel ISA report-only scanner
"""Exercise the unguarded-AVX-512 wheel scanner over clean/dispatched/unguarded objects."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "scan_wheel_isa.py"
_spec = importlib.util.spec_from_file_location("scan_wheel_isa", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
MOD = importlib.util.module_from_spec(_spec)
# Register before exec so ``@dataclass`` can resolve ``LibraryScan.__module__`` in sys.modules.
sys.modules["scan_wheel_isa"] = MOD
_spec.loader.exec_module(MOD)

# AVX2 ``ymm`` only — no AVX-512, no runtime detection needed.
_CLEAN = """
0000000000001b90 <plain>:
    1b90:  c5 fd 10 06    vmovupd (%rsi),%ymm0
    1b98:  c3             ret
"""

# AVX-512 ``zmm``/opmask behind a runtime ``cpuid`` feature check.
_DISPATCHED = """
0000000000001000 <dispatch>:
    1000:  0f a2          cpuid
    1010:  62 f1 fd 48 10 06  vmovupd (%rsi),%zmm0
    1018:  c5 fb 93 c0    kmovd   %k1,%eax
    1020:  c3             ret
"""

# AVX-512 ``zmm``/opmask with no ``cpuid`` anywhere — static, SIGILLs without AVX-512F.
_UNGUARDED = """
0000000000002000 <static>:
    2000:  62 f1 fd 48 10 06  vmovupd (%rsi),%zmm0
    2008:  c5 fb 93 c0    kmovd   %k1,%eax
    2010:  c3             ret
"""


def test_scan_disassembly_counts_clean() -> None:
    assert MOD.scan_disassembly(_CLEAN) == (0, 0)


def test_scan_disassembly_counts_dispatched() -> None:
    avx512, cpuid = MOD.scan_disassembly(_DISPATCHED)
    assert avx512 == 2  # zmm line + opmask line
    assert cpuid == 1


def test_scan_disassembly_counts_unguarded() -> None:
    avx512, cpuid = MOD.scan_disassembly(_UNGUARDED)
    assert avx512 == 2
    assert cpuid == 0


def test_scan_disassembly_ignores_avx2_ymm() -> None:
    assert MOD.scan_disassembly("    vaddpd %ymm1,%ymm0,%ymm0") == (0, 0)


def test_library_scan_risk_levels() -> None:
    lib = Path("x.so")
    assert MOD.LibraryScan(lib, avx512_hits=0, cpuid_hits=5).risk == "clean"
    assert MOD.LibraryScan(lib, avx512_hits=9, cpuid_hits=3).risk == "dispatched"
    assert MOD.LibraryScan(lib, avx512_hits=9, cpuid_hits=0).risk == "unguarded"


def test_scan_library_parses_objdump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lib = tmp_path / "libx.so"
    lib.write_bytes(b"")

    class _Result:
        stdout = _UNGUARDED

    monkeypatch.setattr(MOD.subprocess, "run", lambda *a, **k: _Result())
    scan = MOD.scan_library(lib)
    assert scan.avx512_hits == 2
    assert scan.cpuid_hits == 0
    assert scan.risk == "unguarded"


def test_scan_library_skips_unreadable(tmp_path: Path) -> None:
    not_an_object = tmp_path / "libbroken.so"
    not_an_object.write_text("not an ELF object")
    scan = MOD.scan_library(not_an_object)  # objdump failure -> clean, no crash
    assert scan.avx512_hits == 0
    assert scan.risk == "clean"


def test_discover_libraries_recurses_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "_ext.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    (tmp_path / "pkg.libs").mkdir()
    (tmp_path / "pkg.libs" / "libbundled-abc.so").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    found = MOD.discover_libraries(tmp_path)
    assert {p.name for p in found} == {
        "_ext.cpython-312-x86_64-linux-gnu.so",
        "libbundled-abc.so",
    }  # any *.so, not just lib* — third-party extensions are module-named
    assert found == sorted(found)


def test_scan_filters_to_avx512_carriers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "libclean.so").write_bytes(b"")
    (tmp_path / "libdispatch.so").write_bytes(b"")
    (tmp_path / "libstatic.so").write_bytes(b"")

    def fake(library: Path) -> MOD.LibraryScan:
        if library.name == "libdispatch.so":
            return MOD.LibraryScan(library, avx512_hits=3, cpuid_hits=1)
        if library.name == "libstatic.so":
            return MOD.LibraryScan(library, avx512_hits=3, cpuid_hits=0)
        return MOD.LibraryScan(library, avx512_hits=0, cpuid_hits=0)

    monkeypatch.setattr(MOD, "scan_library", fake)
    findings = MOD.scan(tmp_path)
    assert {f.library.name for f in findings} == {"libdispatch.so", "libstatic.so"}


def test_main_reports_and_strict_fails_on_unguarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "libstatic.so").write_bytes(b"")
    monkeypatch.setattr(
        MOD, "scan", lambda _root: [MOD.LibraryScan(tmp_path / "libstatic.so", 3, 0)]
    )
    assert MOD.main(["--root", str(tmp_path)]) == 0  # report-only by default
    out = capsys.readouterr().out
    assert "UNGUARDED" in out and "libstatic.so" in out
    assert MOD.main(["--root", str(tmp_path), "--strict"]) == 1  # strict -> non-zero


def test_main_dispatched_only_passes_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        MOD, "scan", lambda _root: [MOD.LibraryScan(tmp_path / "libd.so", 5, 2)]
    )
    assert MOD.main(["--root", str(tmp_path), "--strict"]) == 0  # dispatched is advisory only
    assert "dispatched" in capsys.readouterr().out


def test_main_missing_dir_is_noop() -> None:
    assert MOD.main(["--root", "/nonexistent/site-packages"]) == 0
