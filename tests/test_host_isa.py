# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the host instruction-set level

"""The host's x86-64 psABI level, from Linux ``/proc/cpuinfo`` flag names."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.accel import host_isa

_V1 = frozenset({"cmov", "cx8", "fpu", "fxsr", "mmx", "syscall", "sse", "sse2"})
_V2 = _V1 | {"cx16", "lahf_lm", "popcnt", "pni", "sse4_1", "sse4_2", "ssse3"}
_V3 = _V2 | {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe"}
_V4 = _V3 | {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}

# ML350, Intel Xeon E5-2640: the flags line of /proc/cpuinfo, read from the host on 2026-11-19 (87 flags).
_ML350_E5_2640 = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse "
    "sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl "
    "xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 "
    "xtpr pdcm pcid dca sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx lahf_lm epb pti ssbd "
    "ibrs ibpb stibp tpr_shadow flexpriority ept vpid xsaveopt dtherm ida arat pln pts vnmi md_clear "
    "flush_l1d ibpb_exit_to_user"
)

# The workstation, Intel Core i5-11600K: the flags line of /proc/cpuinfo, read from the host on 2026-11-19 (139 flags).
_WS_I5_11600K = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse "
    "sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl "
    "xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est "
    "tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave "
    "avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow "
    "flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx avx512f avx512dq "
    "rdseed adx smap avx512ifma clflushopt intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 "
    "xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip "
    "pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm "
    "md_clear flush_l1d arch_capabilities"
)


@pytest.mark.parametrize(
    ("flags", "level"),
    (
        (_V1, "x86-64-v1"),
        (_V2, "x86-64-v2"),
        (_V3, "x86-64-v3"),
        (_V4, "x86-64-v4"),
    ),
)
def test_each_level_is_reached_by_its_own_flag_set(flags: frozenset[str], level: str) -> None:
    """The four psABI flag sets each give their own level."""
    assert host_isa.level_from_flags(flags) == level


@pytest.mark.parametrize(
    ("full", "below"),
    ((_V2, "x86-64-v1"), (_V3, "x86-64-v2"), (_V4, "x86-64-v3")),
)
def test_one_missing_flag_drops_exactly_one_level(full: frozenset[str], below: str) -> None:
    """Removing any one flag that a level adds drops the result to the level beneath it."""
    previous = {"x86-64-v2": _V1, "x86-64-v3": _V2, "x86-64-v4": _V3}[
        host_isa.level_from_flags(full)
    ]
    for flag in sorted(full - previous):
        assert host_isa.level_from_flags(full - {flag}) == below, flag


def test_a_flag_set_without_the_baseline_is_unknown() -> None:
    """No level is guessed when even the baseline is incomplete."""
    assert host_isa.level_from_flags(_V4 - {"sse2"}) == "unknown"
    assert host_isa.level_from_flags(frozenset()) == "unknown"


def test_lzcnt_is_read_under_its_linux_name_and_osxsave_is_not_required() -> None:
    """Linux prints LZCNT as ``abm`` and does not print OSXSAVE; neither is asked under another name."""
    assert "abm" in host_isa.LEVEL_FLAGS["x86-64-v3"]
    assert "lzcnt" not in host_isa.LEVEL_FLAGS["x86-64-v3"]
    assert all("osxsave" not in flags for flags in host_isa.LEVEL_FLAGS.values())


def test_the_ml350_flag_line_gives_v2() -> None:
    """A Xeon E5-2640 has AVX but neither AVX2 nor FMA: x86-64-v2."""
    assert host_isa.level_from_flags(frozenset(_ML350_E5_2640.split())) == "x86-64-v2"


def test_the_workstation_flag_line_gives_v4() -> None:
    """A Core i5-11600K has the five AVX-512 subsets of x86-64-v4."""
    assert host_isa.level_from_flags(frozenset(_WS_I5_11600K.split())) == "x86-64-v4"


def test_cpuinfo_text_is_read_from_its_first_flags_line(tmp_path: Path) -> None:
    """The reader takes the ``flags`` line of the first processor."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: Test CPU\nflags\t\t: " + " ".join(sorted(_V3)) + "\n\n"
        "processor\t: 1\nflags\t\t: " + " ".join(sorted(_V1)) + "\n",
        encoding="utf-8",
    )
    assert host_isa.host_level(cpuinfo=cpuinfo, machine="x86_64", system="Linux") == "x86-64-v3"


@pytest.mark.parametrize(
    ("machine", "system"),
    (("aarch64", "Linux"), ("x86_64", "Darwin"), ("AMD64", "Windows"), ("", "Linux")),
)
def test_a_non_x86_or_non_linux_host_is_unknown(tmp_path: Path, machine: str, system: str) -> None:
    """No level is read on a host where the Linux x86 flag names do not apply."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags\t\t: " + " ".join(sorted(_V4)) + "\n", encoding="utf-8")
    assert host_isa.host_level(cpuinfo=cpuinfo, machine=machine, system=system) == "unknown"


def test_an_unreadable_or_flagless_cpuinfo_is_unknown(tmp_path: Path) -> None:
    """A missing file or one without a flags line gives ``unknown``, never an error."""
    assert (
        host_isa.host_level(cpuinfo=tmp_path / "absent", machine="x86_64", system="Linux")
        == "unknown"
    )
    empty = tmp_path / "cpuinfo"
    empty.write_text("processor\t: 0\nmodel name\t: Test CPU\n", encoding="utf-8")
    assert host_isa.host_level(cpuinfo=empty, machine="x86_64", system="Linux") == "unknown"


def test_levels_are_ordered_and_unknown_is_below_every_level() -> None:
    """``at_least`` compares levels; an unknown host meets no recorded level."""
    assert host_isa.at_least("x86-64-v4", "x86-64-v3")
    assert host_isa.at_least("x86-64-v3", "x86-64-v3")
    assert not host_isa.at_least("x86-64-v2", "x86-64-v3")
    assert not host_isa.at_least("unknown", "x86-64-v1")
    with pytest.raises(ValueError, match="not a level"):
        host_isa.at_least("x86-64-v3", "v3")


def test_the_running_host_gives_a_known_name() -> None:
    """On any host the reader returns one of the five names; it never raises."""
    assert host_isa.host_level() in (*host_isa.LEVELS, "unknown")
