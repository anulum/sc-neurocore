# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verification of sc_shd_event_axi and its integrity-checked loader

"""Verify ``hdl/shd_event/sc_shd_event_axi.v`` register contracts, formal job, and cosimulation.

Covers:
1. ``T_MAX`` parameter equality between ``sc_shd_event_p.v`` and ``sc_shd_event_axi.v``,
   plus offset compatibility across ``0x00..0x8C`` with ``hdl/pynq/sc_shd_driver.py`` and
   ``hdl/sc_shd_axi_wrapper.v`` except ``0x08..0x10`` (reserved on the event bitstream).
2. Bit-exact AXI4-Lite cosimulation on the anchor job set via ``FRAME_COUNT_0..17``
   (counts >= 2, ``auto_increment=1``, ``zlib.crc32`` seal) across ``generic`` and
   ``xilinx_7series`` in Icarus Verilog and Verilator.
3. Binary-only ``0/1`` cosimulation via ``SPIKE_IN_0..4`` (``auto_increment=0``), plus
   byte-identical output verification after refused ``0x08..0x10`` writes.
4. Exhaustive negative protocol suite (unsealed start, CRC mismatch, loader write while
   busy, out-of-range ``LOAD_TARGET`` and auto-increment past ``d2[127]`` refusals,
   post-seal loader clear, ``t_orig`` 0/1025 refusals, ``0x08..0x10`` and ``0x0B0``/``0x0B4``
   reserved refusals, nonzero ``FRAME_COUNT_17[31:16]``, commit while ``!frame_ready``,
   reset seal clear).
5. CLI and fail-closed input/environment validation in ``tools/cosim_shd_event_axi.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import runpy
import shutil
import sys
import zlib

import numpy as np
import numpy.typing as npt
import pytest

from tools import cosim_shd_event_axi as axi_cosim
from tools import cosim_shd_event_core as core_cosim
from tools import shd_int_reference as ref

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = REPO_ROOT / "hdl" / "shd_event"
CORE_RTL = EVENT_DIR / "sc_shd_event_p.v"
AXI_RTL = EVENT_DIR / "sc_shd_event_axi.v"
OLD_WRAPPER_RTL = REPO_ROOT / "hdl" / "sc_shd_axi_wrapper.v"
PYNQ_DRIVER = REPO_ROOT / "hdl" / "pynq" / "sc_shd_driver.py"
SBY_FILE = REPO_ROOT / "hdl" / "formal" / "sc_shd_event_axi.sby"
FORMAL_V = REPO_ROOT / "hdl" / "formal" / "sc_shd_event_axi_formal.v"
FIXTURE_NETWORK = REPO_ROOT / "hdl" / "fixtures" / "shd_int_network.json"
SIM_TIMEOUT_S = 900.0


def _trained_net() -> ref.IntNetwork:
    return ref.IntNetwork.from_json(FIXTURE_NETWORK.read_text(encoding="utf-8"))


def _sparse_count_frames(seed: int, steps: int) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 10, (steps, ref.N_INPUT)) == 0
    counts = rng.integers(1, 8, (steps, ref.N_INPUT), dtype=np.int64)
    return np.where(mask, counts, 0).astype(np.int64)


def _binary_frames(seed: int, steps: int) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    bits = np.asarray(rng.integers(0, 8, (steps, ref.N_INPUT)) == 0, dtype=np.int64)
    bits[0, 0] = 1
    bits[0, 128] = 1
    bits[0, 139] = 1
    return bits


def test_tmax_equality_and_old_driver_offset_compatibility() -> None:
    """Wrapper T_MAX equals core T_MAX, and 0x00..0x8C offsets match except reserved 0x08..0x10."""
    core_text = CORE_RTL.read_text(encoding="utf-8")
    axi_text = AXI_RTL.read_text(encoding="utf-8")
    old_text = OLD_WRAPPER_RTL.read_text(encoding="utf-8")

    core_m = re.search(r"localparam\s+integer\s+T_MAX\s*=\s*(\d+)", core_text)
    axi_m = re.search(r"localparam\s+\[15:0\]\s+T_MAX\s*=\s*16'd(\d+)", axi_text)
    assert core_m is not None and axi_m is not None
    assert int(core_m.group(1)) == int(axi_m.group(1)) == 1024

    addr_w_m = re.search(r"parameter\s+integer\s+C_S_AXI_ADDR_WIDTH\s*=\s*(\d+)", axi_text)
    assert addr_w_m is not None and int(addr_w_m.group(1)) == 9
    assert "authenticated" not in axi_text.lower()
    assert "does not authenticate" in axi_text.lower()

    spec = importlib.util.spec_from_file_location("sc_shd_driver", PYNQ_DRIVER)
    assert spec is not None and spec.loader is not None
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    retained_old_offsets = {
        "_CTRL": 0x00,
        "_T_ORIG": 0x04,
        "_SPIKE_IN_0": 0x14,
        "_SPIKE_IN_1": 0x18,
        "_SPIKE_IN_2": 0x1C,
        "_SPIKE_IN_3": 0x20,
        "_SPIKE_IN_4": 0x24,
        "_SPIKE_COMMIT": 0x28,
        "_OUT_V_BASE": 0x40,
    }
    for attr, byte_off in retained_old_offsets.items():
        assert getattr(drv, attr) == byte_off
        word_hex = f"6'h{byte_off // 4:02x}"
        if attr != "_OUT_V_BASE":
            assert word_hex.lower() in old_text.lower()
            assert word_hex.lower() in axi_text.lower()

    reserved_scale_offsets = {
        "_SCALE_L1": 0x08,
        "_SCALE_L2": 0x0C,
        "_SCALE_L3": 0x10,
    }
    for attr, byte_off in reserved_scale_offsets.items():
        assert getattr(drv, attr) == byte_off
        word_hex = f"6'h{byte_off // 4:02x}"
        assert word_hex.lower() in old_text.lower()
        assert word_hex.lower() not in axi_text.lower()
    assert "0x008..0x010 -    reserved" in axi_text.lower()
    assert "6'h10" in axi_text and "6'h23" in axi_text


def test_crc32_and_word_packing_match_zlib() -> None:
    """Packed 64-bit network stream has 4,940 words at P=8 and matches zlib.crc32."""
    net = _trained_net()
    words = axi_cosim.pack_network_words(net, lanes=8)
    assert len(words) == 140 * 16 + 128 * 16 + 128 * 3 + 140 + 128 == 4940

    raw = bytearray()
    for _sel, _addr, _grp, w64 in words:
        raw.extend(int(w64).to_bytes(8, "little"))
    assert axi_cosim.network_crc32(net, lanes=8) == (zlib.crc32(bytes(raw)) & 0xFFFF_FFFF)


def test_formal_sby_and_harness_structure() -> None:
    """Formal .sby and harness carry SPDX headers and declare prove and cover tasks."""
    for path in (SBY_FILE, FORMAL_V):
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:7])
        assert "SPDX-License-Identifier: AGPL-3.0-or-later" in head
        assert "ORCID: 0009-0009-3560-0851" in head

    sby_text = SBY_FILE.read_text(encoding="utf-8")
    assert "prove: mode prove" in sby_text
    assert "cover: mode cover" in sby_text
    assert "check -assert" in sby_text

    files_block = sby_text.split("[files]", 1)[1].strip().splitlines()
    assert len(files_block) == 9
    for rel in files_block:
        assert (SBY_FILE.parent / rel.strip()).is_file()


@pytest.mark.parametrize("target", ["generic", "xilinx_7series"])
@pytest.mark.parametrize("use_verilator", [False, True])
def test_anchor_job_set_bit_exact_through_frame_count_window(
    target: str,
    use_verilator: bool,
    tmp_path: Path,
) -> None:
    """AXI loader + FRAME_COUNT_0..17 match Golden Reference #0 bit for bit on anchor jobs."""
    net = _trained_net()
    f0 = _sparse_count_frames(2, 24)
    f1 = _sparse_count_frames(3, 7)
    assert int(f0.max()) >= 2 and int(f1.max()) >= 2
    samples: list[axi_cosim.SampleInput] = [f0, 0, 1025, f1]

    axi_results = axi_cosim.simulate_axi(
        net,
        samples,
        timeout_seconds=SIM_TIMEOUT_S,
        target=target,
        verilator=use_verilator,
        work=tmp_path / f"axi_{target}_{int(use_verilator)}",
    )
    core_results = core_cosim.simulate(
        net,
        samples,
        timeout_seconds=SIM_TIMEOUT_S,
        target=target,
        verilator=use_verilator,
        work=tmp_path / f"core_{target}_{int(use_verilator)}",
    )
    assert [r.refused for r in axi_results] == [False, True, True, False]
    for ar, cr in zip(axi_results, core_results, strict=True):
        assert ar.refused == cr.refused
        assert ar.matches
        assert ar.rtl == cr.rtl


@pytest.mark.parametrize("use_verilator", [False, True])
def test_binary_spike_in_expansion_cosimulation(use_verilator: bool) -> None:
    """Binary 0/1 frames driven through SPIKE_IN_0..4 match shd_int_reference bit for bit."""
    net = _trained_net()
    b0 = _binary_frames(11, 10)
    b1 = _binary_frames(12, 5)
    results = axi_cosim.simulate_axi(
        net,
        [b0, b1],
        timeout_seconds=SIM_TIMEOUT_S,
        target="xilinx_7series",
        binary_spike_mode=True,
        verilator=use_verilator,
    )
    assert all(r.matches and not r.refused for r in results)
    if not use_verilator:
        poked = axi_cosim.simulate_axi(
            net,
            [b0, b1],
            timeout_seconds=SIM_TIMEOUT_S,
            target="xilinx_7series",
            binary_spike_mode=True,
            poke_reserved_scales=True,
            verilator=False,
        )
        assert [r.rtl for r in poked] == [r.rtl for r in results]


@pytest.mark.parametrize("target", ["generic", "xilinx_7series"])
def test_negative_protocol_suite(target: str, tmp_path: Path) -> None:
    """Negative AXI4-Lite and loader protocol suite passes on both targets and simulators."""
    net = _trained_net()
    assert axi_cosim.run_negative_protocol_suite(
        net,
        timeout_seconds=SIM_TIMEOUT_S,
        target=target,
        verilator=False,
        work=tmp_path / f"neg_iv_{target}",
    )
    if target == "generic":
        assert axi_cosim.run_negative_protocol_suite(
            net,
            timeout_seconds=SIM_TIMEOUT_S,
            target=target,
            verilator=True,
        )


def test_cli_and_validation_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI entrypoint and fail-closed parameter/simulator/output refusals behave as specified."""
    net = _trained_net()
    net_path = tmp_path / "net.json"
    net_path.write_text(net.to_json() + "\n", encoding="utf-8")

    b_frames = _binary_frames(21, 4)
    npz_path = tmp_path / "bin_bundle.npz"
    np.savez(
        npz_path,
        frames=b_frames,
        lengths=np.array([4], dtype=np.int64),
        labels=np.array([int(np.argmax(ref.run(net, b_frames)))], dtype=np.int64),
    )

    rc = axi_cosim.main(
        [
            str(net_path),
            str(npz_path),
            "--binary-spikes",
            "--target",
            "xilinx_7series",
            "--timeout-seconds",
            "300",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["accumulator_mismatches"] == 0
    assert payload["rtl_correct"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["cosim_shd_event_axi.py", str(net_path), str(npz_path), "--timeout-seconds", "300"],
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(REPO_ROOT / "tools" / "cosim_shd_event_axi.py"), run_name="__main__")
    assert exc_info.value.code == 0
    capsys.readouterr()

    with pytest.raises(ValueError, match="lanes"):
        axi_cosim.pack_network_words(net, lanes=16)
    with pytest.raises(ValueError, match="timeout_seconds"):
        axi_cosim.simulate_axi(net, [b_frames], timeout_seconds=0.0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        axi_cosim.run_negative_protocol_suite(net, timeout_seconds=-1.0)
    with pytest.raises(ValueError, match="binary 0/1"):
        axi_cosim.simulate_axi(
            net,
            [0, np.full((2, 140), 2, dtype=np.int64)],
            timeout_seconds=10.0,
            binary_spike_mode=True,
        )
    with pytest.raises(ValueError, match="target"):
        axi_cosim.simulate_axi(net, [b_frames], timeout_seconds=10.0, target="bad")

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="Icarus Verilog"):
        axi_cosim.simulate_axi(net, [b_frames], timeout_seconds=10.0, verilator=False)
    with pytest.raises(RuntimeError, match="verilator"):
        axi_cosim.simulate_axi(net, [b_frames], timeout_seconds=10.0, verilator=True)

    def _fake_build(
        work: Path,
        params: dict[str, int],
        *,
        target: str,
        verilator: bool,
        timeout_seconds: float,
    ) -> list[str]:
        del params, target, verilator, timeout_seconds
        (work / "outputs.txt").write_text("", encoding="utf-8")
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(axi_cosim, "_build_axi", _fake_build)
    with pytest.raises(RuntimeError, match="results for"):
        axi_cosim.simulate_axi(net, [b_frames], timeout_seconds=10.0, work=tmp_path / "trunc")
