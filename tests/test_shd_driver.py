# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unit and RTL-replay tests for hdl/pynq/sc_shd_driver.py
"""Tests for ``hdl/pynq/sc_shd_driver.py`` integrity-checked load and event mode.

Verifies:
1. Exact register-offset, bit-mask, field-slice, and ``T_MAX`` parity between
   ``hdl/pynq/sc_shd_driver.py``, ``hdl/shd_event/sc_shd_event_axi.v``, and
   ``hdl/sc_shd_axi_wrapper.v``, plus integrity-checked load wording.
2. Exact IEEE CRC-32 parity with ``zlib.crc32`` and ``tools/cosim_shd_event_axi.py``
   across all supported ``load_network`` input forms (``shd_int_network.json`` path,
   parsed mapping, ``IntNetwork`` instance, and ``(q1, q2, q3)`` + ``delays``),
   both ``auto_increment=True`` and ``auto_increment=False``, and lane counts
   ``(1, 2, 4, 8)``.
3. Event-mode inference routing boolean rasters through ``SPIKE_IN_0..4`` and
   integer rasters (``0..15``) through ``FRAME_COUNT_0..17`` with zero tail padding,
   matching ``tools/shd_int_reference.py`` bit-for-bit, alongside legacy wrapper
   compatibility.
4. Every refusal and bounded-poll abort path (unsealed start refusal before
   ``CTRL.start``, out-of-range counts ``< 0`` or ``> 15`` refused before bus
   writes, loader ``refused_busy`` / ``crc_mismatch`` / count / CRC mismatches,
   ``error``, ``overflow``, ``commit_refused``, ``!busy && !done`` after start,
   premature ``done``, and poll timeouts).
5. Live Icarus Verilog replay of the exact MMIO write trace emitted by
   ``SHDAccelerator.load_network`` and ``SHDAccelerator.run_event`` through
   ``hdl/shd_event/sc_shd_event_axi.v``.
"""

from __future__ import annotations

from collections import deque
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from types import ModuleType
import zlib

import numpy as np
import pytest

from tools import cosim_shd_event_axi as axi_cosim
from tools import shd_int_reference as ref

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "hdl" / "pynq" / "sc_shd_driver.py"
EVENT_AXI_RTL = REPO_ROOT / "hdl" / "shd_event" / "sc_shd_event_axi.v"
LEGACY_AXI_RTL = REPO_ROOT / "hdl" / "sc_shd_axi_wrapper.v"
XILINX_FILELIST = REPO_ROOT / "hdl" / "shd_event" / "xilinx_7series.f"
FIXTURE_NETWORK = REPO_ROOT / "hdl" / "fixtures" / "shd_int_network.json"
# Words of a complete P = 8 load (the wrapper's LOAD_WORDS): inputs and hidden rows per group,
# the output passes, and the two delay tables.
EVENT_LOAD_WORDS = 140 * (128 // 8) + 128 * (128 // 8) + 128 * ((20 + 8 - 1) // 8) + 140 + 128


def _load_driver_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sc_shd_driver", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sc_shd_driver"] = module
    spec.loader.exec_module(module)
    return module


drv = _load_driver_module()


class EventAxiBusModel:
    """Deterministic register-accurate behavioural model of ``sc_shd_event_axi``."""

    def __init__(self) -> None:
        self.writes: list[tuple[int, int]] = []
        self.reads: list[tuple[int, int]] = []
        self.t_orig = 250
        self.auto_inc = False
        self.load_target = 0
        self.wdata_lo = 0
        self.wdata_hi = 0
        self.load_count = 0
        self.crc_bytes = bytearray()
        self.expected_crc32 = 0
        self.sealed = False
        self.crc_mismatch = False
        self.refused_busy = False
        self.refused_range = False

        self.q1 = np.zeros((ref.N_HIDDEN, ref.N_INPUT), dtype=np.int64)
        self.q2 = np.zeros((ref.N_HIDDEN, ref.N_HIDDEN), dtype=np.int64)
        self.q3 = np.zeros((ref.N_OUTPUT, ref.N_HIDDEN), dtype=np.int64)
        self.d1 = np.zeros((ref.N_INPUT,), dtype=np.int64)
        self.d2 = np.zeros((ref.N_HIDDEN,), dtype=np.int64)

        self.busy = False
        self.done_sticky = False
        self.frame_ready = False
        self.error = False
        self.overflow = False
        self.commit_refused = False
        self.frame_buf = np.zeros((ref.N_INPUT,), dtype=np.int64)
        self.captured_frames: list[np.ndarray] = []
        self.out_v = np.zeros((ref.N_OUTPUT,), dtype=np.int32)
        self.ready_delay_polls = 1
        self._polls_left = 0
        self._done_polls_left = 0

    def _next_target(self, target: int) -> int:
        sel = target & 0x7
        grp = (target >> 3) & 0x7F
        addr = (target >> 10) & 0xFF
        if sel == 0:
            if grp + 1 < 16:
                grp += 1
            elif addr + 1 < 140:
                grp = 0
                addr += 1
            else:
                sel, grp, addr = 1, 0, 0
        elif sel == 1:
            if grp + 1 < 16:
                grp += 1
            elif addr + 1 < 128:
                grp = 0
                addr += 1
            else:
                sel, grp, addr = 2, 0, 0
        elif sel == 2:
            if grp + 1 < 3:
                grp += 1
            elif addr + 1 < 128:
                grp = 0
                addr += 1
            else:
                sel, grp, addr = 3, 0, 0
        elif sel == 3:
            grp = 0
            if addr + 1 < 140:
                addr += 1
            else:
                sel, addr = 4, 0
        else:
            grp = 0
            addr = (addr + 1) & 0xFF
        return ((addr & 0xFF) << 10) | ((grp & 0x7F) << 3) | (sel & 0x7)

    def _commit_loader_word(self, word64: int) -> None:
        sel = self.load_target & 0x7
        grp = (self.load_target >> 3) & 0x7F
        addr = (self.load_target >> 10) & 0xFF
        for k in range(8):
            byte_val = (word64 >> (8 * k)) & 0xFF
            signed_val = byte_val - 256 if byte_val >= 128 else byte_val
            if sel == 0:
                self.q1[grp * 8 + k, addr] = signed_val
            elif sel == 1:
                self.q2[grp * 8 + k, addr] = signed_val
            elif sel == 2 and grp * 8 + k < ref.N_OUTPUT:
                self.q3[grp * 8 + k, addr] = signed_val
        if sel == 3:
            self.d1[addr] = word64 & 0x1F
        elif sel == 4:
            self.d2[addr] = word64 & 0x1F
        self.load_count += 1
        self.crc_bytes.extend(struct.pack("<Q", word64 & 0xFFFF_FFFF_FFFF_FFFF))
        if self.auto_inc:
            self.load_target = self._next_target(self.load_target)

    @property
    def load_crc32(self) -> int:
        """Return the reflected IEEE CRC-32 of committed loader words."""
        return zlib.crc32(self.crc_bytes) & 0xFFFF_FFFF

    def read(self, offset: int) -> int:
        """Read the 32-bit register at byte ``offset``."""
        val = 0
        if offset == drv.REG_CTRL:
            if self.busy and len(self.captured_frames) < self.t_orig:
                if self._polls_left > 0:
                    self._polls_left -= 1
                    self.frame_ready = False
                else:
                    self.frame_ready = True
            elif self.busy and len(self.captured_frames) == self.t_orig:
                self.frame_ready = False
                if self._done_polls_left > 0:
                    self._done_polls_left -= 1
                else:
                    net = ref.IntNetwork(
                        frac_bits=0,
                        d1=self.d1.copy(),
                        d2=self.d2.copy(),
                        q1=self.q1.copy(),
                        q2=self.q2.copy(),
                        q3=self.q3.copy(),
                        theta1=72,
                        theta2=51,
                    )
                    mat = np.stack(self.captured_frames, axis=0)
                    sums = ref.run(net, mat)
                    self.out_v = sums.astype(np.int32)
                    self.busy = False
                    self.done_sticky = True
            val = (
                (int(self.busy) << 1)
                | (int(self.done_sticky) << 2)
                | (int(self.frame_ready) << 3)
                | (int(self.error) << 4)
                | (int(self.overflow) << 5)
                | (int(self.commit_refused) << 6)
            )
        elif offset == drv.REG_LOAD_COUNT:
            val = self.load_count
        elif offset == drv.REG_LOAD_CRC32:
            val = self.load_crc32
        elif offset == drv.REG_LOAD_STATUS:
            val = (
                int(self.sealed)
                | (int(self.crc_mismatch) << 1)
                | (int(self.refused_busy) << 2)
                | (int(self.refused_range) << 3)
            )
        else:
            assert drv.REG_OUT_V_BASE <= offset < drv.REG_OUT_V_BASE + 80
            idx = (offset - drv.REG_OUT_V_BASE) // 4
            val = int(self.out_v[idx]) & 0xFFFF_FFFF
        self.reads.append((offset, val))
        return val

    def write(self, offset: int, value: int) -> None:
        """Write 32-bit ``value`` to byte ``offset``."""
        assert offset not in drv.RESERVED_EVENT_OFFSETS
        u32 = value & 0xFFFF_FFFF
        self.writes.append((offset, u32))
        if offset == drv.REG_CTRL:
            assert (u32 & drv.CTRL_START) != 0
            assert self.sealed and 1 <= self.t_orig <= drv.EVENT_T_MAX
            self.busy = True
            self.done_sticky = False
            self.commit_refused = False
            self.captured_frames = []
            self._polls_left = self.ready_delay_polls
            self._done_polls_left = self.ready_delay_polls
        elif offset == drv.REG_T_ORIG:
            self.t_orig = u32 & 0xFFFF
        elif drv.REG_SPIKE_IN_0 <= offset <= drv.REG_SPIKE_IN_4:
            word_idx = (offset - drv.REG_SPIKE_IN_0) // 4
            for bit in range(32):
                ch = word_idx * 32 + bit
                if ch < ref.N_INPUT:
                    self.frame_buf[ch] = (u32 >> bit) & 0x1
        elif drv.REG_FRAME_COUNT_0 <= offset <= drv.REG_FRAME_COUNT_17:
            word_idx = (offset - drv.REG_FRAME_COUNT_0) // 4
            assert word_idx < 17 or (u32 >> 16) == 0
            limit = 8 if word_idx < 17 else 4
            for sub in range(limit):
                self.frame_buf[word_idx * 8 + sub] = (u32 >> (4 * sub)) & 0xF
        elif offset == drv.REG_SPIKE_COMMIT:
            assert self.frame_ready
            self.captured_frames.append(self.frame_buf.copy())
            self.frame_buf.fill(0)
            self.frame_ready = False
            self._polls_left = self.ready_delay_polls
        elif offset == drv.REG_LOAD_CTRL:
            self.auto_inc = bool(u32 & drv.LOAD_CTRL_AUTO_INC)
            if u32 & drv.LOAD_CTRL_CLEAR:
                self.load_count = 0
                self.crc_bytes.clear()
                self.sealed = False
                self.crc_mismatch = False
                self.refused_busy = False
                self.refused_range = False
            else:
                assert (u32 & drv.LOAD_CTRL_SEAL) != 0
                assert (
                    self.load_count == EVENT_LOAD_WORDS and self.load_crc32 == self.expected_crc32
                )
                self.sealed = True
                self.crc_mismatch = False
        elif offset == drv.REG_LOAD_TARGET:
            self.load_target = u32 & 0x3FFFF
            self.sealed = False
        elif offset == drv.REG_LOAD_WDATA_LO:
            self.wdata_lo = u32
            self.sealed = False
        elif offset == drv.REG_LOAD_WDATA_HI:
            self.wdata_hi = u32
            self.sealed = False
            self._commit_loader_word((u32 << 32) | self.wdata_lo)
        else:
            assert offset == drv.REG_EXPECTED_CRC32
            self.expected_crc32 = u32
            self.sealed = False


class ScriptedBus:
    """Minimal MMIO double returning scripted status reads for fault-path tests."""

    def __init__(self, responses: dict[int, list[int] | int]) -> None:
        self._queues: dict[int, deque[int]] = {}
        self._defaults: dict[int, int] = {}
        for off, seq in responses.items():
            if isinstance(seq, list):
                self._queues[off] = deque(seq)
                self._defaults[off] = seq[-1]
            else:
                self._defaults[off] = seq
        self.writes: list[tuple[int, int]] = []

    def read(self, offset: int) -> int:
        """Pop the next scripted register value at byte ``offset``."""
        q = self._queues.get(offset)
        if q:
            return q.popleft()
        return self._defaults.get(offset, 0)

    def write(self, offset: int, value: int) -> None:
        """Record a 32-bit write to byte ``offset``."""
        self.writes.append((offset, value & 0xFFFF_FFFF))


def test_register_constants_and_bitmasks_match_rtl() -> None:
    """Every register offset, status bit, and T_MAX constant matches sc_shd_event_axi.v."""
    axi_text = EVENT_AXI_RTL.read_text(encoding="utf-8")
    old_text = LEGACY_AXI_RTL.read_text(encoding="utf-8")
    drv_text = DRIVER_PATH.read_text(encoding="utf-8")

    assert "authenticated" not in drv_text.lower()
    assert "integrity-checked load" in drv_text.lower()
    assert "does not authenticate" in drv_text.lower()

    tmax_m = re.search(r"localparam\s+\[15:0\]\s+T_MAX\s*=\s*16'd(\d+)", axi_text)
    assert tmax_m is not None and int(tmax_m.group(1)) == drv.EVENT_T_MAX == 1024

    word_checks = {
        drv.REG_CTRL: "6'h00",
        drv.REG_T_ORIG: "6'h01",
        drv.REG_SPIKE_IN_0: "6'h05",
        drv.REG_SPIKE_IN_1: "6'h06",
        drv.REG_SPIKE_IN_2: "6'h07",
        drv.REG_SPIKE_IN_3: "6'h08",
        drv.REG_SPIKE_IN_4: "6'h09",
        drv.REG_SPIKE_COMMIT: "6'h0a",
        drv.REG_OUT_V_BASE: "6'h10",
        drv.REG_LOAD_CTRL: "6'h24",
        drv.REG_LOAD_TARGET: "6'h25",
        drv.REG_LOAD_WDATA_LO: "6'h26",
        drv.REG_LOAD_WDATA_HI: "6'h27",
        drv.REG_LOAD_COUNT: "6'h28",
        drv.REG_LOAD_CRC32: "6'h29",
        drv.REG_EXPECTED_CRC32: "6'h2a",
        drv.REG_LOAD_STATUS: "6'h2b",
        drv.REG_FRAME_COUNT_0: "6'h2e",
        drv.REG_FRAME_COUNT_17: "6'h3f",
    }
    for byte_off, word_tok in word_checks.items():
        assert byte_off % 4 == 0
        assert int(word_tok.split("h")[1], 16) == byte_off // 4
        assert word_tok in axi_text.lower()

    assert drv.RESERVED_EVENT_OFFSETS == (0x08, 0x0C, 0x10, 0xB0, 0xB4)
    for scale_tok in ("6'h02", "6'h03", "6'h04"):
        assert scale_tok not in axi_text.lower()
        assert scale_tok in old_text.lower()
    assert "0x008..0x010 -    reserved" in axi_text.lower()

    for old_tok in ("6'h00", "6'h01", "6'h02", "6'h03", "6'h04", "6'h0a", "6'h10"):
        assert old_tok in old_text.lower()

    assert (
        "{25'd0, commit_refused, core_overflow, core_error,\n"
        "                                         status_frame_ready, done_sticky, wrapper_busy, 1'b0}"
    ) in axi_text
    assert (
        "{27'd0, count_short, refused_range, refused_busy,\n"
        "                                         crc_mismatch, sealed}"
    ) in axi_text
    assert "(load_count == LOAD_WORDS) &&" in axi_text
    assert drv.CTRL_START == 0x01
    assert drv.CTRL_BUSY == 0x02
    assert drv.CTRL_DONE == 0x04
    assert drv.CTRL_FRAME_READY == 0x08
    assert drv.CTRL_ERROR == 0x10
    assert drv.CTRL_OVERFLOW == 0x20
    assert drv.CTRL_COMMIT_REFUSED == 0x40
    assert drv.LOAD_CTRL_CLEAR == 0x01
    assert drv.LOAD_CTRL_AUTO_INC == 0x02
    assert drv.LOAD_CTRL_SEAL == 0x04
    assert drv.LOAD_STATUS_SEALED == 0x01
    assert drv.LOAD_STATUS_CRC_MISMATCH == 0x02
    assert drv.LOAD_STATUS_REFUSED_BUSY == 0x04
    assert drv.LOAD_STATUS_REFUSED_RANGE == 0x08
    assert drv.LOAD_STATUS_COUNT_SHORT == 0x10


def test_pack_network_and_crc32_across_all_input_forms() -> None:
    """All four load_network input forms produce identical words and zlib.crc32 digests."""
    net = ref.IntNetwork.from_json(FIXTURE_NETWORK.read_text(encoding="utf-8"))
    raw_dict = json.loads(FIXTURE_NETWORK.read_text(encoding="utf-8"))

    from_path = drv.pack_shd_int_network(FIXTURE_NETWORK)
    from_str = drv.pack_shd_int_network(str(FIXTURE_NETWORK))
    from_dict = drv.pack_shd_int_network(raw_dict)
    from_obj = drv.pack_shd_int_network(net)
    from_tuples = drv.pack_shd_int_network((net.q1, net.q2, net.q3), delays=(net.d1, net.d2))
    cosim_words = axi_cosim.pack_network_words(net, lanes=8)

    assert len(from_path) == 4940
    assert from_path == from_str == from_dict == from_obj == from_tuples == cosim_words

    words_u64 = [w for _sel, _addr, _grp, w in from_path]
    expected_zlib = zlib.crc32(b"".join(struct.pack("<Q", w) for w in words_u64)) & 0xFFFF_FFFF
    assert drv.compute_load_crc32(words_u64) == expected_zlib
    assert axi_cosim.network_crc32(net, lanes=8) == expected_zlib

    for lanes in (1, 2, 4):
        assert drv.pack_shd_int_network(net, lanes=lanes) == axi_cosim.pack_network_words(
            net, lanes=lanes
        )


def test_load_network_and_event_inference_count_and_bool_modes() -> None:
    """load_network + run_event/classify match shd_int_reference bit-for-bit."""
    net = ref.IntNetwork.from_json(FIXTURE_NETWORK.read_text(encoding="utf-8"))
    bus = EventAxiBusModel()

    class OverlayIPWrapper:
        def __init__(self, inner: EventAxiBusModel) -> None:
            self.mmio = inner

    accel = drv.SHDAccelerator(OverlayIPWrapper(bus))
    assert not accel.is_sealed
    assert not accel.is_running
    assert not accel.is_done

    crc_auto = accel.load_network(FIXTURE_NETWORK, auto_increment=True)
    assert accel.is_sealed
    assert crc_auto == axi_cosim.network_crc32(net, lanes=8)

    rng = np.random.default_rng(1234)
    count_raster = rng.integers(0, 16, size=(12, 140), dtype=np.int64)
    bus.writes.clear()
    rtl_sums = accel.run_event(count_raster)
    ref_sums = ref.run(net, count_raster)
    assert np.array_equal(rtl_sums, ref_sums.astype(np.int32))
    assert accel.read_output_argmax() == ref.predict(net, count_raster)
    assert accel.is_done

    written_offsets = {off for off, _ in bus.writes}
    assert drv.REG_FRAME_COUNT_0 in written_offsets
    assert drv.REG_FRAME_COUNT_17 in written_offsets
    assert drv.REG_SPIKE_IN_0 not in written_offsets
    commit_count = sum(1 for off, _ in bus.writes if off == drv.REG_SPIKE_COMMIT)
    assert commit_count == 12

    crc_explicit = accel.load_network(
        (net.q1, net.q2, net.q3),
        delays=(net.d1, net.d2),
        auto_increment=False,
    )
    assert crc_explicit == crc_auto
    assert accel.is_sealed

    bool_raster = rng.integers(0, 2, size=(9, 140), dtype=np.int64).astype(bool)
    bus.writes.clear()
    pred_cls = accel.classify(bool_raster, mode="event")
    assert pred_cls == ref.predict(net, bool_raster.astype(np.int64))
    bool_offsets = {off for off, _ in bus.writes}
    assert drv.REG_SPIKE_IN_0 in bool_offsets
    assert drv.REG_SPIKE_IN_4 in bool_offsets
    assert drv.REG_FRAME_COUNT_0 not in bool_offsets
    assert sum(1 for off, _ in bus.writes if off == drv.REG_SPIKE_COMMIT) == 9


def test_legacy_wrapper_mode_and_signed_voltages_preserved() -> None:
    """Legacy set_scales and classify(mode='legacy') keep 30-step tail padding intact."""
    responses: dict[int, list[int] | int] = {
        drv.REG_CTRL: [0x2, 0x4],
    }
    for i in range(20):
        raw_val = (-50 + i * 7) & 0xFFFF_FFFF
        responses[drv.REG_OUT_V_BASE + 4 * i] = raw_val
    bus = ScriptedBus(responses)
    accel = drv.SHDAccelerator(bus)

    accel.set_scales(-1, 0x751, 0x11B)
    assert bus.writes[:3] == [
        (drv._SCALE_L1, 0xFFFF_FFFF),
        (drv._SCALE_L2, 0x0000_0751),
        (drv._SCALE_L3, 0x0000_011B),
    ]

    bus.writes.clear()
    raster = np.zeros((4, 140), dtype=np.uint8)
    raster[0, 0] = 1
    raster[0, 139] = 1
    pred = accel.classify(raster)
    assert pred == 19
    assert sum(1 for off, _ in bus.writes if off == drv.REG_SPIKE_COMMIT) == 34
    voltages = accel.read_output_voltages()
    assert int(voltages[0]) == -50
    assert int(voltages[19]) == 83


class BudgetedBus(ScriptedBus):
    """Scripted bus that fails the test instead of hanging once a read budget is spent."""

    def __init__(self, responses: dict[int, list[int] | int], *, budget: int) -> None:
        super().__init__(responses)
        self._budget = budget
        self.ctrl_reads = 0

    def read(self, offset: int) -> int:
        """Count ``CTRL`` reads and refuse every read past the budget."""
        assert self._budget > 0, "read budget spent: the done wait is unbounded"
        self._budget -= 1
        if offset == drv.REG_CTRL:
            self.ctrl_reads += 1
        return super().read(offset)


def test_legacy_classify_bounds_the_done_wait() -> None:
    """classify(mode='legacy') polls CTRL.done a bounded number of times and aborts on idle."""
    raster = np.zeros((3, 140), dtype=np.uint8)

    hung = BudgetedBus({drv.REG_CTRL: drv.CTRL_BUSY}, budget=10_000)
    with pytest.raises(drv.SHDDriverError, match="timed out after 50 polls") as timed_out:
        drv.SHDAccelerator(hung).classify(raster, max_poll_iters=50)
    assert hung.ctrl_reads == 50
    assert timed_out.value.status == drv.CTRL_BUSY

    idle = BudgetedBus({drv.REG_CTRL: 0}, budget=10_000)
    with pytest.raises(drv.SHDDriverError, match="idle without done") as went_idle:
        drv.SHDAccelerator(idle).classify(raster, max_poll_iters=50)
    assert idle.ctrl_reads == 1
    assert went_idle.value.status == 0

    refused = BudgetedBus({drv.REG_CTRL: drv.CTRL_DONE}, budget=10_000)
    with pytest.raises(ValueError, match="max_poll_iters"):
        drv.SHDAccelerator(refused).classify(raster, max_poll_iters=0)
    assert refused.writes == []

    finished = BudgetedBus(
        {drv.REG_CTRL: [drv.CTRL_BUSY, drv.CTRL_BUSY, drv.CTRL_DONE]}, budget=10_000
    )
    assert drv.SHDAccelerator(finished).classify(raster, max_poll_iters=3) == 0
    assert finished.ctrl_reads == 3


def test_validation_and_unsealed_or_abort_refusals(tmp_path: Path) -> None:
    """Driver refuses invalid inputs before bus writes and aborts with status on faults."""
    with pytest.raises(TypeError, match="read.*write"):
        drv.SHDAccelerator(object())

    with pytest.raises(ValueError, match="lanes"):
        drv.encode_load_target(0, 0, 0, lanes=16)
    with pytest.raises(ValueError, match="load_sel"):
        drv.encode_load_target(5, 0, 0)
    for bad_sel, bad_grp in ((0, 16), (1, 16), (2, 3), (3, 1), (4, 1), (0, -1)):
        with pytest.raises(ValueError, match="load_grp"):
            drv.encode_load_target(bad_sel, bad_grp, 0, lanes=8)
    for bad_sel, bad_addr in ((0, 140), (1, 128), (2, 128), (3, 140), (4, 128), (0, -1)):
        with pytest.raises(ValueError, match="load_addr"):
            drv.encode_load_target(bad_sel, 0, bad_addr, lanes=8)

    net = ref.IntNetwork.from_json(FIXTURE_NETWORK.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="lanes"):
        drv.pack_shd_int_network(net, lanes=16)
    with pytest.raises(ValueError, match="delays must be None"):
        drv.pack_shd_int_network(FIXTURE_NETWORK, delays=(net.d1, net.d2))
    with pytest.raises(ValueError, match="delays must be None"):
        drv.pack_shd_int_network({"q1": net.q1}, delays=(net.d1, net.d2))
    with pytest.raises(ValueError, match="delays must be None"):
        drv.pack_shd_int_network(net, delays=(net.d1, net.d2))
    with pytest.raises(ValueError, match="delays=\\(d1, d2\\) is required"):
        drv.pack_shd_int_network((net.q1, net.q2, net.q3), delays=None)
    with pytest.raises(TypeError, match="network must be"):
        drv.pack_shd_int_network(42)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        drv.pack_shd_int_network(bad_json)

    with pytest.raises(ValueError, match="expected integer array"):
        drv.pack_shd_int_network(
            (np.zeros((2, 2), dtype=np.int64), net.q2, net.q3), (net.d1, net.d2)
        )
    bad_w = net.q1.copy()
    bad_w[0, 0] = 128
    with pytest.raises(ValueError, match="q1 outside"):
        drv.pack_shd_int_network((bad_w, net.q2, net.q3), (net.d1, net.d2))
    bad_d = net.d1.copy()
    bad_d[0] = 31
    with pytest.raises(ValueError, match="d1 outside"):
        drv.pack_shd_int_network((net.q1, net.q2, net.q3), (bad_d, net.d2))

    # Before-bus validation in run_event / classify
    unsealed_bus = ScriptedBus({drv.REG_LOAD_STATUS: 0})
    accel = drv.SHDAccelerator(unsealed_bus)

    with pytest.raises(ValueError, match="mode"):
        accel.classify(np.zeros((2, 140), dtype=np.uint8), mode="unknown")
    with pytest.raises(ValueError, match="max_poll_iters"):
        accel.run_event(np.zeros((2, 140), dtype=np.uint8), max_poll_iters=0)
    with pytest.raises(ValueError, match="expected \\(T, 140\\)"):
        accel.run_event(np.zeros((2, 139), dtype=np.uint8))
    with pytest.raises(ValueError, match="outside 1..1024"):
        accel.run_event(np.zeros((0, 140), dtype=np.uint8))
    with pytest.raises(ValueError, match="outside 1..1024"):
        accel.run_event(np.zeros((1025, 140), dtype=np.uint8))
    with pytest.raises(ValueError, match="bool or integer dtype"):
        accel.run_event(np.zeros((2, 140), dtype=np.float32))

    over_15 = np.zeros((2, 140), dtype=np.int64)
    over_15[1, 7] = 16
    with pytest.raises(ValueError, match="0..15"):
        accel.run_event(over_15)
    neg_cnt = np.zeros((2, 140), dtype=np.int64)
    neg_cnt[0, 0] = -1
    with pytest.raises(ValueError, match="0..15"):
        accel.run_event(neg_cnt)
    assert unsealed_bus.writes == []

    # Unsealed start refused before CTRL.start is written!
    with pytest.raises(drv.SHDDriverError, match="unsealed") as exc_info:
        accel.run_event(np.zeros((2, 140), dtype=np.int64))
    assert exc_info.value.status == 0
    assert unsealed_bus.writes == []

    # Loader error statuses: refused_busy, crc_mismatch, count mismatch, crc mismatch
    small_w = (
        np.zeros((128, 140), dtype=np.int64),
        np.zeros((128, 128), dtype=np.int64),
        np.zeros((20, 128), dtype=np.int64),
    )
    small_d = (np.zeros((140,), dtype=np.int64), np.zeros((128,), dtype=np.int64))
    zero_crc = drv.compute_load_crc32([0] * 4940)

    busy_loader = ScriptedBus({drv.REG_LOAD_STATUS: drv.LOAD_STATUS_REFUSED_BUSY})
    with pytest.raises(drv.SHDDriverError, match="busy") as exc_busy:
        drv.SHDAccelerator(busy_loader).load_network(small_w, small_d, lanes=8)
    assert exc_busy.value.status == drv.LOAD_STATUS_REFUSED_BUSY

    range_loader = ScriptedBus({drv.REG_LOAD_STATUS: drv.LOAD_STATUS_REFUSED_RANGE})
    with pytest.raises(drv.SHDDriverError, match="out-of-range LOAD_TARGET") as exc_rng:
        drv.SHDAccelerator(range_loader).load_network(small_w, small_d, lanes=8)
    assert exc_rng.value.status == drv.LOAD_STATUS_REFUSED_RANGE

    short_loader = ScriptedBus({drv.REG_LOAD_STATUS: drv.LOAD_STATUS_COUNT_SHORT})
    with pytest.raises(drv.SHDDriverError, match="incomplete network") as exc_short:
        drv.SHDAccelerator(short_loader).load_network(small_w, small_d, lanes=8)
    assert exc_short.value.status == drv.LOAD_STATUS_COUNT_SHORT

    mismatch_loader = ScriptedBus({drv.REG_LOAD_STATUS: drv.LOAD_STATUS_CRC_MISMATCH})
    with pytest.raises(drv.SHDDriverError, match="failed to seal") as exc_seal:
        drv.SHDAccelerator(mismatch_loader).load_network(small_w, small_d, lanes=8)
    assert exc_seal.value.status == drv.LOAD_STATUS_CRC_MISMATCH

    bad_cnt_loader = ScriptedBus(
        {drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED, drv.REG_LOAD_COUNT: 10}
    )
    with pytest.raises(drv.SHDDriverError, match="LOAD_COUNT mismatch"):
        drv.SHDAccelerator(bad_cnt_loader).load_network(small_w, small_d, lanes=8)

    bad_crc_loader = ScriptedBus(
        {
            drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED,
            drv.REG_LOAD_COUNT: 4940,
            drv.REG_LOAD_CRC32: zero_crc ^ 1,
        }
    )
    with pytest.raises(drv.SHDDriverError, match="LOAD_CRC32 mismatch"):
        drv.SHDAccelerator(bad_crc_loader).load_network(small_w, small_d, lanes=8)

    # Event polling abort conditions (error, overflow, commit_refused, !busy&&!done, early done, timeouts)
    one_frame = np.zeros((1, 140), dtype=np.int64)
    two_frames = np.zeros((2, 140), dtype=np.int64)

    for fault_bit in (drv.CTRL_ERROR, drv.CTRL_OVERFLOW, drv.CTRL_COMMIT_REFUSED):
        f_bus = ScriptedBus(
            {drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED, drv.REG_CTRL: drv.CTRL_BUSY | fault_bit}
        )
        with pytest.raises(drv.SHDDriverError, match="fault waiting for frame_ready") as exc_f:
            drv.SHDAccelerator(f_bus).run_event(one_frame)
        assert exc_f.value.status == (drv.CTRL_BUSY | fault_bit)

        d_bus = ScriptedBus(
            {
                drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED,
                drv.REG_CTRL: [drv.CTRL_BUSY | drv.CTRL_FRAME_READY, fault_bit],
            }
        )
        with pytest.raises(drv.SHDDriverError, match="fault waiting for CTRL.done") as exc_d:
            drv.SHDAccelerator(d_bus).run_event(one_frame)
        assert exc_d.value.status == fault_bit

    idle_fr_bus = ScriptedBus({drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED, drv.REG_CTRL: 0})
    with pytest.raises(drv.SHDDriverError, match="idle without done") as exc_ifr:
        drv.SHDAccelerator(idle_fr_bus).run_event(one_frame)
    assert exc_ifr.value.status == 0

    early_done_bus = ScriptedBus(
        {
            drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED,
            drv.REG_CTRL: [drv.CTRL_BUSY | drv.CTRL_FRAME_READY, drv.CTRL_DONE],
        }
    )
    with pytest.raises(drv.SHDDriverError, match="asserted done before frame 1") as exc_ed:
        drv.SHDAccelerator(early_done_bus).run_event(two_frames)
    assert exc_ed.value.status == drv.CTRL_DONE

    timeout_fr_bus = ScriptedBus(
        {drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED, drv.REG_CTRL: drv.CTRL_BUSY}
    )
    with pytest.raises(drv.SHDDriverError, match="timed out.*frame_ready") as exc_tfr:
        drv.SHDAccelerator(timeout_fr_bus).run_event(one_frame, max_poll_iters=3)
    assert exc_tfr.value.status == drv.CTRL_BUSY

    idle_done_bus = ScriptedBus(
        {
            drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED,
            drv.REG_CTRL: [drv.CTRL_BUSY | drv.CTRL_FRAME_READY, 0],
        }
    )
    with pytest.raises(drv.SHDDriverError, match="idle without done") as exc_id:
        drv.SHDAccelerator(idle_done_bus).run_event(one_frame)
    assert exc_id.value.status == 0

    timeout_done_bus = ScriptedBus(
        {
            drv.REG_LOAD_STATUS: drv.LOAD_STATUS_SEALED,
            drv.REG_CTRL: [
                drv.CTRL_BUSY | drv.CTRL_FRAME_READY,
                drv.CTRL_BUSY,
                drv.CTRL_BUSY,
                drv.CTRL_BUSY,
            ],
        }
    )
    with pytest.raises(drv.SHDDriverError, match="timed out.*CTRL.done") as exc_td:
        drv.SHDAccelerator(timeout_done_bus).run_event(one_frame, max_poll_iters=2)
    assert exc_td.value.status == drv.CTRL_BUSY


def test_driver_mmio_trace_replays_bit_exact_on_rtl(tmp_path: Path) -> None:
    """One RTL trace replays load_network, an integer raster, and a boolean raster back-to-back."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None

    net = ref.IntNetwork.from_json(FIXTURE_NETWORK.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261119)
    count_sample = rng.integers(0, 12, size=(6, 140), dtype=np.int64)
    bool_sample = rng.integers(0, 2, size=(5, 140), dtype=np.int64).astype(bool)
    bool_sample[0, 0] = True
    bool_sample[0, 64] = True
    bool_sample[0, 128] = True
    bool_sample[0, 139] = True

    bus = EventAxiBusModel()
    accel = drv.SHDAccelerator(bus)
    expected_crc = accel.load_network(FIXTURE_NETWORK, auto_increment=True)
    load_writes = list(bus.writes)

    bus.writes.clear()
    py_count_sums = accel.run_event(count_sample)
    count_writes = list(bus.writes)
    ref_count_sums = ref.run(net, count_sample)
    assert np.array_equal(py_count_sums, ref_count_sums.astype(np.int32))

    bus.writes.clear()
    py_bool_sums = accel.run_event(bool_sample)
    bool_writes = list(bus.writes)
    ref_bool_sums = ref.run(net, bool_sample.astype(np.int64))
    assert np.array_equal(py_bool_sums, ref_bool_sums.astype(np.int32))

    # Emit one trace on the same RTL instance:
    # op 0 = MMIO write, op 1 = wait CTRL.frame_ready, op 2 = wait CTRL.done & dump 20 sums
    trace_lines: list[str] = [f"00000000 {off:08x} {val:08x}" for off, val in load_writes]
    for sample_writes in (count_writes, bool_writes):
        for off, val in sample_writes:
            if off in (drv.REG_FRAME_COUNT_0, drv.REG_SPIKE_IN_0):
                trace_lines.append("00000001 00000000 00000000")
            trace_lines.append(f"00000000 {off:08x} {val:08x}")
        trace_lines.append("00000002 00000000 00000000")
    (tmp_path / "mmio_trace.hex").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    tb_verilog = f"""`timescale 1ns / 1ps
module tb_driver_replay;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;
    reg [8:0] awaddr = 9'd0, araddr = 9'd0;
    reg awvalid = 1'b0, wvalid = 1'b0, bready = 1'b0, arvalid = 1'b0, rready = 1'b0;
    reg [31:0] wdata = 32'd0;
    wire awready, wready, bvalid, arready, rvalid;
    wire [1:0] bresp, rresp;
    wire [31:0] rdata;

    sc_shd_event_axi #(
        .P(8), .FRAC_BITS({net.frac_bits}), .THETA1({net.theta1}), .THETA2({net.theta2}),
        .V1_WIDTH({net.membrane_width(1)}), .V2_WIDTH({net.membrane_width(2)})
    ) dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(awaddr), .S_AXI_AWPROT(3'd0), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata), .S_AXI_WSTRB(4'hF), .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp), .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr), .S_AXI_ARPROT(3'd0), .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata), .S_AXI_RRESP(rresp), .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready)
    );

    task mmio_write(input [8:0] a, input [31:0] d);
        begin
            @(negedge clk); awaddr = a; wdata = d; awvalid = 1'b1; wvalid = 1'b1; bready = 1'b1;
            @(posedge clk); while (!(awready && wready)) @(posedge clk);
            @(negedge clk); awvalid = 1'b0; wvalid = 1'b0;
            @(posedge clk); while (!bvalid) @(posedge clk);
            if (bresp !== 2'b00) $fatal(1, "SLVERR at 0x%03x", a);
            @(negedge clk); bready = 1'b0;
        end
    endtask

    task mmio_read(input [8:0] a, output [31:0] d);
        begin
            @(negedge clk); araddr = a; arvalid = 1'b1; rready = 1'b1;
            @(posedge clk); while (!arready) @(posedge clk);
            @(negedge clk); arvalid = 1'b0;
            @(posedge clk); while (!rvalid) @(posedge clk);
            d = rdata;
            @(negedge clk); rready = 1'b0;
        end
    endtask

    integer fd, out_fd, rc, k;
    reg [31:0] op, off, val, st;
    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1'b1; @(posedge clk);
        fd = $fopen("mmio_trace.hex", "r");
        out_fd = $fopen("rtl_replay_out.txt", "w");
        while (!$feof(fd)) begin
            rc = $fscanf(fd, "%h %h %h\\n", op, off, val);
            if (rc == 3) begin
                if (op == 32'd0) mmio_write(off[8:0], val);
                else if (op == 32'd1) begin
                    mmio_read(9'h000, st);
                    while (!st[3]) mmio_read(9'h000, st);
                end else if (op == 32'd2) begin
                    mmio_read(9'h000, st);
                    while (!st[2]) mmio_read(9'h000, st);
                    for (k = 0; k < 20; k = k + 1) begin
                        mmio_read(9'h040 + k[8:0] * 9'd4, st);
                        $fwrite(out_fd, "%0d ", $signed(st));
                    end
                    $fwrite(out_fd, "\\n");
                end
            end
        end
        $fclose(fd);
        mmio_read(9'h0A4, st);
        $fwrite(out_fd, "%08x\\n", st);
        $fclose(out_fd);
        $finish;
    end
endmodule
"""
    tb_path = tmp_path / "tb_driver_replay.v"
    tb_path.write_text(tb_verilog, encoding="utf-8")
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-c",
            str(XILINX_FILELIST.relative_to(REPO_ROOT)),
            "-o",
            str(tmp_path / "replay.vvp"),
            str(EVENT_AXI_RTL.relative_to(REPO_ROOT)),
            str(tb_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [vvp, "-n", str(tmp_path / "replay.vvp")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = (tmp_path / "rtl_replay_out.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    rtl_count_vals = [int(tok) for tok in lines[0].split()]
    rtl_bool_vals = [int(tok) for tok in lines[1].split()]
    assert rtl_count_vals == ref_count_sums.tolist() == py_count_sums.tolist()
    assert rtl_bool_vals == ref_bool_sums.tolist() == py_bool_sums.tolist()
    assert int(lines[2], 16) == expected_crc
