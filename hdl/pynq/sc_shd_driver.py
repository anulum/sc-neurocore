# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SHD FPGA Driver for PYNQ-Z2
"""SC-NeuroCore SHD FPGA driver for PYNQ-Z2.

Provides a Python MMIO interface to both the legacy cycle-stepped wrapper
(``hdl/sc_shd_axi_wrapper.v``) and the event-driven AXI4-Lite wrapper with
integrity-checked weight and delay loading (``hdl/shd_event/sc_shd_event_axi.v``).

The event-mode loader consumes the canonical integer network export
``hdl/fixtures/shd_int_network.json`` (produced by
``tools/export_shd_int_network.py``) without introducing a second file format.
Its reflected IEEE 802.3 CRC-32 check provides an integrity-checked load
against accidental bus or transfer corruption; it does not authenticate the
network against an adversary with write access to the AXI4-Lite bus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import struct
from typing import Protocol, cast, runtime_checkable
import zlib

import numpy as np
import numpy.typing as npt


class RegisterBus(Protocol):
    """Read and write 32-bit AXI-Lite registers, as ``pynq.MMIO`` does."""

    def read(self, offset: int) -> int:
        """Return the unsigned 32-bit register value at byte ``offset``."""

    def write(self, offset: int, value: int) -> None:
        """Write unsigned 32-bit ``value`` to byte ``offset``."""


# Legacy-compatible private register offsets (0x00..0x8C; _SCALE_L1..L3 exist only on
# hdl/sc_shd_axi_wrapper.v and are reserved with SLVERR on hdl/shd_event/sc_shd_event_axi.v)
_CTRL = 0x00
_T_ORIG = 0x04
_SCALE_L1 = 0x08
_SCALE_L2 = 0x0C
_SCALE_L3 = 0x10
_SPIKE_IN_0 = 0x14
_SPIKE_IN_1 = 0x18
_SPIKE_IN_2 = 0x1C
_SPIKE_IN_3 = 0x20
_SPIKE_IN_4 = 0x24
_SPIKE_COMMIT = 0x28
_OUT_V_BASE = 0x40

# Public register map matching hdl/shd_event/sc_shd_event_axi.v (9-bit byte offsets;
# 0x08..0x10 and 0xB0..0xB4 are reserved on the event bitstream, read 0, write SLVERR)
RESERVED_EVENT_OFFSETS = (0x08, 0x0C, 0x10, 0xB0, 0xB4)
REG_CTRL = _CTRL
REG_T_ORIG = _T_ORIG
REG_SPIKE_IN_BASE = _SPIKE_IN_0
REG_SPIKE_IN_0 = _SPIKE_IN_0
REG_SPIKE_IN_1 = _SPIKE_IN_1
REG_SPIKE_IN_2 = _SPIKE_IN_2
REG_SPIKE_IN_3 = _SPIKE_IN_3
REG_SPIKE_IN_4 = _SPIKE_IN_4
REG_SPIKE_COMMIT = _SPIKE_COMMIT
REG_OUT_V_BASE = _OUT_V_BASE

REG_LOAD_CTRL = 0x90
REG_LOAD_TARGET = 0x94
REG_LOAD_WDATA_LO = 0x98
REG_LOAD_WDATA_HI = 0x9C
REG_LOAD_COUNT = 0xA0
REG_LOAD_CRC32 = 0xA4
REG_EXPECTED_CRC32 = 0xA8
REG_LOAD_STATUS = 0xAC

REG_FRAME_COUNT_BASE = 0xB8
REG_FRAME_COUNT_0 = 0xB8
REG_FRAME_COUNT_17 = 0xFC

# CTRL register bit masks (0x00)
CTRL_START = 1 << 0
CTRL_BUSY = 1 << 1
CTRL_DONE = 1 << 2
CTRL_FRAME_READY = 1 << 3
CTRL_ERROR = 1 << 4
CTRL_OVERFLOW = 1 << 5
CTRL_COMMIT_REFUSED = 1 << 6
CTRL_FAULT_MASK = CTRL_ERROR | CTRL_OVERFLOW | CTRL_COMMIT_REFUSED

# LOAD_CTRL register bit masks (0x90)
LOAD_CTRL_CLEAR = 1 << 0
LOAD_CTRL_AUTO_INC = 1 << 1
LOAD_CTRL_SEAL = 1 << 2

# LOAD_STATUS register bit masks (0xAC)
LOAD_STATUS_SEALED = 1 << 0
LOAD_STATUS_CRC_MISMATCH = 1 << 1
LOAD_STATUS_REFUSED_BUSY = 1 << 2
LOAD_STATUS_REFUSED_RANGE = 1 << 3

# Architectural dimensions and bounds matching sc_shd_event_p / sc_shd_event_axi
N_INPUT = 140
N_HIDDEN = 128
N_OUTPUT = 20
MAX_DELAY = 30
WEIGHT_MAX = 127
COUNT_MAX = 15
EVENT_T_MAX = 1024
DEFAULT_LANES = 8
DEFAULT_MAX_POLL_ITERS = 100_000
VALID_AXI_LANES = (1, 2, 4, 8)

WeightTuple = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]
DelayTuple = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]


class SHDDriverError(RuntimeError):
    """Hardware refusal, integrity mismatch, or bounded-poll timeout on MMIO.

    Parameters
    ----------
    message : str
        Diagnostic explanation of the refusal or abort condition.
    status : int, optional
        Raw 32-bit ``CTRL`` or ``LOAD_STATUS`` word read when the abort fired.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def encode_load_target(
    load_sel: int,
    load_grp: int,
    load_addr: int,
    *,
    lanes: int = DEFAULT_LANES,
) -> int:
    """Validate per-selector memory bounds and pack ``REG_LOAD_TARGET``.

    Enforces the exact selector, group, and row bounds of ``sc_shd_event_p`` and
    ``sc_shd_event_axi`` before any write reaches the AXI4-Lite bus:

    - ``sel=0`` (``q1``): ``grp < 128 // lanes``, ``addr < 140``
    - ``sel=1`` (``q2``): ``grp < 128 // lanes``, ``addr < 128``
    - ``sel=2`` (``q3``): ``grp < ceil(20 / lanes)``, ``addr < 128``
    - ``sel=3`` (``d1``): ``grp == 0``, ``addr < 140``
    - ``sel=4`` (``d2``): ``grp == 0``, ``addr < 128``

    Parameters
    ----------
    load_sel : int
        Target table selector in ``0..4`` (``q1``, ``q2``, ``q3``, ``d1``, ``d2``).
    load_grp : int
        Lane group index occupying bits ``[9:3]``.
    load_addr : int
        Row address occupying bits ``[17:10]``.
    lanes : int, default 8
        Parallel lane count in ``(1, 2, 4, 8)``; **must equal** the synthesised
        bitstream's ``P`` parameter (``8`` for the default build). A smaller
        ``lanes`` overruns the memory bounds and is refused by ``refused_range``,
        whereas a larger ``lanes`` sends fewer words in the wrong group layout
        while ``LOAD_COUNT`` and ``LOAD_CRC32`` still match the sent stream, so
        it cannot be detected from the bus and loads the wrong weights.

    Returns
    -------
    int
        Unsigned 18-bit ``LOAD_TARGET`` word ``{addr[7:0], grp[6:0], sel[2:0]}``.

    Raises
    ------
    ValueError
        If ``lanes`` is invalid or ``(load_sel, load_grp, load_addr)`` lies outside
        the target memory's bounds.
    """
    if lanes not in VALID_AXI_LANES:
        raise ValueError(f"lanes must be one of {VALID_AXI_LANES}, got {lanes}")
    bounds_by_sel: dict[int, tuple[int, int]] = {
        0: (N_HIDDEN // lanes, N_INPUT),
        1: (N_HIDDEN // lanes, N_HIDDEN),
        2: ((N_OUTPUT + lanes - 1) // lanes, N_HIDDEN),
        3: (1, N_INPUT),
        4: (1, N_HIDDEN),
    }
    if load_sel not in bounds_by_sel:
        raise ValueError(f"load_sel must be in 0..4, got {load_sel}")
    max_grp, max_addr = bounds_by_sel[load_sel]
    if not (0 <= load_grp < max_grp):
        raise ValueError(
            f"load_grp for sel={load_sel} (lanes={lanes}) must be in 0..{max_grp - 1}, "
            f"got {load_grp}"
        )
    if not (0 <= load_addr < max_addr):
        raise ValueError(
            f"load_addr for sel={load_sel} must be in 0..{max_addr - 1}, got {load_addr}"
        )
    return ((load_addr & 0xFF) << 10) | ((load_grp & 0x7F) << 3) | (load_sel & 0x7)


def compute_load_crc32(words_u64: Sequence[int]) -> int:
    """Compute reflected IEEE CRC-32 over 64-bit group words in little-endian order.

    Parameters
    ----------
    words_u64 : sequence of int
        Unsigned 64-bit group words in the exact order written to
        ``LOAD_WDATA_LO`` / ``LOAD_WDATA_HI``.

    Returns
    -------
    int
        Unsigned 32-bit IEEE 802.3 CRC-32 matching ``LOAD_CRC32`` in
        ``hdl/shd_event/sc_shd_event_axi.v`` and ``zlib.crc32``.
    """
    payload = b"".join(struct.pack("<Q", int(w) & 0xFFFF_FFFF_FFFF_FFFF) for w in words_u64)
    return zlib.crc32(payload) & 0xFFFF_FFFF


def _validate_int_array(
    name: str, raw: object, expected_shape: tuple[int, ...]
) -> npt.NDArray[np.int64]:
    arr = np.asarray(raw)
    if arr.shape != expected_shape or arr.dtype.kind not in ("i", "u"):
        raise ValueError(
            f"{name}: expected integer array of shape {expected_shape}, got {arr.dtype} {arr.shape}"
        )
    return np.asarray(arr, dtype=np.int64)


@runtime_checkable
class _IntNetworkLike(Protocol):
    q1: object
    q2: object
    q3: object
    d1: object
    d2: object


def _extract_tables(
    network: str | Path | Mapping[str, object] | object,
    delays: DelayTuple | Sequence[npt.ArrayLike] | None,
) -> tuple[WeightTuple, DelayTuple]:
    if isinstance(network, (str, Path)):
        if delays is not None:
            raise ValueError("delays must be None when loading from a shd_int_network.json path")
        raw_map = json.loads(Path(network).read_text(encoding="utf-8"))
        if not isinstance(raw_map, Mapping):
            raise ValueError("shd_int_network.json root must be a JSON object")
        return _extract_tables(raw_map, None)

    if isinstance(network, Mapping):
        if delays is not None:
            raise ValueError("delays must be None when network mapping already carries d1 and d2")
        q1 = _validate_int_array("q1", network["q1"], (N_HIDDEN, N_INPUT))
        q2 = _validate_int_array("q2", network["q2"], (N_HIDDEN, N_HIDDEN))
        q3 = _validate_int_array("q3", network["q3"], (N_OUTPUT, N_HIDDEN))
        d1 = _validate_int_array("d1", network["d1"], (N_INPUT,))
        d2 = _validate_int_array("d2", network["d2"], (N_HIDDEN,))
    elif isinstance(network, _IntNetworkLike):
        if delays is not None:
            raise ValueError("delays must be None when network object already carries d1 and d2")
        q1 = _validate_int_array("q1", network.q1, (N_HIDDEN, N_INPUT))
        q2 = _validate_int_array("q2", network.q2, (N_HIDDEN, N_HIDDEN))
        q3 = _validate_int_array("q3", network.q3, (N_OUTPUT, N_HIDDEN))
        d1 = _validate_int_array("d1", network.d1, (N_INPUT,))
        d2 = _validate_int_array("d2", network.d2, (N_HIDDEN,))
    elif isinstance(network, Sequence) and len(network) == 3:
        if delays is None or len(delays) != 2:
            raise ValueError("delays=(d1, d2) is required when passing (q1, q2, q3) weight tables")
        q1 = _validate_int_array("q1", network[0], (N_HIDDEN, N_INPUT))
        q2 = _validate_int_array("q2", network[1], (N_HIDDEN, N_HIDDEN))
        q3 = _validate_int_array("q3", network[2], (N_OUTPUT, N_HIDDEN))
        d1 = _validate_int_array("d1", delays[0], (N_INPUT,))
        d2 = _validate_int_array("d2", delays[1], (N_HIDDEN,))
    else:
        raise TypeError(
            "network must be a path to shd_int_network.json, a parsed mapping, an IntNetwork "
            "object, or a (q1, q2, q3) tuple paired with delays=(d1, d2)"
        )

    for w_name, w_arr in (("q1", q1), ("q2", q2), ("q3", q3)):
        if int(np.abs(w_arr).max()) > WEIGHT_MAX:
            raise ValueError(f"{w_name} outside -{WEIGHT_MAX}..{WEIGHT_MAX}")
    for d_name, d_arr in (("d1", d1), ("d2", d2)):
        if int(d_arr.min()) < 0 or int(d_arr.max()) > MAX_DELAY:
            raise ValueError(f"{d_name} outside 0..{MAX_DELAY}")
    return (q1, q2, q3), (d1, d2)


def pack_shd_int_network(
    network: str | Path | Mapping[str, object] | object,
    delays: DelayTuple | Sequence[npt.ArrayLike] | None = None,
    *,
    lanes: int = DEFAULT_LANES,
) -> list[tuple[int, int, int, int]]:
    """Serialise an SHD integer network into 64-bit AXI loader group words.

    Uses the exact table layout of ``hdl/fixtures/shd_int_network.json`` and the
    auto-increment address order of ``hdl/shd_event/sc_shd_event_axi.v``
    (``q1`` groups, ``q2`` groups, ``q3`` groups, ``d1`` rows, ``d2`` rows).

    Parameters
    ----------
    network : str or Path or Mapping or object or tuple
        Path to ``shd_int_network.json``, a mapping parsed from that JSON, an
        ``IntNetwork`` instance, or ``(q1, q2, q3)`` paired with ``delays``.
    delays : tuple of (array_like, array_like), optional
        ``(d1, d2)`` per-channel integer delays when ``network`` is ``(q1, q2, q3)``.
    lanes : int, default 8
        Parallel lane count in ``(1, 2, 4, 8)``; **must equal** the synthesised
        bitstream's ``P`` parameter (``8`` for the default build). A smaller
        ``lanes`` overruns the memory bounds and is refused by ``refused_range``,
        whereas a larger ``lanes`` sends fewer words in the wrong group layout
        while ``LOAD_COUNT`` and ``LOAD_CRC32`` still match the sent stream, so
        it cannot be detected from the bus and loads the wrong weights.

    Returns
    -------
    list of tuple of (int, int, int, int)
        Ordered ``(load_sel, load_addr, load_grp, word64)`` tuples.

    Raises
    ------
    ValueError
        If ``lanes`` is unsupported or any table shape/range is invalid.
    TypeError
        If ``network`` does not match one of the supported input forms.
    """
    if lanes not in VALID_AXI_LANES:
        raise ValueError(f"lanes must be one of {VALID_AXI_LANES}, got {lanes}")
    (q1, q2, q3), (d1, d2) = _extract_tables(network, delays)

    entries: list[tuple[int, int, int, int]] = []
    g_groups = N_HIDDEN // lanes
    pc_passes = (N_OUTPUT + lanes - 1) // lanes

    for r in range(N_INPUT):
        for gq in range(g_groups):
            w64 = 0
            for k in range(lanes):
                w64 |= (int(q1[gq * lanes + k, r]) & 0xFF) << (8 * k)
            entries.append((0, r, gq, w64))

    for r in range(N_HIDDEN):
        for gq in range(g_groups):
            w64 = 0
            for k in range(lanes):
                w64 |= (int(q2[gq * lanes + k, r]) & 0xFF) << (8 * k)
            entries.append((1, r, gq, w64))

    for r in range(N_HIDDEN):
        for gq in range(pc_passes):
            w64 = 0
            for k in range(lanes):
                cls_idx = gq * lanes + k
                if cls_idx < N_OUTPUT:
                    w64 |= (int(q3[cls_idx, r]) & 0xFF) << (8 * k)
            entries.append((2, r, gq, w64))

    for r in range(N_INPUT):
        entries.append((3, r, 0, int(d1[r]) & 0x1F))

    for r in range(N_HIDDEN):
        entries.append((4, r, 0, int(d2[r]) & 0x1F))

    return entries


def _pack_binary_row_words(row: npt.NDArray[np.bool_]) -> tuple[int, int, int, int, int]:
    bits = 0
    for ch, fired in enumerate(row.tolist()):
        if fired:
            bits |= 1 << ch
    return (
        bits & 0xFFFF_FFFF,
        (bits >> 32) & 0xFFFF_FFFF,
        (bits >> 64) & 0xFFFF_FFFF,
        (bits >> 96) & 0xFFFF_FFFF,
        (bits >> 128) & 0xFFF,
    )


def _pack_count_row_words(row: npt.NDArray[np.int64]) -> tuple[int, ...]:
    vals = row.tolist()
    words: list[int] = []
    for j in range(17):
        base = 8 * j
        w32 = 0
        for sub in range(8):
            w32 |= (int(vals[base + sub]) & 0xF) << (4 * sub)
        words.append(w32)
    w17 = 0
    for sub in range(4):
        w17 |= (int(vals[136 + sub]) & 0xF) << (4 * sub)
    words.append(w17)
    return tuple(words)


class SHDAccelerator:
    """Hardware accelerator client for SHD speech digit classification.

    Supports both the legacy cycle-stepped ``sc_shd_axi_wrapper`` core
    (:meth:`set_scales` and ``classify(..., mode="legacy")``) and the
    event-driven ``sc_shd_event_axi`` core (:meth:`load_network`,
    :meth:`run_event`, :meth:`classify_event`, and ``classify(..., mode="event")``).
    """

    def __init__(self, mmio: object) -> None:
        """Initialise with a PYNQ MMIO or DefaultIP handle.

        Parameters
        ----------
        mmio : object
            Memory-mapped IO providing ``read(offset)`` and ``write(offset, value)``
            either directly or on a ``.mmio`` attribute.

        Raises
        ------
        TypeError
            If ``mmio`` does not expose ``read`` and ``write`` callables.
        """
        bus = getattr(mmio, "mmio", mmio)
        if not (callable(getattr(bus, "read", None)) and callable(getattr(bus, "write", None))):
            raise TypeError("mmio must provide read(offset) and write(offset, value)")
        self._mmio = cast(RegisterBus, bus)

    def set_scales(self, scale_l1: int, scale_l2: int, scale_l3: int) -> None:
        """Set Q16.16 per-layer scales on the legacy ``sc_shd_axi_wrapper`` only.

        Offsets ``0x08..0x10`` are reserved (``SLVERR``) on the event bitstream
        ``sc_shd_event_axi``, whose layer scaling is fixed at synthesis by its
        parameters; event-mode callers never invoke this method.

        Parameters
        ----------
        scale_l1, scale_l2, scale_l3 : int
            Signed 32-bit Q16.16 fixed-point scale factors for ``sc_shd_axi_wrapper``.
        """
        self._mmio.write(_SCALE_L1, scale_l1 & 0xFFFFFFFF)
        self._mmio.write(_SCALE_L2, scale_l2 & 0xFFFFFFFF)
        self._mmio.write(_SCALE_L3, scale_l3 & 0xFFFFFFFF)

    def load_network(
        self,
        network: str | Path | Mapping[str, object] | object,
        delays: DelayTuple | Sequence[npt.ArrayLike] | None = None,
        *,
        lanes: int = DEFAULT_LANES,
        auto_increment: bool = True,
    ) -> int:
        """Perform an integrity-checked load of weights and delays over AXI4-Lite.

        Clears the loader state, streams the 64-bit group words exported in
        ``hdl/fixtures/shd_int_network.json`` order, writes ``EXPECTED_CRC32``
        computed via ``zlib.crc32`` over the identical little-endian byte stream,
        requests ``LOAD_CTRL.seal``, and verifies ``LOAD_COUNT``, ``LOAD_CRC32``,
        and ``LOAD_STATUS.sealed``. This CRC-32 check guards against accidental
        transfer corruption and does not authenticate the network.

        Parameters
        ----------
        network : str or Path or Mapping or object or tuple
            Path to ``shd_int_network.json``, parsed JSON mapping, ``IntNetwork``
            instance, or ``(q1, q2, q3)`` weight tuple paired with ``delays``.
        delays : tuple of (array_like, array_like), optional
            ``(d1, d2)`` delay arrays when ``network`` is ``(q1, q2, q3)``.
        lanes : int, default 8
            Parallel lane count in ``(1, 2, 4, 8)``; **must equal** the
            synthesised bitstream's ``P`` parameter (``8`` for the default
            build). A smaller value is refused by ``LOAD_STATUS.refused_range``,
            whereas a larger value sends fewer words in the wrong group layout
            and seals with the wrong weights because no bus register exposes ``P``.
        auto_increment : bool, default True
            Use ``LOAD_CTRL.auto_increment`` when ``True``, or write
            ``LOAD_TARGET`` explicitly before each word when ``False``.

        Returns
        -------
        int
            Verified unsigned 32-bit IEEE CRC-32 of the committed word stream.

        Raises
        ------
        SHDDriverError
            If the core is busy, an out-of-range target is refused, ``LOAD_COUNT``
            or ``LOAD_CRC32`` disagrees, or ``LOAD_STATUS`` reports
            ``crc_mismatch``, ``refused_busy``, ``refused_range``, or unsealed.
        """
        entries = pack_shd_int_network(network, delays=delays, lanes=lanes)
        expected_crc = compute_load_crc32([w for _sel, _addr, _grp, w in entries])

        auto_bit = LOAD_CTRL_AUTO_INC if auto_increment else 0
        self._mmio.write(REG_LOAD_CTRL, LOAD_CTRL_CLEAR | auto_bit)
        if auto_increment:
            self._mmio.write(REG_LOAD_TARGET, 0)

        for sel, addr, grp, word64 in entries:
            if not auto_increment:
                self._mmio.write(REG_LOAD_TARGET, encode_load_target(sel, grp, addr, lanes=lanes))
            self._mmio.write(REG_LOAD_WDATA_LO, word64 & 0xFFFF_FFFF)
            self._mmio.write(REG_LOAD_WDATA_HI, (word64 >> 32) & 0xFFFF_FFFF)

        self._mmio.write(REG_EXPECTED_CRC32, expected_crc)
        self._mmio.write(REG_LOAD_CTRL, LOAD_CTRL_SEAL | auto_bit)

        load_status = self._mmio.read(REG_LOAD_STATUS) & 0xFFFF_FFFF
        if load_status & LOAD_STATUS_REFUSED_BUSY:
            raise SHDDriverError(
                f"integrity-checked load refused while accelerator was busy "
                f"(LOAD_STATUS=0x{load_status:08x})",
                status=load_status,
            )
        if load_status & LOAD_STATUS_REFUSED_RANGE:
            raise SHDDriverError(
                f"integrity-checked load refused out-of-range LOAD_TARGET "
                f"(LOAD_STATUS=0x{load_status:08x})",
                status=load_status,
            )
        if (load_status & LOAD_STATUS_CRC_MISMATCH) or not (load_status & LOAD_STATUS_SEALED):
            raise SHDDriverError(
                f"integrity-checked load failed to seal (LOAD_STATUS=0x{load_status:08x})",
                status=load_status,
            )
        hw_count = self._mmio.read(REG_LOAD_COUNT) & 0xFFFF_FFFF
        if hw_count != len(entries):
            raise SHDDriverError(
                f"LOAD_COUNT mismatch after seal: hardware {hw_count} != expected {len(entries)}",
                status=load_status,
            )
        hw_crc = self._mmio.read(REG_LOAD_CRC32) & 0xFFFF_FFFF
        if hw_crc != expected_crc:
            raise SHDDriverError(
                f"LOAD_CRC32 mismatch after seal: 0x{hw_crc:08x} != 0x{expected_crc:08x}",
                status=load_status,
            )
        return expected_crc

    def _poll_frame_ready(self, frame_index: int, max_poll_iters: int) -> None:
        status = 0
        for _ in range(max_poll_iters):
            status = self._mmio.read(REG_CTRL) & 0xFFFF_FFFF
            if status & CTRL_FAULT_MASK:
                raise SHDDriverError(
                    f"event inference fault waiting for frame_ready at frame {frame_index} "
                    f"(CTRL=0x{status:08x})",
                    status=status,
                )
            if (status & (CTRL_BUSY | CTRL_DONE)) == 0:
                raise SHDDriverError(
                    f"event accelerator idle without done after start at frame {frame_index} "
                    f"(CTRL=0x{status:08x})",
                    status=status,
                )
            if status & CTRL_FRAME_READY:
                return
            if status & CTRL_DONE:
                raise SHDDriverError(
                    f"event accelerator asserted done before frame {frame_index} was committed "
                    f"(CTRL=0x{status:08x})",
                    status=status,
                )
        raise SHDDriverError(
            f"timed out after {max_poll_iters} polls waiting for CTRL.frame_ready at frame "
            f"{frame_index} (CTRL=0x{status:08x})",
            status=status,
        )

    def _poll_done(self, max_poll_iters: int) -> None:
        status = 0
        for _ in range(max_poll_iters):
            status = self._mmio.read(REG_CTRL) & 0xFFFF_FFFF
            if status & CTRL_FAULT_MASK:
                raise SHDDriverError(
                    f"event inference fault waiting for CTRL.done (CTRL=0x{status:08x})",
                    status=status,
                )
            if status & CTRL_DONE:
                return
            if (status & CTRL_BUSY) == 0:
                raise SHDDriverError(
                    f"event accelerator idle without done after start (CTRL=0x{status:08x})",
                    status=status,
                )
        raise SHDDriverError(
            f"timed out after {max_poll_iters} polls waiting for CTRL.done (CTRL=0x{status:08x})",
            status=status,
        )

    def run_event(
        self,
        spike_raster: np.ndarray,
        *,
        max_poll_iters: int = DEFAULT_MAX_POLL_ITERS,
    ) -> npt.NDArray[np.int32]:
        """Run one event-mode sample on ``sc_shd_event_axi`` and return class sums.

        Verifies ``LOAD_STATUS.sealed`` before touching ``CTRL.start``, validates
        ``1 <= T <= 1024`` and input channel counts before writing to the bus,
        drives boolean rasters through ``SPIKE_IN_0..4`` and integer rasters
        (``0..15``) through ``FRAME_COUNT_0..17`` with no tail padding, and uses
        bounded polling with immediate status-word abort on ``error``,
        ``overflow``, ``commit_refused``, or ``!busy && !done``.

        Parameters
        ----------
        spike_raster : np.ndarray
            2-D array of shape ``(T, 140)`` with ``1 <= T <= 1024``; either
            ``bool`` dtype (sent via ``SPIKE_IN_0..4``) or integer dtype with
            values in ``0..15`` (sent via ``FRAME_COUNT_0..17``).
        max_poll_iters : int, default 100000
            Maximum MMIO status reads per ``frame_ready`` or ``done`` wait.

        Returns
        -------
        np.ndarray
            Signed ``int32`` array of shape ``(20,)`` holding ``OUT_V_0..19``.

        Raises
        ------
        ValueError
            If ``max_poll_iters < 1``, ``spike_raster`` shape/dtype is invalid,
            ``T`` is outside ``1..1024``, or any integer count is outside ``0..15``.
        SHDDriverError
            If the loader is not sealed, a hardware fault bit fires, the core
            drops ``busy`` without ``done``, or polling exceeds ``max_poll_iters``.
        """
        if max_poll_iters < 1:
            raise ValueError(f"max_poll_iters must be >= 1, got {max_poll_iters}")

        arr = np.asarray(spike_raster)
        if arr.ndim != 2 or arr.shape[1] != N_INPUT:
            raise ValueError(f"expected (T, {N_INPUT}) raster, got shape {arr.shape}")
        t_orig = int(arr.shape[0])
        if not (1 <= t_orig <= EVENT_T_MAX):
            raise ValueError(f"event raster length {t_orig} outside 1..{EVENT_T_MAX}")

        is_bool = arr.dtype == np.bool_
        if not is_bool:
            if arr.dtype.kind not in ("i", "u"):
                raise ValueError(f"spike_raster must have bool or integer dtype, got {arr.dtype}")
            int_arr = np.asarray(arr, dtype=np.int64)
            if int(int_arr.min()) < 0 or int(int_arr.max()) > COUNT_MAX:
                raise ValueError(
                    f"integer frame counts must lie in 0..{COUNT_MAX} before bus transfer"
                )
        else:
            bool_arr = np.asarray(arr, dtype=np.bool_)

        load_status = self._mmio.read(REG_LOAD_STATUS) & 0xFFFF_FFFF
        if not (load_status & LOAD_STATUS_SEALED):
            raise SHDDriverError(
                f"refusing to start event inference while loader is unsealed "
                f"(LOAD_STATUS=0x{load_status:08x})",
                status=load_status,
            )

        self._mmio.write(REG_T_ORIG, t_orig)
        self._mmio.write(REG_CTRL, CTRL_START)

        for t in range(t_orig):
            self._poll_frame_ready(t, max_poll_iters)
            if is_bool:
                w0, w1, w2, w3, w4 = _pack_binary_row_words(bool_arr[t])
                self._mmio.write(REG_SPIKE_IN_0, w0)
                self._mmio.write(REG_SPIKE_IN_1, w1)
                self._mmio.write(REG_SPIKE_IN_2, w2)
                self._mmio.write(REG_SPIKE_IN_3, w3)
                self._mmio.write(REG_SPIKE_IN_4, w4)
            else:
                for j, word in enumerate(_pack_count_row_words(int_arr[t])):
                    self._mmio.write(REG_FRAME_COUNT_BASE + 4 * j, word)
            self._mmio.write(REG_SPIKE_COMMIT, 1)

        self._poll_done(max_poll_iters)
        return self.read_output_voltages()

    def classify_event(
        self,
        spike_raster: np.ndarray,
        *,
        max_poll_iters: int = DEFAULT_MAX_POLL_ITERS,
    ) -> int:
        """Run event-mode inference and return the predicted class index ``0..19``.

        Parameters
        ----------
        spike_raster : np.ndarray
            Boolean or ``0..15`` integer array of shape ``(T, 140)``.
        max_poll_iters : int, default 100000
            Maximum MMIO status reads per ``frame_ready`` or ``done`` wait.

        Returns
        -------
        int
            Predicted class index ``argmax(OUT_V_0..19)`` (lowest index on ties).
        """
        voltages = self.run_event(spike_raster, max_poll_iters=max_poll_iters)
        return int(np.argmax(voltages))

    def classify(
        self,
        spike_raster: np.ndarray,
        *,
        mode: str = "legacy",
        max_poll_iters: int = DEFAULT_MAX_POLL_ITERS,
    ) -> int:
        """Run inference on a spike raster and return the predicted class index.

        Parameters
        ----------
        spike_raster : np.ndarray
            Input raster of shape ``(T, 140)``.
        mode : {"legacy", "event"}, default "legacy"
            ``"legacy"`` targets ``sc_shd_axi_wrapper`` (30 zero-padded tail
            commits); ``"event"`` targets ``sc_shd_event_axi`` via
            :meth:`classify_event`.
        max_poll_iters : int, default 100000
            Bounded poll limit used when ``mode="event"``.

        Returns
        -------
        int
            Predicted class index (0-19) = argmax of output voltages.

        Raises
        ------
        ValueError
            If ``mode`` is neither ``"legacy"`` nor ``"event"``.
        """
        if mode == "event":
            return self.classify_event(spike_raster, max_poll_iters=max_poll_iters)
        if mode != "legacy":
            raise ValueError(f"mode must be 'legacy' or 'event', got {mode!r}")

        t_orig = spike_raster.shape[0]
        self._mmio.write(_T_ORIG, t_orig)

        self._mmio.write(_CTRL, 1)

        t_total = t_orig + 30
        for t in range(t_total):
            if t < t_orig:
                row = spike_raster[t]
            else:
                row = np.zeros(140, dtype=np.uint8)

            bits = 0
            for i in range(140):
                if row[i]:
                    bits |= 1 << i

            self._mmio.write(_SPIKE_IN_0, bits & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_1, (bits >> 32) & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_2, (bits >> 64) & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_3, (bits >> 96) & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_4, (bits >> 128) & 0xFFF)
            self._mmio.write(_SPIKE_COMMIT, 1)

        while True:
            status = self._mmio.read(_CTRL)
            if status & 0x4:
                break

        return self.read_output_argmax()

    def read_output_argmax(self) -> int:
        """Read the 20 output voltage sums and return argmax."""
        voltages = self.read_output_voltages()
        return int(np.argmax(voltages))

    def read_output_voltages(self) -> npt.NDArray[np.int32]:
        """Read all 20 class output voltages as signed int32 array."""
        out = np.zeros(20, dtype=np.int32)
        for i in range(20):
            raw = self._mmio.read(_OUT_V_BASE + i * 4)
            if raw >= 0x80000000:
                raw -= 0x100000000
            out[i] = raw
        return out

    @property
    def is_sealed(self) -> bool:
        """Return whether the event-mode integrity-checked loader is sealed."""
        return bool(self._mmio.read(REG_LOAD_STATUS) & LOAD_STATUS_SEALED)

    @property
    def is_running(self) -> bool:
        """Check if inference is currently in progress."""
        return bool(self._mmio.read(_CTRL) & 0x2)

    @property
    def is_done(self) -> bool:
        """Check if inference has completed."""
        return bool(self._mmio.read(_CTRL) & 0x4)
