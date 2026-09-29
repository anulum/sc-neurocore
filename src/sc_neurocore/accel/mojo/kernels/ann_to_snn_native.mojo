# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo dense IF native result ownership

"""C ownership ABI-one; build with FP contraction off on 64-bit Linux.

Callers retain complete live aligned input spans throughout replay. Result handles
are library-created live owners; buffer borrows last until exactly one free, with
no concurrent release. Address checks do not prove OS accessibility. Refused calls
leave destination slots unchanged. Initialization retains the process runtime.
"""

from std.memory import Pointer
from std.memory.alloc import alloc, dealloc, Layout, ThinAllocation
from std.runtime import initialize_runtime
from ann_to_snn_compute import ReplayResult
from ann_to_snn_native_types import ReplayRequest, BufferView, check_span
from ann_to_snn_native_request import execute


@export
def sc_if_abi_version() abi("C") -> UInt32:
    """Return exact dense IF request/buffer/ownership ABI version one.

    Returns:
        Exact request, result-buffer and release ABI version one.
    """
    return 1


@export
def sc_if_replay(request_addr: UInt, output_addr: UInt) abi("C") -> Int32:
    """Publish one independently owned replay only after complete successful computation.

    Args:
        request_addr: Live immutable aligned 64-byte ABI request address.
        output_addr: Exclusive aligned pointer-sized writable result-handle slot.

    Returns:
        Zero success; -1 domain, -2 resource, -3 arithmetic or -4 internal refusal.
    """
    initialize_runtime()
    try:
        check_span(request_addr, 1, 64)
        check_span(output_addr, 1, 8)
        var request = Pointer[ReplayRequest, ImmUntrackedOrigin](unsafe_from_address=Int(request_addr))
        var result = execute(request[])
        var storage = alloc(Layout[ReplayResult].single()).unsafe_leak()
        storage.unsafe_write(result^)
        var destination = Pointer[UInt, MutUntrackedOrigin](unsafe_from_address=Int(output_addr))
        destination[] = UInt(Int(storage))
        return 0
    except error:
        var message = String(error)
        if message == "IF invalid input":
            return -1
        if message == "IF resource limit":
            return -2
        if message == "IF overflow":
            return -3
        return -4


def publish(values: List[Float64], output_addr: UInt):
    """Publish a borrowed numeric view while its result owner remains live.

    Args:
        values: Result-owned contiguous vector, borrowed without copying.
        output_addr: Admitted exclusive writable 16-byte BufferView address.
    """
    var address = UInt(0)
    if len(values) != 0:
        address = UInt(Int(values.unsafe_ptr()))
    var destination = Pointer[BufferView, MutUntrackedOrigin](unsafe_from_address=Int(output_addr))
    destination[] = BufferView(address, UInt(len(values)))


@export
def sc_if_buffer(handle: UInt, kind: UInt32, index: UInt, output_addr: UInt) abi("C") -> Int32:
    """Borrow output, final state, state trace or event trace from one live owner.

    Args:
        handle: Successful unreleased library-created result handle.
        kind: Zero output, one final state, two state trace or three event trace.
        index: Layer index; output requires zero.
        output_addr: Exclusive aligned writable 16-byte BufferView address.

    Returns:
        Zero success or -1 invalid address/selector, leaving refused slots unchanged.
    """
    initialize_runtime()
    try:
        check_span(handle, 1, 8)
        check_span(output_addr, 1, 16)
    except:
        return -1
    var owner = Pointer[ReplayResult, ImmUntrackedOrigin](unsafe_from_address=Int(handle))
    if kind == 0 and index == 0:
        publish(owner[].output, output_addr)
    elif kind == 1 and index < UInt(len(owner[].final_state)):
        publish(owner[].final_state[Int(index)], output_addr)
    elif kind == 2 and index < UInt(len(owner[].state_trace)):
        publish(owner[].state_trace[Int(index)], output_addr)
    elif kind == 3 and index < UInt(len(owner[].spike_trace)):
        publish(owner[].spike_trace[Int(index)], output_addr)
    else:
        return -1
    return 0


@export
def sc_if_free(handle: UInt) abi("C"):
    """Release exactly one live result and all numeric storage; null is harmless.

    Args:
        handle: Unreleased library-created owner, with no outstanding/concurrent borrows.
    """
    if handle == 0:
        return
    var owner = Pointer[ReplayResult, MutUntrackedOrigin](unsafe_from_address=Int(handle))
    owner.unsafe_deinit_pointee()
    dealloc(ThinAllocation(unsafe_owned_ptr=owner).unsafe_with_layout(Layout[ReplayResult].single()))
