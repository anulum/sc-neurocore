# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for _native/core_engine_bridge

module CoreEngineBridgeAccel

using Statistics, LinearAlgebra

function is_available()
    return _HAS_CORE_ENGINE
end

function sc_multiply(a, b)
    if ! _HAS_CORE_ENGINE
        return a & b
    return int(_lib.sc_multiply(_ct.c_uint32(a), _ct.c_uint32(b)))
end

function sc_mux(a, b, sel)
    if ! _HAS_CORE_ENGINE
        return (sel & a) | (~sel & b) & 0xFFFFFFFF
    return int(_lib.sc_mux(_ct.c_uint32(a), _ct.c_uint32(b), _ct.c_uint32(sel)))
end

function sc_popcount(a)
    if ! _HAS_CORE_ENGINE
        return bin(a).count("1")
    return int(_lib.sc_popcount(_ct.c_uint32(a)))
end

function sc_popcount64(a)
    if ! _HAS_CORE_ENGINE
        return bin(a).count("1")
    return int(_lib.sc_popcount64(_ct.c_uint64(a)))
end

function sc_popcount_packed(data)
    if ! _HAS_CORE_ENGINE
        return sum(bin(w).count("1") for w in data)
    n = length(data)
    arr = (_ct.c_uint64 * n)(*data)
    return int(_lib.sc_popcount_packed(arr, _ct.c_size_t(n)))
end

function sc_popcount_packed_np(data)
    import numpy as np
    data = np.ascontiguousarray(data, dtype=np.uint64)
    ptr = data.ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    return int(_lib.sc_popcount_packed(ptr, _ct.c_size_t(data.size)))
end

function sc_scc_packed(a, b)
    n = min(length(a), length(b))
    if ! _HAS_CORE_ENGINE || n == 0
        return 0.0
    arr_a = (_ct.c_uint64 * n)(*a[:n])
    arr_b = (_ct.c_uint64 * n)(*b[:n])
    return float(_lib.sc_scc_packed(arr_a, arr_b, _ct.c_size_t(n)))
end

function sc_scc_packed_np(a, b)
    import numpy as np
    a = np.ascontiguousarray(a, dtype=np.uint64)
    b = np.ascontiguousarray(b, dtype=np.uint64)
    n = min(a.size, b.size)
    if n == 0
        return 0.0
    ptr_a = a[:n].ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    ptr_b = b[:n].ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    return float(_lib.sc_scc_packed(ptr_a, ptr_b, _ct.c_size_t(n)))
end

end # module CoreEngineBridgeAccel
