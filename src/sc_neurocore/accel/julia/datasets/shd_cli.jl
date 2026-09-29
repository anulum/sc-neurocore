# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia SHD native binary interface

# Emit one actual Julia SHD recording through the native worker binary protocol.

"""Bind supervised Linux lifetime before loading or compiling the recording reader."""
function bind_lifetime(args::Vector{String})
    length(args) in (4,5) || throw(ArgumentError("path index budget HDF5-library required"))
    if length(args) == 5
        Sys.islinux() || error("supervised Julia SHD requires Linux")
        ccall(:prctl, Cint, (Cint,Culong,Culong,Culong,Culong), 1, 9, 0, 0, 0) == 0 ||
            error("Julia SHD parent-death signal setup failed")
        ccall(:getppid, Cint, ()) == parse(Int,args[5]) || exit(124)
        ccall(:alarm, Cuint, (Cuint,), 30)
    end
    return nothing
end

"""Read one zero-based recording and emit its bounded row-major float64 binary result."""
function main(args::Vector{String})
    sample = SHDRecordings.read_shd_recording(args[1], parse(Int,args[2]);
                                             maximum_bytes=parse(Int,args[3]), hdf5_library=args[4])
    write(stdout, codeunits("SHD1"))
    write(stdout, htol(UInt64(length(sample.events))))
    write(stdout, htol(sample.label))
    write(stdout, sample.events)
    flush(stdout)
    return nothing
end

try
    bind_lifetime(ARGS)
    include("shd.jl")
    main(ARGS)
catch error
    showerror(stderr, error)
    println(stderr)
    exit(1)
end
