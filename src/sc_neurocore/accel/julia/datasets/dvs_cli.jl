# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia converted DVS recording decoder

# Emit one actual Julia DVS recording through the guarded binary protocol.
"""Bind Linux parent death and the deadline before loading or compiling the reader."""
function bind_lifetime(args::Vector{String})
    length(args) in (2,3) || throw(ArgumentError("path budget [expected-parent] required"))
    Sys.islinux() || throw(ArgumentError("DVS lifetime guards require Linux"))
    parent = ccall(:getppid,Cint,())
    expected = length(args) == 3 ? parse(Int,args[3]) : parent
    expected > 0 && parent == expected || throw(ArgumentError("DVS expected parent absent"))
    ccall(:prctl,Cint,(Cint,Culong,Culong,Culong,Culong),1,9,0,0,0) == 0 || error("DVS parent guard refused")
    ccall(:getppid,Cint,()) == expected || throw(ArgumentError("DVS parent changed"))
    ccall(:alarm,Cuint,(Cuint,),30)
    return nothing
end
"""Read one recording and emit its little-endian row-major float64 frame."""
function main(args::Vector{String})
    events = DVSRecordings.read_dvs_recording(args[1];maximum_bytes=parse(Int,args[2]))
    write(stdout,codeunits("DVS1"));write(stdout,htol(UInt64(length(events))))
    for row in axes(events,1), column in axes(events,2)
        write(stdout,htol(reinterpret(UInt64,events[row,column])))
    end
    flush(stdout)
    return nothing
end
try
    bind_lifetime(ARGS)
    include("dvs.jl")
    main(ARGS)
catch error
    showerror(stderr,error);println(stderr);exit(1)
end
