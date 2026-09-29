# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia N-MNIST public reader tests

using Test
include("nmnist.jl")
using .NMNISTRecordings

@testset "N-MNIST recorded fields and refusal" begin
    raw = UInt8[33, 32, 0x80, 3, 234, 1, 0, 0, 7, 212, 0, 33, 0x7f, 0xff, 0xff]
    golden = Float64[33, 32, 1, 1.002, 1, 0, 0, 2.004, 0, 33, 0, 8388.607]
    @test decode_nmnist(raw) == golden
    @test isempty(decode_nmnist(UInt8[]))
    destination = fill(9.0, 12)
    @test_throws ArgumentError decode_nmnist!(destination, raw[1:end-1])
    @test destination == fill(9.0, 12)
    @test_throws ArgumentError decode_nmnist!(destination[1:end-1], raw)
    @test_throws ArgumentError decode_nmnist(raw[1:end-1])
    @test decode_nmnist!(destination, raw) === destination
    @test destination == golden
    mktempdir() do root
        @test_throws SystemError read_nmnist(joinpath(root, "missing.bin"))
        @test_throws Base.IOError load_nmnist(root)
        mkpath(joinpath(root, "Train", "2"))
        mkpath(joinpath(root, "Train", "10"))
        mkpath(joinpath(root, "Test", "3"))
        write(joinpath(root, "Train", "2", "a.bin"), raw)
        write(joinpath(root, "Train", "10", "a.bin"), UInt8[])
        write(joinpath(root, "Train", "ignored.txt"), "ignored")
        write(joinpath(root, "Train", "2", "ignored.txt"), "ignored")
        write(joinpath(root, "Test", "3", "a.bin"), raw)
        samples, labels = load_nmnist(root)
        @test labels == Int64[10, 2]
        @test samples == [Float64[], golden]
        @test load_nmnist(root; train=false) == ([golden], Int64[3])
        write(joinpath(root, "Train", "2", "a.bin"), raw[1:end-1])
        @test_throws ArgumentError load_nmnist(root)
        write(joinpath(root, "Train", "2", "a.bin"), raw)
        mkpath(joinpath(root, "Train", "not-a-label"))
        @test_throws ArgumentError load_nmnist(root)
    end
end


@testset "N-MNIST borrowed buffer boundary" begin
    raw = UInt8[33, 32, 0x80, 3, 234]
    output = fill(9.0, 4)
    GC.@preserve raw output begin
        r = UInt(pointer(raw))
        o = UInt(pointer(output))
        @test decode_nmnist_pointer(r, 4, o, 4) == -1
        @test decode_nmnist_pointer(r, 5, o, 3) == -1
        @test decode_nmnist_pointer(0, 5, o, 4) == -1
        @test decode_nmnist_pointer(r, 5, 0, 4) == -1
        @test decode_nmnist_pointer(r, 5, o + 1, 4) == -1
        @test decode_nmnist_pointer(o, 5, o, 4) == -1
        @test decode_nmnist_pointer(r, -5, o, 4) == -1
        @test decode_nmnist_pointer(r, typemax(Int), o, 4) == -1
        @test decode_nmnist_pointer(r, 5, o, typemax(Int)) == -1
        @test decode_nmnist_pointer(typemax(Int), 5, o, 4) == -1
        @test decode_nmnist_pointer(r, 5, typemax(Int) - 7, 4) == -1
        @test output == fill(9.0, 4)
        @test decode_nmnist_pointer(0, 0, 0, 0) == 0
        @test decode_nmnist_pointer(r, 5, o, 4) == 0
        @test output == Float64[33, 32, 1, 1.002]
    end
end
