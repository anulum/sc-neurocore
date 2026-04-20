# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sleep/protocol_library

module ProtocolLibraryAccel

using Statistics, LinearAlgebra

mutable struct SleepProtocolState
    binaural_hz::Float64
    noise_color::Float64
    base_freq_hz::Float64
    volume::Float64
    isochronic_hz::Float64
    spatial_rotation::Float64
    name::Float64
    description::Float64
    stage_audio::Float64
    stage_targets::Float64
    total_duration_min::Float64
end

function SleepProtocolState()
    SleepProtocolState(2.0, 0.0, 200.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 480.0)
end

function get_audio_for_stage(s::SleepProtocolState, stage)
    return s.stage_audio.get(
        stage, s.stage_audio.get(SleepStage.WAKE, StageAudioParams())
    )
end

function get_target_stage(s::SleepProtocolState, progress)
    progress = max(0.0, min(1.0, progress))
    cumulative = 0.0
    for stage in (SleepStage.WAKE, SleepStage.N1, SleepStage.N2, SleepStage.N3, SleepStage.REM)
        cumulative += s.stage_targets.get(stage, 0.0)
        if progress <= cumulative
            return stage
    return SleepStage.REM
end

function to_dict(s::SleepProtocolState)
    return {
        "name": s.name,
        "description": s.description,
        "total_duration_min": s.total_duration_min,
        "stage_targets": {s.name: v for s, v in s.stage_targets.items()},
        "stage_audio": {
            s.name: {
                "binaural_hz": a.binaural_hz,
                "noise_color": a.noise_color,
                "base_freq_hz": a.base_freq_hz,
                "volume": a.volume,
                "isochronic_hz": a.isochronic_hz,
                "spatial_rotation": a.spatial_rotation,
            }
            for s, a in s.stage_audio.items()
        },
    }
end

function get_protocol(name)
    return PROTOCOL_REGISTRY[name]
end

function list_protocols()
    return sorted(PROTOCOL_REGISTRY.keys())
end

end # module ProtocolLibraryAccel
