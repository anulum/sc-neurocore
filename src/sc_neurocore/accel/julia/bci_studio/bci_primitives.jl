# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bci_studio/bci_primitives

module BciPrimitivesAccel

using Statistics, LinearAlgebra

mutable struct BCIClosedLoopEngineState
    channels::Float64
    weights::Float64
    learners::Float64
end

function BCIClosedLoopEngineState()
    BCIClosedLoopEngineState(0.0, 0.0, 0.0)
end

function process_bci_frame(s::BCIClosedLoopEngineState, raw_ephys, reward)
    start_time = time.perf_counter()
    spikes = (abs(diff(raw_ephys, prepend=0)) > 0.5).astype(bool)
    total_voltage = dot(spikes, s.weights)
    if FFI_ENABLED
        for i in 1:s.channels
            s.learners[i].step(spikes[i], spikes[i], reward)
    command = 1 if total_voltage > (s.channels * 0.1) else 0
    latency = (time.perf_counter() - start_time) * 1000.0
    return {"command": command, "latency_ms": latency, "spikes": int(sum(spikes))}
end

end # module BciPrimitivesAccel
