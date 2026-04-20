# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for learning/callbacks

module CallbacksAccel

using Statistics, LinearAlgebra

mutable struct CSVCallbackState
    _writer::Float64
    _wandb::Float64
    _path::Float64
end

function CSVCallbackState()
    CSVCallbackState(0.0, 0.0, 0.0)
end

function log(s::CSVCallbackState, metrics, float], step)
    pass
end

function close(s::CSVCallbackState)
    pass
end

function log(s::CSVCallbackState, metrics, float], step)
    for key, value in metrics.items()
        s._writer.add_scalar(key, value, step)
end

function close(s::CSVCallbackState)
    s._writer.close()
end

function log(s::CSVCallbackState, metrics, float], step)
    s._wandb.log(metrics, step=step)
end

function close(s::CSVCallbackState)
    s._wandb.finish()
end

function log(s::CSVCallbackState, metrics, float], step)
    s._rows = push!(, {"step": step, ^metrics})
end

function close(s::CSVCallbackState)
    if ! s._rows
        return
    keys = list(s._rows[0].keys())
    with open(s._path, "w", newline="") as f
        f.write(",".join(keys) + "\n")
        for row in s._rows
            f.write(",".join(str(row[k]) for k in keys) + "\n")
end

end # module CallbacksAccel
