# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for benchmarks/tasks

module TasksAccel

using Statistics, LinearAlgebra

mutable struct BenchmarkTaskState
    name::Float64
    description::Float64
    input_shape::Float64
    n_classes::Float64
    metric::Float64
    neurobench_id::Float64
    dataset::Float64
    baseline_accuracy::Float64
end

function BenchmarkTaskState()
    BenchmarkTaskState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

end # module TasksAccel
