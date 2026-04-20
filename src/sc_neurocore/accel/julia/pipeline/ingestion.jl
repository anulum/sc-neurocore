# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for pipeline/ingestion

module IngestionAccel

using Statistics, LinearAlgebra

mutable struct DataIngestorState
    data::Float64
    labels::Float64
end

function DataIngestorState()
    DataIngestorState(0.0, 0.0)
end

function get_sample(s::DataIngestorState, idx)
    return {k: v[idx] for k, v in s.data.items()}
end

function prepare_dataset(s::DataIngestorState, raw_data, Any])
    processed_data = {}
    for k, v in raw_data.items()
        arr = collect(v)
        # Normalize to [0, 1]
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max > arr_min
            processed_data[k] = (arr - arr_min) / (arr_max - arr_min)
        else
            processed_data[k] = np.zeros_like(arr)
    return MultimodalDataset(
        data=processed_data, labels=zeros(length(list(processed_data.values())[0]))
    )
end

end # module IngestionAccel
