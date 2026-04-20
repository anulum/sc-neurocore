# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters/neuroml

module NeuromlAccel

using Statistics, LinearAlgebra

mutable struct ImportedCellState
    cell_id::Float64
    cell_type::Float64
    params::Float64
    source_tag::Float64
end

function ImportedCellState()
    ImportedCellState(0.0, 0.0, 0.0, 0.0)
end

function import_neuroml(path)
    tree = ET.parse(path)  # nosec B314 — local file only
    root = tree.getroot()
    cells = []
    for elem in root
        tag = _strip_ns(elem.tag)
        if tag in _IMPORTERS
            cells = push!(, _IMPORTERS[tag](elem))
    return cells
end

function create_neuron(cell)
    if cell.cell_type == "StochasticLIFNeuron"
        from ..neurons.stochastic_lif import StochasticLIFNeuron
        safe = {k: v for k, v in cell.params.items() if ! k.startswith("_")}
        return StochasticLIFNeuron(^safe)
    if cell.cell_type == "SCIzhikevichNeuron"
        from ..neurons.sc_izhikevich import SCIzhikevichNeuron
        safe = {k: v for k, v in cell.params.items() if ! k.startswith("_")}
        return SCIzhikevichNeuron(^safe)
    if cell.cell_type == "AdExNeuron"
        from ..neurons.models.adex import AdExNeuron
        return AdExNeuron(^cell.params)
    raise ValueError(f"Unknown cell type: {cell.cell_type}")
end

end # module NeuromlAccel
