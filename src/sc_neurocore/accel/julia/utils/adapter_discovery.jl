# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/adapter_discovery

module AdapterDiscoveryAccel

const ADAPTER_ENTRY_POINT_GROUP = "sc_neurocore.adapters"

const FIRST_PARTY_ADAPTERS = Dict{String,String}(
    "neuroml" => "sc_neurocore.adapters.importers:NeuroMLImporter",
    "sonata" => "sc_neurocore.adapters.importers:SONATAImporter",
    "spikeinterface" => "sc_neurocore.adapters.importers:SpikeInterfaceImporter",
    "holonomic_dna_storage" => "sc_neurocore.adapters.holonomic.dna_storage:DNAEncoder",
    "holonomic_grn" => "sc_neurocore.adapters.holonomic.grn:GeneticRegulatoryLayer",
    "holonomic_neuromodulation" => "sc_neurocore.adapters.holonomic.neuromodulation:NeuromodulatorSystem",
)

function adapter_entry_point_group()::String
    return ADAPTER_ENTRY_POINT_GROUP
end

function first_party_adapters()::Dict{String,String}
    return copy(FIRST_PARTY_ADAPTERS)
end

function discover_adapters()::Dict{String,String}
    # Python owns importlib.metadata loading and ComponentRegistry mutation.
    return first_party_adapters()
end

end # module AdapterDiscoveryAccel
