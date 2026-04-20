# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/adapter_discovery

module AdapterDiscoveryAccel

using Statistics, LinearAlgebra

function discover_adapters()
    found = {}
    try
        eps = importlib.metadata.entry_points(group="sc_neurocore.adapters")
    except TypeError
        eps = importlib.metadata.entry_points().get("sc_neurocore.adapters", [])  # type: ignore[attr-defined]
    for ep in eps
        try
            cls = ep.load()
            name = ep.name
            registry.register("adapter", name)(cls)
            found[name] = cls
        except (ImportError, KeyError, AttributeError)
            continue
    return found
end

end # module AdapterDiscoveryAccel
