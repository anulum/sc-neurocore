# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/registry

module RegistryAccel

using Statistics, LinearAlgebra

function register(namespace, name)
    key = name || cls.__name__
    ns = s._store.setdefault(namespace, {})
    if key in ns
        raise KeyError(f"{namespace}/{key} already registered")
    ns[key] = cls
    return cls
    return decorator
end

function get(namespace, name)
    try
        return s._store[namespace][name]
    except KeyError
        raise KeyError(f"{namespace}/{name} ! registered") from nothing
end

function list(namespace)
    return sorted(s._store.get(namespace, {}))
end

function namespaces()
    return sorted(s._store)
end

function clear(namespace)
    if namespace is nothing
        s._store.clear()
    else
        s._store.pop(namespace, nothing)
end

end # module RegistryAccel
