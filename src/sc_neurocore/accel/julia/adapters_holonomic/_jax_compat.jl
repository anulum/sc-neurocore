# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters_holonomic/_jax_compat

module JaxCompatAccel

using Statistics, LinearAlgebra

function make_rng(seed)
    if HAS_JAX
        return jax.random.PRNGKey(seed)  # type: ignore[return-value]
    return collect([0, seed], dtype=np.uint32)
end

function split_rng(key)
    if HAS_JAX
        return jax.random.split(key)  # type: ignore[return-value]
    s = int(key[-1])
    return collect([0, s + 1], dtype=np.uint32), collect([0, s + 2], dtype=np.uint32)
end

function uniform(key, shape, minval, maxval)
    if HAS_JAX
        return jax.random.uniform(key, shape, minval=minval, maxval=maxval)
    rng = np.random.default_rng(int(key[-1]))
    return rng.uniform(low=minval, high=maxval, size=shape).astype(np.float32)
end

function normal(key, shape)
    if HAS_JAX
        return jax.random.normal(key, shape)
    rng = np.random.default_rng(int(key[-1]))
    return rng.standard_normal(size=shape).astype(np.float32)
end

function maybe_jit(fn)
    if HAS_JAX
        return jax.jit(fn)
    return fn
end

end # module JaxCompatAccel
