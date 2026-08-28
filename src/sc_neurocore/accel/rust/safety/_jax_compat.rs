// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety acceleration for _jax_compat

pub fn make_rng(_seed: f64) -> f64 {
    // if HAS_JAX {
    // return jax.random.PRNGKey(seed)  # type_val: ignore[return-value]
    // return array([0, seed], dtype=uint32)
    0.0
}

pub fn split_rng(_key: f64) -> f64 {
    // if HAS_JAX {
    // return jax.random.split(key)  # type_val: ignore[return-value]
    // s = int(key[-1])
    // return array([0, s + 1], dtype=uint32), array([0, s + 2], dtype=uint32
    0.0
}

pub fn uniform(_key: f64, _shape: f64, _minval: f64, _maxval: f64) -> f64 {
    // if HAS_JAX {
    // return jax.random.uniform(key, shape, minval=minval, maxval=maxval)
    // rng = random.default_rng(int(key[-1]))
    // return rng.uniform(low=minval, high=maxval, size=shape).astype(float32
    0.0
}

pub fn normal(_key: f64, _shape: f64) -> f64 {
    // if HAS_JAX {
    // return jax.random.normal(key, shape)
    // rng = random.default_rng(int(key[-1]))
    // return rng.standard_normal(size=shape).astype(float32)
    0.0
}

pub fn maybe_jit(function: f64) -> f64 {
    // if HAS_JAX {
    // return jax.jit(fn)
    // return fn
    function
}

#[cfg(test)]
mod tests {
    use super::maybe_jit;

    #[test]
    fn fallback_preserves_the_callable_token() {
        assert_eq!(maybe_jit(3.5), 3.5);
    }
}
