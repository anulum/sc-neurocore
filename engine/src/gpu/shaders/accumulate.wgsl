// SPDX-License-Identifier: AGPL-3.0-or-later
// AND + popcount + reduction kernel for stochastic computing DenseLayer.
// One workgroup per (neuron, sample) pair. 256 threads cooperatively process
// all input words for one neuron and reduce to a single f32 output.

struct AccumParams {
    n_inputs: u32,
    n_neurons: u32,
    words_per_input: u32,
    inv_length: f32,
    n_samples: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> packed_weights: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> packed_inputs: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: AccumParams;

var<workgroup> partial_sums: array<u32, 256>;

// SWAR popcount for u32 (Hamming weight).
fn popcount32(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24u;
}

@compute @workgroup_size(256, 1, 1)
fn accumulate_main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let neuron_idx = wid.x;
    let sample_idx = wid.y;

    if (neuron_idx >= params.n_neurons || sample_idx >= params.n_samples) {
        partial_sums[lid] = 0u;
        workgroupBarrier();
        return;
    }

    let total_words = params.n_inputs * params.words_per_input;
    let weight_base = neuron_idx * total_words;
    let input_base  = sample_idx * total_words;

    // Each thread strides across the word array with step 256.
    var local_sum = 0u;
    var idx = lid;
    loop {
        if (idx >= total_words) { break; }
        let w   = packed_weights[weight_base + idx];
        let inp = packed_inputs[input_base + idx];
        let and_lo = w.x & inp.x;
        let and_hi = w.y & inp.y;
        local_sum = local_sum + popcount32(and_lo) + popcount32(and_hi);
        idx = idx + 256u;
    }

    partial_sums[lid] = local_sum;
    workgroupBarrier();

    // Tree reduction across 256 threads.
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            partial_sums[lid] = partial_sums[lid] + partial_sums[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final normalised result.
    if (lid == 0u) {
        let out_idx = sample_idx * params.n_neurons + neuron_idx;
        output[out_idx] = f32(partial_sums[0]) * params.inv_length;
    }
}
