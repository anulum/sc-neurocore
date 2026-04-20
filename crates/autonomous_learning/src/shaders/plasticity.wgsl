// plasticity.wgsl
// WGSL kernels for ELIGENT, STDP, REWARD_STDP, and BCM plasticity rules

@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> pre_trace: array<f32>;
@group(0) @binding(2) var<storage, read_write> post_trace: array<f32>;
@group(0) @binding(3) var<storage, read> pre_probs: array<f32>;
@group(0) @binding(4) var<storage, read> post_probs: array<f32>;
@group(0) @binding(5) var<storage, read_write> param_extra: array<f32>; // ELIG: eligibility | BCM: act_avg
@group(0) @binding(6) var<storage, read_write> param_extra2: array<f32>; // ELIG: sum_weights | BCM: theta_m
@group(0) @binding(7) var<storage, read_write> param_extra3: array<f32>; // ELIG: threshold 
@group(0) @binding(8) var<storage, read> rewards: array<f32>;

struct RuleParams {
    rule_type: u32,     // 0=ELIGENT, 1=STDP, 2=R-STDP, 3=BCM
    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
    dt: f32,
    count: u32,
    seed: u32,
    param_c: f32,
    param_d: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(9) var<uniform> params: RuleParams;

// PCG hash PRNG for stochastic execution mapping float arrays internally
fn pcg_hash(input: u32) -> f32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    var result = (word >> 22u) ^ word;
    return f32(result) / 4294967296.0;
}

// Note on WGSL Tuning: The workgroup_size is statically set to 256. 
// For optimal scaling on diverse edge deployments (Jetson, Apple Silicon, etc.) 
// this size can be tuned (e.g., to 64 or 128) based on wgpu adapter limits natively.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // True 2D Dispatch scaling seamlessly to bypass 65k dimensions
    let index = global_id.y * 65535u * 256u + global_id.x;
    
    if (index >= params.count) {
        return;
    }

    let dt = params.dt;
    let pre_prob = pre_probs[index];
    let post_prob = post_probs[index];
    let reward = rewards[index];

    // Stochastic Float Evaluation
    var is_pre = false;
    var is_post = false;

    if (pre_prob >= 1.0) {
        is_pre = true;
    } else if (pre_prob > 0.0) {
        let rand_val = pcg_hash(params.seed + index * 2u);
        is_pre = rand_val < pre_prob;
    }

    if (post_prob >= 1.0) {
        is_post = true;
    } else if (post_prob > 0.0) {
        let rand_val = pcg_hash(params.seed + index * 2u + 1u);
        is_post = rand_val < post_prob;
    }

    // ELIGENT Rule (0)
    if (params.rule_type == 0u) {
        var w = weights[index];
        var threshold = param_extra3[index]; // Explicit mapping to threshold
        var elig = param_extra[index];
        var sum_w = param_extra2[index];
        
        let target_rate = params.a_plus;
        let eta_intrinsic = params.a_minus;
        let tau_e = params.param_c;

        var current_rate = 0.0;
        if (is_post) { current_rate = 1.0; }
        
        threshold += eta_intrinsic * (current_rate - target_rate) * dt;
        
        if (is_pre) { elig += 1.0; }
        elig *= exp(-dt / tau_e);
        
        let delta = elig * reward;
        w += delta;
        sum_w += delta;
        
        // Exact hardware mapping bounded constraints normalization
        // NOTE: This implements Local Synaptic Exhaustion (LSE) boundary logic
        // natively exactly matching the Rust CPU baseline implementation, operating 
        // deliberately distinct from Torch tensor-wide normalizers!
        if (sum_w > 0.0) {
            let scale = params.param_d / sum_w; // using target_sum_weights explicitly!
            w *= scale;
            sum_w = params.param_d;
        }

        weights[index] = clamp(w, 0.0, 1.0);
        param_extra3[index] = threshold;
        param_extra[index] = clamp(elig, -100.0, 100.0);
        param_extra2[index] = sum_w;
    }
    // STDP Rule (1)
    else if (params.rule_type == 1u) {
        var w = weights[index];
        var pre_t = pre_trace[index];
        var post_t = post_trace[index];
        
        pre_t = pre_t * exp(-dt / params.tau_plus);
        post_t = post_t * exp(-dt / params.tau_minus);
        
        if (is_post) { w += params.a_plus * pre_t; }
        if (is_pre) { w -= params.a_minus * post_t; }
        
        if (is_pre) { pre_t += 1.0; }
        if (is_post) { post_t += 1.0; }
        
        weights[index] = clamp(w, 0.0, 1.0);
        pre_trace[index] = pre_t;
        post_trace[index] = post_t;
    }
    // REWARD_STDP Rule (2)
    else if (params.rule_type == 2u) {
        var w = weights[index];
        var pre_t = pre_trace[index];
        var post_t = post_trace[index];
        var elig = param_extra[index];
        let tau_e = params.param_c; // Explicit mapping passed safely!
        
        pre_t = pre_t * exp(-dt / params.tau_plus);
        post_t = post_t * exp(-dt / params.tau_minus);
        
        if (is_post) { elig += params.a_plus * pre_t; }
        if (is_pre) { elig -= params.a_minus * post_t; }
        
        elig *= exp(-dt / tau_e);
        w += elig * reward;
        
        if (is_pre) { pre_t += 1.0; }
        if (is_post) { post_t += 1.0; }
        
        weights[index] = clamp(w, 0.0, 1.0);
        pre_trace[index] = pre_t;
        post_trace[index] = post_t;
        param_extra[index] = clamp(elig, -100.0, 100.0);
    }
    // BCM Rule (3)
    else if (params.rule_type == 3u) {
        var w = weights[index];
        var act_avg = param_extra[index];
        var theta_m = param_extra2[index];
        
        let is_pre_f = f32(is_pre);
        let is_post_f = f32(is_post);
        
        w += params.a_plus * is_post_f * (is_post_f - theta_m) * is_pre_f * dt;
        
        act_avg += (is_post_f - act_avg) * (dt / params.tau_plus);
        theta_m += (act_avg * act_avg - theta_m) * (dt / params.tau_plus);
        
        // Exact parity constraints bounding drift zeroing correctly
        if (theta_m < 0.01) { theta_m = 0.01; }
        
        weights[index] = clamp(w, 0.0, 1.0);
        param_extra[index] = act_avg;
        param_extra2[index] = theta_m;
    }
}
