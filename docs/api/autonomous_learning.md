# Autonomous Learning API Reference

The Autonomous Learning module provides high-performance, stochastic-compatible online plasticity rules written in Rust and bridged to Python, Go, and Julia via C-FFI.

## Available Rules

The engine supports 4 primary rules, identified via C-FFI integer enumerations:

- `RULE_ELIGENT = 0`: Eligibility traces with intrinsic rate homeostasis.
- `RULE_STDP = 1`: Classic Spike-Timing Dependent Plasticity.
- `RULE_REWARD_STDP = 2`: Neuromodulatory/Reward-gated STDP (R-STDP).
- `RULE_BCM = 3`: Bienenstock-Cooper-Munro sliding threshold metaplasticity.

## Python Integration

`sc_neurocore._native.learning_bridge` is the stable compatibility facade. Its
implementation is separated into runtime/ABI binding, validation, scalar Rust
owners, Rayon layers, WGPU layers, Torch dynamics, mixed precision, autograd,
and backend selection. The facade preserves historical class identities and
does not make Rust backends depend on PyTorch being installed.

```python
from sc_neurocore._native.learning_bridge import RULE_STDP, RustPlasticityRule

with RustPlasticityRule(
    rule_type=RULE_STDP, weight=0.5, param_a=0.01, param_b=0.012
) as rule:
    rule.step(pre_spike=True, post_spike=False, dt=0.001, reward=0.0)
    current_weight = rule.weight
```

`param_a` and `param_b` are rule-family parameters. For STDP, `param_a`
selects the potentiation rate and the native engine derives depression as half
that rate; `param_b` is reserved and native STDP trace constants remain 20
seconds. For R-STDP,
`param_b` is the eligibility time constant. For BCM it is the sliding-threshold
time constant, and for ELIGENT it is the eligibility time constant. Constructors
reject unknown rule identifiers, non-finite parameters, and weights outside
`[0, 1]`; step methods reject non-Boolean spikes, non-finite rewards, and
non-positive timesteps.

Use `RustRuleLayer` for an equal-length vector updated by Rayon and
`RustWgpuRuleLayer` for the optional WGPU backend. Both expose `get_weights()`,
`reset()`, explicit `close()`, and context-manager ownership. Rayon state
dictionaries contain an opaque, versioned byte buffer. Restore uses the
length-aware Rust parser and swaps the native handle only after the entire
payload is validated. WGPU state restore checks the weight vector and calls the
native `set_wgpu_weights` ABI; it never silently ignores requested state.

`create_plasticity_layer(count, backend=...)` accepts `"rust"`, `"rust-wgpu"`,
or `"torch"`. Torch is imported only for the Torch choice. `TorchRuleLayer`
validates public tensor shapes/domains, supports per-state mixed-precision
controls, and uses the custom biological autograd transition only when
`autograd=True`.

## Go Integration

The Go services wrap the `libautonomous_learning.so` object using `cgo`. Run
local snippets from `src/sc_neurocore/accel/go` so the module
`github.com/anulum/sc-neurocore/accel` resolves. For a source-bound library,
point both the linker and runtime loader at the artifact directory:

```bash
export SC_NEUROCORE_LIB_PATH=/absolute/path/libautonomous_learning.so
export CGO_LDFLAGS="-L$(dirname "$SC_NEUROCORE_LIB_PATH")${CGO_LDFLAGS:+ $CGO_LDFLAGS}"
export LD_LIBRARY_PATH="$(dirname "$SC_NEUROCORE_LIB_PATH")"
```

```go
import "github.com/anulum/sc-neurocore/accel/autonomous_learning"

rule := autonomous_learning.NewPlasticityRule(autonomous_learning.RuleStdp, 0.5, 0.1, 0.05)
defer rule.Destroy()

if err := rule.Step(true, false, 0.0); err != nil {
    return err
}
if err := rule.StepDt(false, true, 0.0, autonomous_learning.DefaultDt); err != nil {
    return err
}
currWeight, err := rule.TryWeight()
if err != nil {
    return err
}
```

Go constructors return `nil` for invalid configurations. Step/reset methods
return sentinel-compatible errors for closed handles, invalid timesteps,
non-finite rewards, and length mismatches. `TryWeight()` and
`TryGetWeights()` are the non-panicking read APIs; `Weight()` and
`GetWeights()` retain legacy panic semantics only for already-closed handles.
Explicit `Destroy()` prevents live native objects from leaking.

## Julia Integration

```julia
include("src/sc_neurocore/accel/julia/_native/learning_bridge.jl")
using .LearningBridgeAccel

rule = LearningBridgeAccel.RustPlasticityRule(
    LearningBridgeAccel.RULE_STDP, 0.5f0, 0.01f0, 20.0f0
)
LearningBridgeAccel.step(rule, true, false, 0.0f0, 0.001f0)
w = LearningBridgeAccel.weight(rule)
LearningBridgeAccel.destroy_rule(rule)
```

Set `SC_NEUROCORE_LIB_PATH` before Julia starts to select the same exact
artifact used by Python and Go. Julia rejects invalid configuration domains,
null construction, closed handles, non-finite values, non-positive timesteps,
and unequal or empty batch vectors.

## Hardware Notes

BCM thresholds, STDP traces, and ELIGENT eligibility state live behind opaque
native handles. Release them explicitly: `close()`/a `with` block in Python,
`Destroy()` in Go, and `destroy_rule()` in Julia. Finalizers are a last-resort
safety net, not the deterministic lifecycle contract.

`benchmarks/bench_autonomous_learning.py` measures the exact parent and
candidate source trees and exact native libraries through isolated Python,
Rust, Torch, Go, and Julia probes. The committed JSON is local regression
evidence, not a hardware-throughput claim; see the benchmark report for the
captured load and interpretation.
