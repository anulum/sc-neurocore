# Autonomous Learning API Reference

The Autonomous Learning module provides high-performance, stochastic-compatible online plasticity rules written in Rust and bridged to Python, Go, and Julia via C-FFI.

## Available Rules

The engine supports 4 primary rules, identified via C-FFI integer enumerations:

- `RULE_ELIGENT = 0`: Eligibility traces with intrinsic rate homeostasis.
- `RULE_STDP = 1`: Classic Spike-Timing Dependent Plasticity.
- `RULE_REWARD_STDP = 2`: Neuromodulatory/Reward-gated STDP (R-STDP).
- `RULE_BCM = 3`: Bienenstock-Cooper-Munro sliding threshold metaplasticity.

## Python Integration

The Python bridge now keeps the public Rust, Rust-WGPU, and Torch wrapper
methods inside the scoped docstring policy. Single-step calls, batched calls,
state dictionaries, weight reads, resets, and the PyTorch-unavailable fallback
are covered by the dedicated native bridge tests.

```python
from sc_neurocore._native.learning_bridge import RustPlasticityRule, RULE_STDP

# Initialize an STDP rule via Rust backend
rule = RustPlasticityRule(
    rule_type=RULE_STDP,
    weight=0.5,
    param_a=0.01, # a_plus / learning rate
    param_b=0.012 # a_minus
)

# Provide binary spikes and float rewards to the engine
rule.step(pre_spike=True, post_spike=False)
```

## Go Integration

The Go services wrap the `libautonomous_learning.so` object using `cgo`:

```go
import "sc_neurocore/accel/go/autonomous_learning"

rule := autonomous_learning.NewPlasticityRule(autonomous_learning.RuleStdp, 0.5, 0.1, 0.05)
defer rule.Destroy()

rule.Step(true, false, 0.0)
currWeight := rule.Weight()
```

## Julia Integration

```julia
include("src/sc_neurocore/accel/julia/_native/learning_bridge.jl")
using .LearningBridgeAccel

rule = LearningBridgeAccel.RustPlasticityRule(LearningBridgeAccel.RULE_STDP, 0.5f0, 0.1f0, 0.05f0)
LearningBridgeAccel.step(rule, true, false)
w = LearningBridgeAccel.weight(rule)
```

## Hardware Notes
Any state updates (such as BCM sliding thresholds or Eligent eligibility traces) occur within the native Rust memory space. Therefore, creating a rule allocates a `*mut c_void` trace which must be deterministically dropped by the language's native garbage collector (`__del__` in Python, `Drop` in Rust wrapper, or `finalizer` in Julia).
