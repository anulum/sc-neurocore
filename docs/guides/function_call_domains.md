<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Function-call domains in fixed-point numeric contracts -->

# Function-call domains

The numeric contract reports each approximation call separately. Two calls
can share a unit implementation while accepting different encoded arguments.

```python
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.equation_builder import EquationNeuron

model = EquationNeuron({"v": "exp(I)+exp(I/4)"}, method="map")
contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
assert len(contract.function_units) == 1
assert len(contract.function_calls) == 2
assert [
    (call.ranges[0].minimum, call.ranges[0].maximum)
    for call in contract.function_calls
] == [(-128, 127), (-32, 31)]
assert [
    call.accuracy.codes_evaluated
    for call in contract.function_calls
    if call.accuracy is not None
] == [256, 64]
```

`QFormat(4, 4)` has eight total bits: four integer bits including the sign,
and four fractional bits. The first call admits the full signed word,
`[-8, 7.9375]`. Division by the literal four multiplies by its encoded
reciprocal and narrows the product, giving `[-2, 1.9375]` for the second call.
Both measurements compare the existing generated unit with the real
exponential clipped to the output word. A narrower argument range need not
reduce the measured maximum error.

## What the intervals mean

Parameters and constants are encoded before analysis. Base literals follow
the compiler's binary64 conversion; general literal power exponents retain
their independent precision. Sums remain wide until the datapath narrows them.
Products use the configured rounding and intermediate width. Constant
division uses the encoded reciprocal; dynamic division truncates towards zero.
Modulo, floor division, comparisons, conditional expressions, clipping and
minimum/maximum follow the expression emitters' fixed-point rules.

Every state and input starts at the complete signed-word range, regardless
of its initial value. Unknown subexpressions and wrapping discontinuities
widen the interval conservatively. Bounds apply to defined evaluations;
a possibly zero dynamic divisor is explicitly noted. These records do not
prove that such a divisor is avoided.

Each record retains its equation, Jacobian, event or integration location,
one-based line and zero-based UTF-8 byte column. Root aliases are reported as
their existing units; repeated-multiply integer powers are arithmetic, not
extra function units. General powers distinguish independently encoded
literal exponents from dynamic exponent words. Exponential Euler includes
its implicit `exprel` call. Integration stages can replicate an authored call;
the source interval covers their unconstrained state words.

## Numerical evidence

The measured domain is the Cartesian product of the inclusive argument
intervals, intersected with the real function's mathematical domain. It can
contain holes and combinations the neuron never produces. `domain_cardinality`
counts valid tuples in this hull; `excluded_domain_codes` counts tuples
outside the unit's declared real-function domain. Dynamic powers measure strictly
positive bases for every signed exponent. Literal integral powers also admit
negative bases, excluding zero when the exponent is negative. An empty domain has `max_error=None`,
not zero error. Outside-domain hardware sentinels are not approximation evidence.

Products of at most 65,536 valid tuples are exhausted; powers use a 4,096-tuple
limit. Wider domains use deterministic samples with endpoints and critical
codes. `codes_evaluated`, `exhaustive` and the complete worst-input tuple state
what was actually measured. Power references use 96-digit Decimal arithmetic;
other calls use the Python model's functions. References are clipped to the
output word. An existing declared arithmetic bound is reported separately
and a measured violation is refused.

The measurements concern unit error on encoded arguments. They do not include
input quantisation error, propagation through the model, state reachability,
timing, synthesis area or board behaviour. The report does not specialise the
generated datapath or change the `lut-v1` default. Legacy table calls retain
their old grids. Banker product rounding is supported by the RTL; the
generated bit-true C/Rust neuron kernel retains its existing refusal of that
rounding policy.

## Measuring analysis time

Run the metadata benchmark from the repository root:

```bash
python benchmarks/bench_function_call_domains.py --calls 100 --repeats 7
```

The recorded [benchmark result](https://github.com/anulum/sc-neurocore/blob/main/benchmarks/results/bench_function_call_domains.json)
uses a fresh interpreter for each authored model. Imports and model construction
are outside timing. The first collector invocation includes numerical sweeps;
later invocations can reuse accuracy caches. Every repeat compares the complete
serialised contract outside timing. Each cached sample averages 100 calls, and
the table gives the median of seven samples.

| Model | First collector call (ms) | Cached median per call (µs) |
| --- | ---: | ---: |
| Paired exponentials | 7.85 | 259.78 |
| Clipped hypot | 153.22 | 252.85 |
| Literal and dynamic powers | 831.27 | 170.53 |
| Exponential Euler with events | 14.39 | 377.74 |

These are local measurements on a shared host, with CPU, memory, affinity,
versions and load recorded in the result. Source hashes are checked before and
after capture and bound to the committed result by the benchmark evidence gate.
They measure Python analysis cost, not generated hardware throughput. No native
implementation of this metadata analysis is asserted; generated C, Rust and RTL
remain separate numerical validation targets.

The complete equation-IR benchmark was also rerun on this source overlay: `benchmarks/results/local_python_2026-10-29_equation_ir.json` contains eleven
measured transactions after two warmups for each of ten models. It includes
source replay and both native emitters, pins all compiler source files and the
unchanged native binary, and checks identical graph/output checksums on every
call. It makes no throughput or speedup claim.
