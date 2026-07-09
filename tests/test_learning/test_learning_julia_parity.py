# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

import pytest
import subprocess
import os
import math

try:
    from sc_neurocore._native.learning_bridge import is_available, RustPlasticityRule, RULE_STDP

    FFI_AVAILABLE = is_available()
except ImportError:
    FFI_AVAILABLE = False

JULIA_SCRIPT = """
include("src/sc_neurocore/accel/julia/_native/learning_bridge.jl")
using .LearningBridgeAccel

if !LearningBridgeAccel._HAS_LEARNING
    println("NO_FFI")
    exit()
end

rule = LearningBridgeAccel.RustPlasticityRule(LearningBridgeAccel.RULE_STDP, 0.5f0, 0.1f0, 0.05f0)
# Pre-before-post (LTP)
LearningBridgeAccel.step(rule, true, false, 0.0f0)
LearningBridgeAccel.step(rule, false, true, 0.0f0)
w = LearningBridgeAccel.weight(rule)
println(w)
"""


@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust FFI not available")
def test_julia_python_learning_parity():
    # Python baseline
    rule = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
    rule.step(True, False)
    rule.step(False, True)
    py_weight = rule.weight

    # Julia evaluation
    env = os.environ.copy()
    try:
        res = subprocess.run(
            ["julia", "-e", JULIA_SCRIPT], capture_output=True, text=True, env=env, check=True
        )
    except FileNotFoundError:
        pytest.skip("Julia not installed")

    out = res.stdout.strip().split()[-1]
    if out == "NO_FFI":
        pytest.skip("Julia could not load FFI lib")

    jl_weight = float(out)

    assert math.isclose(py_weight, jl_weight, rel_tol=1e-5), (
        f"Parity mismatch: Python={py_weight}, Julia={jl_weight}"
    )
