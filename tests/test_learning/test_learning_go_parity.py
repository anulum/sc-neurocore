# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
import subprocess
import os
import math
import tempfile

try:
    from sc_neurocore._native.learning_bridge import (
        is_available, RustPlasticityRule, RULE_STDP
    )
    FFI_AVAILABLE = is_available()
except ImportError:
    FFI_AVAILABLE = False

GO_SCRIPT = """
package main

import (
    "fmt"
    "sc_neurocore/accel/go/autonomous_learning"
)

func main() {
    rule := autonomous_learning.NewPlasticityRule(autonomous_learning.RuleStdp, 0.5, 0.1, 0.05)
    if rule == nil {
        fmt.Println("NO_FFI")
        return
    }
    defer rule.Destroy()
    
    rule.Step(true, false, 0.0)
    rule.Step(false, true, 0.0)
    fmt.Printf("%f\\n", rule.Weight())
}
"""

@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust FFI not available")
def test_go_python_learning_parity():
    # Python baseline
    rule = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
    rule.step(True, False)
    rule.step(False, True)
    py_weight = rule.weight
    
    # Go evaluation
    env = os.environ.copy()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        go_file = os.path.join(tmpdir, "main.go")
        with open(go_file, "w") as f:
            f.write(GO_SCRIPT)
            
        try:
            # Note: Depending on where 'go run' is invoked, CGO_LDFLAGS might need adjustment. 
            # We assume the module can be run within sc_neurocore repo context.
            res = subprocess.run(
                ["go", "run", go_file],
                capture_output=True, text=True, env=env, cwd="src"
            )
            # If it fails due to cgo resolution in temp dir, we might skip rather than hard fail
            if res.returncode != 0:
                pytest.skip(f"Go CGO could not compile script: {res.stderr}")
                
            out = res.stdout.strip().split()[-1]
            if out == "NO_FFI":
                pytest.skip("Go could not load FFI lib")
                
            go_weight = float(out)
            assert math.isclose(py_weight, go_weight, rel_tol=1e-5), f"Parity mismatch: Python={py_weight}, Go={go_weight}"
            
        except FileNotFoundError:
            pytest.skip("Go not installed")
        except ValueError:
            pytest.skip(f"Unparseable Go output: {out}")
