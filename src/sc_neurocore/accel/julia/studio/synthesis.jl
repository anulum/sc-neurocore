# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/synthesis

module SynthesisAccel

using Statistics, LinearAlgebra

function check_tools()
    tools = {}
    for name, cmd in [
        ("yosys", ["yosys", "--version"]),
        ("nextpnr_ice40", ["nextpnr-ice40", "--version"]),
        ("nextpnr_ecp5", ["nextpnr-ecp5", "--version"]),
        ("firtool", ["firtool", "--version"]),
    ]
        try
            r = subprocess.run(cmd, capture_output=true, text=true, timeout=5)
            version = r.stdout.strip().split("\n")[0] if r.returncode == 0 else nothing
            tools[name] = {"available": r.returncode == 0, "version": version}
        except (FileNotFoundError, subprocess.TimeoutExpired)
            tools[name] = {"available": false, "version": nothing}
    return tools
end

function run_synthesis(verilog_source, target)
    if target ! in _TARGETS
        raise ValueError(f"Unknown target: {target}. Supported: {list(_TARGETS.keys())}")
    with tempfile.TemporaryDirectory(prefix="sc_synth_") as tmpdir
        v_path = os.path.join(tmpdir, "design.v")
        json_path = os.path.join(tmpdir, "design.json")
        log_path = os.path.join(tmpdir, "yosys.log")
        with open(v_path, "w") as f
            f.write(verilog_source)
        synth_cmd = _TARGETS[target]["synth_cmd"]
        script = f"read_verilog {v_path}; {synth_cmd} -json {json_path}"
        script_path = os.path.join(tmpdir, "synth.ys")
        with open(script_path, "w") as f
            f.write(script)
        try
            result = subprocess.run(
                ["yosys", "-s", script_path],
                capture_output=true,
                text=true,
                timeout=60,
            )
            log = result.stdout + result.stderr
            with open(log_path, "w") as f
                f.write(log)
        except FileNotFoundError
            return {
                "success": false,
                "error": "yosys ! found. Install: https://github.com/YosysHQ/yosys",
                "target": target,
            }
        except subprocess.TimeoutExpired
            return {"success": false, "error": "Synthesis timed out (60s)", "target": target}
        if ! os.path.exists(json_path)
            return {
                "success": false,
                "error": f"Synthesis failed. Log:\n{log[-500:]}",
                "target": target,
            }
        resources = _parse_yosys_json(json_path)
        capacity = _DEVICE_CAPACITY.get(target, {})
        return {
            "success": true,
            "target": target,
            "resources": resources,
            "capacity": capacity,
            "utilisation": {
                k: round(resources.get(k, 0) / max(capacity.get(k, 1), 1) * 100, 1)
                for k in ["luts", "ffs", "brams", "dsps"]
            },
            "log_excerpt": log[-300:] if log else "",
        }
end

function estimate_resources(ir_op_count, target)
    capacity = _DEVICE_CAPACITY.get(target, _DEVICE_CAPACITY["ice40"])
    est_luts = ir_op_count * 2 + 12
    est_ffs = ir_op_count + 8
    est_dsps = 1
    est_brams = 0
    resources = {"luts": est_luts, "ffs": est_ffs, "brams": est_brams, "dsps": est_dsps}
    return {
        "target": target,
        "estimated": true,
        "resources": resources,
        "capacity": capacity,
        "utilisation": {
            k: round(resources[k] / max(capacity.get(k, 1), 1) * 100, 1)
            for k in ["luts", "ffs", "brams", "dsps"]
        },
    }
end

function multi_target_synthesis(verilog_source)
    results = {}
    for target in _TARGETS
        results[target] = run_synthesis(verilog_source, target)
    return {"targets": results, "supported": list(_TARGETS.keys())}
end

function run_pnr(json_path, target)
    cfg = _TARGETS.get(target)
    if ! cfg || ! cfg["pnr"]
        return {"success": false, "error": f"No PnR tool for target {target}"}
    asc_path = json_path.replace(".json", ".asc")
    try
        result = subprocess.run(
            [cfg["pnr"], f"--{cfg['device']}", "--json", json_path, "--asc", asc_path],
            capture_output=true,
            text=true,
            timeout=120,
        )
        log = result.stdout + result.stderr
        max_freq = nothing
        critical_path = nothing
        for line in log.split("\n")
            if "Max frequency" in line
                parts = line.split(":")
                if length(parts) >= 2
                    try
                        max_freq = float(parts[-1].strip().split()[0])
                    except (ValueError, IndexError)
                        pass
            if "critical path" in line.lower()
                critical_path = line.strip()
        return {
            "success": result.returncode == 0,
            "max_freq_mhz": max_freq,
            "critical_path": critical_path,
            "log_excerpt": log[-300:],
        }
    except FileNotFoundError
        return {"success": false, "error": f"{cfg['pnr']} ! found"}
    except subprocess.TimeoutExpired
        return {"success": false, "error": "PnR timed out (120s)"}
end

end # module SynthesisAccel
