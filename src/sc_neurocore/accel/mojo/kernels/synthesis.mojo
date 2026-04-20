# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for synthesis

fn check_tools() -> Int:
    var _check_tools_line = 'tools = {}'
    var _check_tools_line = 'for name, cmd in ['
    var _check_tools_line = '("yosys", ["yosys", "--version"]),'
    var _check_tools_line = '("nextpnr_ice40", ["nextpnr-ice40", "--version"]),'
    var _check_tools_line = '("nextpnr_ecp5", ["nextpnr-ecp5", "--version"]),'
    var _check_tools_line = '("firtool", ["firtool", "--version"]),'
    var _check_tools_line = ']:'
    var _check_tools_line = 'try:'
    var _check_tools_line = 'r = subprocess.run(cmd, capture_output=True, text=True, time'
    return 0  # version = r.stdout.strip().split("\n")[0] if r.ret
    return 0  # tools[name] = {"available": r.returncode == 0, "ve
    var _check_tools_line = 'except (FileNotFoundError, subprocess.TimeoutExpired):'
    var _check_tools_line = 'tools[name] = {"available": False, "version": 0}'
    return 0  # return tools

fn run_synthesis(verilog_source: Int, target: Int) -> Int:
    var _run_synthesis_line = 'if target not in _TARGETS:'
    var _run_synthesis_line = 'raise ValueError(f"Unknown target: {target}. Supported: {lis'
    var _run_synthesis_line = 'with tempfile.TemporaryDirectory(prefix="sc_synth_") as tmpd'
    var _run_synthesis_line = 'v_path = os.path.join(tmpdir, "design.v")'
    var _run_synthesis_line = 'json_path = os.path.join(tmpdir, "design.json")'
    var _run_synthesis_line = 'log_path = os.path.join(tmpdir, "yosys.log")'
    var _run_synthesis_line = 'with open(v_path, "w") as f:'
    var _run_synthesis_line = 'f.write(verilog_source)'
    var _run_synthesis_line = 'synth_cmd = _TARGETS[target]["synth_cmd"]'
    var _run_synthesis_line = 'script = f"read_verilog {v_path}; {synth_cmd} -json {json_pa'
    var _run_synthesis_line = 'script_path = os.path.join(tmpdir, "synth.ys")'
    var _run_synthesis_line = 'with open(script_path, "w") as f:'
    var _run_synthesis_line = 'f.write(script)'
    var _run_synthesis_line = 'try:'
    var _run_synthesis_line = 'result = subprocess.run('
    var _run_synthesis_line = '["yosys", "-s", script_path],'
    var _run_synthesis_line = 'capture_output=True,'
    var _run_synthesis_line = 'text=True,'
    var _run_synthesis_line = 'timeout=60,'
    var _run_synthesis_line = ')'
    var _run_synthesis_line = 'log = result.stdout + result.stderr'
    var _run_synthesis_line = 'with open(log_path, "w") as f:'
    var _run_synthesis_line = 'f.write(log)'
    var _run_synthesis_line = 'except FileNotFoundError:'
    return 0  # return {
    var _run_synthesis_line = '"success": False,'
    var _run_synthesis_line = '"error": "yosys not found. Install: https://github.com/Yosys'
    var _run_synthesis_line = '"target": target,'
    var _run_synthesis_line = '}'
    var _run_synthesis_line = 'except subprocess.TimeoutExpired:'
    return 0  # return {"success": False, "error": "Synthesis time
    var _run_synthesis_line = 'if not os.path.exists(json_path):'
    return 0  # return {
    var _run_synthesis_line = '"success": False,'
    var _run_synthesis_line = '"error": f"Synthesis failed. Log:\\n{log[-500:]}",'
    var _run_synthesis_line = '"target": target,'
    var _run_synthesis_line = '}'
    var _run_synthesis_line = 'resources = _parse_yosys_json(json_path)'
    var _run_synthesis_line = 'capacity = _DEVICE_CAPACITY.get(target, {})'
    return 0  # return {
    var _run_synthesis_line = '"success": True,'
    var _run_synthesis_line = '"target": target,'
    var _run_synthesis_line = '"resources": resources,'
    var _run_synthesis_line = '"capacity": capacity,'
    var _run_synthesis_line = '"utilisation": {'
    var _run_synthesis_line = 'k: round(resources.get(k, 0) / max(capacity.get(k, 1), 1) * '
    var _run_synthesis_line = 'for k in ["luts", "ffs", "brams", "dsps"]'
    var _run_synthesis_line = '},'
    var _run_synthesis_line = '"log_excerpt": log[-300:] if log else "",'
    var _run_synthesis_line = '}'

fn _parse_yosys_json(json_path: Int) -> Int:
    var __parse_yosys_json_line = 'with open(json_path) as f:'
    var __parse_yosys_json_line = 'data = json.load(f)'
    var __parse_yosys_json_line = 'resources = {"luts": 0, "ffs": 0, "brams": 0, "dsps": 0, "ce'
    var __parse_yosys_json_line = 'for mod_name, mod in data.get("modules", {}).items():'
    var __parse_yosys_json_line = 'cells = mod.get("cells", {})'
    var __parse_yosys_json_line = 'resources["cells"] += len(cells)'
    var __parse_yosys_json_line = 'for cell_name, cell in cells.items():'
    var __parse_yosys_json_line = 'ctype = cell.get("type", "")'
    var __parse_yosys_json_line = 'if "LUT" in ctype or "SB_LUT" in ctype:'
    var __parse_yosys_json_line = 'resources["luts"] += 1'
    var __parse_yosys_json_line = 'elif "DFF" in ctype or "SB_DFF" in ctype:'
    var __parse_yosys_json_line = 'resources["ffs"] += 1'
    var __parse_yosys_json_line = 'elif "RAM" in ctype or "BRAM" in ctype or "SB_RAM" in ctype:'
    var __parse_yosys_json_line = 'resources["brams"] += 1'
    var __parse_yosys_json_line = 'elif "DSP" in ctype or "MUL" in ctype:'
    var __parse_yosys_json_line = 'resources["dsps"] += 1'
    var __parse_yosys_json_line = 'resources["wires"] += len(mod.get("netnames", {}))'
    return 0  # return resources

fn estimate_resources(ir_op_count: Int, target: Int) -> Int:
    var _estimate_resources_line = 'capacity = _DEVICE_CAPACITY.get(target, _DEVICE_CAPACITY["ic'
    var _estimate_resources_line = 'est_luts = ir_op_count * 2 + 12'
    var _estimate_resources_line = 'est_ffs = ir_op_count + 8'
    var _estimate_resources_line = 'est_dsps = 1'
    var _estimate_resources_line = 'est_brams = 0'
    var _estimate_resources_line = 'resources = {"luts": est_luts, "ffs": est_ffs, "brams": est_'
    return 0  # return {
    var _estimate_resources_line = '"target": target,'
    var _estimate_resources_line = '"estimated": True,'
    var _estimate_resources_line = '"resources": resources,'
    var _estimate_resources_line = '"capacity": capacity,'
    var _estimate_resources_line = '"utilisation": {'
    var _estimate_resources_line = 'k: round(resources[k] / max(capacity.get(k, 1), 1) * 100, 1)'
    var _estimate_resources_line = 'for k in ["luts", "ffs", "brams", "dsps"]'
    var _estimate_resources_line = '},'
    var _estimate_resources_line = '}'

fn multi_target_synthesis(verilog_source: Int) -> Int:
    var _multi_target_synthesis_line = 'results = {}'
    var _multi_target_synthesis_line = 'for target in _TARGETS:'
    var _multi_target_synthesis_line = 'results[target] = run_synthesis(verilog_source, target)'
    return 0  # return {"targets": results, "supported": list(_TAR

fn run_pnr(json_path: Int, target: Int) -> Int:
    var _run_pnr_line = 'cfg = _TARGETS.get(target)'
    var _run_pnr_line = 'if not cfg or not cfg["pnr"]:'
    return 0  # return {"success": False, "error": f"No PnR tool f
    var _run_pnr_line = 'asc_path = json_path.replace(".json", ".asc")'
    var _run_pnr_line = 'try:'
    var _run_pnr_line = 'result = subprocess.run('
    var _run_pnr_line = '[cfg["pnr"], f"--{cfg[\'device\']}", "--json", json_path, "--a'
    var _run_pnr_line = 'capture_output=True,'
    var _run_pnr_line = 'text=True,'
    var _run_pnr_line = 'timeout=120,'
    var _run_pnr_line = ')'
    var _run_pnr_line = 'log = result.stdout + result.stderr'
    var _run_pnr_line = 'max_freq = 0'
    var _run_pnr_line = 'critical_path = 0'
    var _run_pnr_line = 'for line in log.split("\\n"):'
    var _run_pnr_line = 'if "Max frequency" in line:'
    var _run_pnr_line = 'parts = line.split(":")'
    var _run_pnr_line = 'if len(parts) >= 2:'
    var _run_pnr_line = 'try:'
    var _run_pnr_line = 'max_freq = float(parts[-1].strip().split()[0])'
    var _run_pnr_line = 'except (ValueError, IndexError):'
    var _run_pnr_line = 'pass'
    var _run_pnr_line = 'if "critical path" in line.lower():'
    var _run_pnr_line = 'critical_path = line.strip()'
    return 0  # return {
    return 0  # "success": result.returncode == 0,
    var _run_pnr_line = '"max_freq_mhz": max_freq,'
    var _run_pnr_line = '"critical_path": critical_path,'
    var _run_pnr_line = '"log_excerpt": log[-300:],'
    var _run_pnr_line = '}'
    var _run_pnr_line = 'except FileNotFoundError:'
    return 0  # return {"success": False, "error": f"{cfg['pnr']}
    var _run_pnr_line = 'except subprocess.TimeoutExpired:'
    return 0  # return {"success": False, "error": "PnR timed out
