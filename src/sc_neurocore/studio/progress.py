# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WebSocket progress streaming for Studio long-running ops

from __future__ import annotations

import asyncio
import json
import threading
import queue
from collections.abc import Callable
from typing import Any

import numpy as np

from sc_neurocore.studio.codegen import classify_firing_pattern


def _characterize_with_progress(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    q: queue.Queue[dict[str, Any]],
) -> None:
    """Run characterisation with progress updates pushed to queue."""
    try:
        total_steps = 20 + 15 * 2 + 2
        step = 0

        q.put({"type": "progress", "step": "trace", "pct": 0, "msg": "Running default simulation"})
        trace = simulate_fn(**base_config)
        pattern = classify_firing_pattern(trace["spikes"], trace["n_steps"], trace["dt"])
        step += 1

        base_current = base_config.get("current", 10.0)
        i_max = max(abs(base_current) * 3, 50)
        currents = np.linspace(0, i_max, 20).tolist()
        rates: list[float] = []

        for i, I in enumerate(currents):
            pct = int((step / total_steps) * 100)
            q.put(
                {"type": "progress", "step": "fi_curve", "pct": pct, "msg": f"f-I curve {i + 1}/20"}
            )
            try:
                r = simulate_fn(**{**base_config, "current": I})
                rates.append(r["stats"]["rate_hz"])
            except Exception:
                rates.append(0.0)
            step += 1

        threshold_current = None
        for i, rate in enumerate(rates):
            if rate > 0:
                threshold_current = round(currents[i], 2)
                break

        max_rate = round(max(rates), 1) if rates else 0.0

        state_ranges = {}
        for var, values in trace["states"].items():
            arr = np.array(values)
            state_ranges[var] = {
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "mean": round(float(np.mean(arr)), 2),
            }

        params = base_config.get("params") or {}
        sensitivities: list[dict[str, Any]] = []
        param_list = list(params.items())[:15]
        for pi, (pname, pval) in enumerate(param_list):
            if pval == 0:
                step += 2
                continue
            pct = int((step / total_steps) * 100)
            q.put(
                {
                    "type": "progress",
                    "step": "sensitivity",
                    "pct": pct,
                    "msg": f"Sensitivity {pi + 1}/{len(param_list)}: {pname}",
                }
            )
            delta = abs(pval) * 0.1
            try:
                r_lo = simulate_fn(**{**base_config, "params": {**params, pname: pval - delta}})
                r_hi = simulate_fn(**{**base_config, "params": {**params, pname: pval + delta}})
                rate_change = abs(r_hi["stats"]["rate_hz"] - r_lo["stats"]["rate_hz"])
                sensitivities.append({"param": pname, "rate_change": round(rate_change, 2)})
            except (ValueError, ZeroDivisionError, KeyError, RuntimeError):
                pass
            step += 2

        sensitivities.sort(key=lambda s: s["rate_change"], reverse=True)

        result = {
            "pattern": pattern,
            "fi_curve": {"currents": currents, "rates": rates},
            "threshold_current": threshold_current,
            "max_rate": max_rate,
            "state_ranges": state_ranges,
            "top_sensitivities": sensitivities[:5],
            "spike_count": trace["spike_count"],
            "stats": trace["stats"],
        }
        q.put({"type": "complete", "pct": 100, "result": result})
    except Exception as e:
        q.put({"type": "error", "msg": str(e)})


def _heatmap_with_progress(
    simulate_fn: Callable[..., dict[str, Any]],
    base_config: dict[str, Any],
    param_x: str,
    x_vals: list[float],
    param_y: str,
    y_vals: list[float],
    q: queue.Queue[dict[str, Any]],
) -> None:
    """Run 2D heatmap sweep with progress updates."""
    try:
        total = len(x_vals) * len(y_vals)
        rates = []
        done = 0
        params = base_config.get("params") or {}

        for xi, xv in enumerate(x_vals):
            row = []
            for yi, yv in enumerate(y_vals):
                pct = int((done / total) * 100)
                q.put(
                    {
                        "type": "progress",
                        "step": "heatmap",
                        "pct": pct,
                        "msg": f"Sweep {done + 1}/{total}",
                    }
                )
                try:
                    sweep_params = {**params, param_x: float(xv), param_y: float(yv)}
                    r = simulate_fn(**{**base_config, "params": sweep_params})
                    row.append(r["stats"]["rate_hz"])
                except Exception:
                    row.append(0.0)
                done += 1
            rates.append(row)

        all_rates = [r for row in rates for r in row]
        result = {
            "param_x": param_x,
            "x_values": [float(v) for v in x_vals],
            "param_y": param_y,
            "y_values": [float(v) for v in y_vals],
            "rates": rates,
            "rate_min": round(min(all_rates), 2) if all_rates else 0,
            "rate_max": round(max(all_rates), 2) if all_rates else 0,
        }
        q.put({"type": "complete", "pct": 100, "result": result})
    except Exception as e:
        q.put({"type": "error", "msg": str(e)})


def _scan_with_progress(q: queue.Queue[dict[str, Any]]) -> None:
    """Scan all models with progress updates."""
    try:
        from sc_neurocore.studio.models import list_models, simulate_model
        from sc_neurocore.studio.codegen import classify_firing_pattern

        models = list_models()
        total = len(models)
        results = []

        for i, m in enumerate(models):
            pct = int((i / total) * 100)
            q.put(
                {
                    "type": "progress",
                    "step": "scan",
                    "pct": pct,
                    "msg": f"Scanning {m['name']} ({i + 1}/{total})",
                }
            )
            try:
                r = simulate_model(name=m["name"], current=10.0, duration=100.0)
                pattern = classify_firing_pattern(r["spikes"], r["n_steps"], r["dt"])
                results.append(
                    {
                        "name": m["name"],
                        "category": m["category"],
                        "pattern": pattern["pattern"],
                        "description": pattern["description"],
                        "rate_hz": r["stats"]["rate_hz"],
                        "spike_count": r["spike_count"],
                    }
                )
            except Exception:
                results.append(
                    {
                        "name": m["name"],
                        "category": m["category"],
                        "pattern": "error",
                        "description": "Simulation failed",
                        "rate_hz": 0,
                        "spike_count": 0,
                    }
                )

        q.put({"type": "complete", "pct": 100, "result": results})
    except Exception as e:
        q.put({"type": "error", "msg": str(e)})


async def ws_progress_handler(websocket: Any) -> None:
    """WebSocket handler for long-running operations with progress."""
    try:
        raw = await websocket.receive_text()
        request = json.loads(raw)
    except Exception:
        await websocket.send_json({"type": "error", "msg": "Invalid request"})
        await websocket.close()
        return

    op = request.get("op")
    q: queue.Queue[Any] = queue.Queue(maxsize=200)

    if op == "characterize":
        from sc_neurocore.studio.models import simulate_model

        config = request.get("config", {})

        def sim_fn(**kw: Any) -> dict[str, Any]:
            return simulate_model(
                name=config.get("name", "LIFNeuron"),
                param_overrides=kw.get("params", config.get("params")),
                dt=kw.get("dt", config.get("dt")),
                duration=kw.get("duration", config.get("duration", 200)),
                current=kw.get("current", config.get("current", 10)),
                protocol=kw.get("protocol", "constant"),
            )

        base_cfg = {
            "params": config.get("params"),
            "dt": config.get("dt"),
            "duration": config.get("duration", 200),
            "current": config.get("current", 10),
        }
        thread = threading.Thread(
            target=_characterize_with_progress, args=(sim_fn, base_cfg, q), daemon=True
        )

    elif op == "heatmap":
        from sc_neurocore.studio.models import simulate_model

        config = request.get("config", {})

        def sim_fn(**kw: Any) -> dict[str, Any]:
            return simulate_model(
                name=config.get("name", "LIFNeuron"),
                param_overrides=kw.get("params"),
                duration=kw.get("duration", config.get("duration", 100)),
                current=kw.get("current", config.get("current", 10)),
            )

        base_cfg = {
            "params": config.get("params", {}),
            "duration": config.get("duration", 100),
            "current": config.get("current", 10),
        }
        x_vals = np.linspace(
            config.get("x_min", 0), config.get("x_max", 1), config.get("x_steps", 10)
        ).tolist()
        y_vals = np.linspace(
            config.get("y_min", 0), config.get("y_max", 1), config.get("y_steps", 10)
        ).tolist()
        thread = threading.Thread(
            target=_heatmap_with_progress,
            args=(
                sim_fn,
                base_cfg,
                config.get("param_x", ""),
                x_vals,
                config.get("param_y", ""),
                y_vals,
                q,
            ),
            daemon=True,
        )

    elif op == "scan":
        thread = threading.Thread(target=_scan_with_progress, args=(q,), daemon=True)

    else:
        await websocket.send_json({"type": "error", "msg": f"Unknown op: {op}"})
        await websocket.close()
        return

    thread.start()

    while True:
        try:
            msg = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 1.0)
            await websocket.send_json(msg)
            if msg["type"] in ("complete", "error"):
                break
        except queue.Empty:
            if not thread.is_alive():
                break
            await websocket.send_json({"type": "heartbeat"})

    await websocket.close()
