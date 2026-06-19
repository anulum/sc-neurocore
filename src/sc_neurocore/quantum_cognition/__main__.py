# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum Cognition CLI

"""CLI entry point for the quantum cognition learning system.

Usage::

    # One-shot: index a repo, run learning steps, save state
    python -m sc_neurocore.quantum_cognition learn /path/to/repo

    # Continuous daemon: shuffle GOTM, learn in cycles
    python -m sc_neurocore.quantum_cognition daemon --dashboard

    # Print saved learning state
    python -m sc_neurocore.quantum_cognition status

The daemon mode replaces the legacy ``run_gotm_brain.sh`` bash script,
using the ``agentic-shared/llm`` library for LLM communication instead
of raw ``curl`` calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .gotm_brain import HAS_LLM, GOTMBrain

logger = logging.getLogger("sc_neurocore.quantum_cognition")

# GOTM collection master path (NTFS, never SAS mirror)
_DEFAULT_GOTM_PATH = "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection"
_DEFAULT_STATE_FILE = "gotm_brain_state.json"
_DEFAULT_SNN_DIR = os.path.join(_DEFAULT_GOTM_PATH, "04_ARCANE_SAPIENCE", "snn_stimuli")


def _emit_snn_stimulus(snn_dir: str, chunk_summary: str, directive: str, step_index: int) -> None:
    """Write an SNN stimulus JSON file for downstream orchestration."""
    os.makedirs(snn_dir, exist_ok=True)
    ts = int(time.time())
    payload = {
        "text": f"QC step {step_index}: {directive} — {chunk_summary[:100]}",
        "source": "quantum_cognition",
        "project": "SC-NEUROCORE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(snn_dir, f"qc_{ts}.json")
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except OSError as exc:
        logger.warning("SNN stimulus write failed: %s", exc)


def _make_llm_endpoint(model: str | None) -> Any:
    """Create an agentic-shared Endpoint if model override requested."""
    if model is None:
        return None
    try:
        _sys_path = "/media/anulum/724AA8E84AA8AA75/agentic-shared"
        if _sys_path not in sys.path:
            sys.path.insert(0, _sys_path)
        from llm import Endpoint

        return Endpoint(model=model)
    except ImportError:
        logger.warning("agentic-shared not available, --model ignored")
        return None


def cmd_learn(args: argparse.Namespace) -> int:
    """One-shot learning from a repository."""
    endpoint = _make_llm_endpoint(args.model)
    brain = GOTMBrain(
        n_neurons=args.n_neurons,
        seed=args.seed,
        llm_endpoint=endpoint,
    )

    # Restore prior state if exists
    if args.state_file and os.path.exists(args.state_file):
        brain.load_state(args.state_file)
        logger.info("Resumed from %s (%d prior steps)", args.state_file, brain._total_steps)

    repo_path = args.repo_path or _DEFAULT_GOTM_PATH
    logger.info("Learning from: %s", repo_path)
    steps = brain.learn_from_repo(
        repo_path,
        max_chunks=args.max_chunks,
    )

    # Emit SNN stimuli for each step
    for step in steps:
        _emit_snn_stimulus(args.snn_dir, step.chunk_summary, step.directive, step.step_index)

    # Save state
    if args.state_file:
        brain.save_state(args.state_file)

    # Print summary
    state = brain.get_learning_state()
    print(f"\n{'=' * 60}")
    print(f"Learning complete: {len(steps)} chunks processed")
    print(f"  Total steps:   {state['total_steps']}")
    print(f"  Total spikes:  {state['total_spikes']}")
    print(f"  Avg ATP:       {state['avg_atp']:.4f}")
    print(f"  Avg entangle:  {state['avg_entanglement']:.6f}")
    print(f"  LLM available: {state['has_llm']}")
    print(f"  Backend:       {state['bridge_backend']}")
    print(f"{'=' * 60}")

    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Continuous learning daemon — shuffles GOTM, learns in cycles."""
    endpoint = _make_llm_endpoint(args.model)
    brain = GOTMBrain(
        n_neurons=args.n_neurons,
        seed=args.seed,
        llm_endpoint=endpoint,
    )

    # Restore prior state
    if args.state_file and os.path.exists(args.state_file):
        brain.load_state(args.state_file)
        logger.info("Daemon resumed from %s", args.state_file)

    # Graceful shutdown on SIGTERM/SIGINT
    shutdown = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown
        logger.info("Received signal %d, shutting down gracefully...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    repo_path = args.repo_path or _DEFAULT_GOTM_PATH
    cycle = 0

    # Optional dashboard
    dashboard = None
    if args.dashboard:
        try:
            from .dashboard import TerminalDashboard

            dashboard = TerminalDashboard()
        except ImportError:
            logger.warning("Dashboard module not available")

    print("\033[1;35m--- GOTM Brain Daemon Started ---\033[0m")
    print(f"  Neurons: {args.n_neurons}  |  LLM: {HAS_LLM}  |  Path: {repo_path}")
    print(f"  State file: {args.state_file}")
    print("  Press Ctrl+C to stop (state auto-saved)\n")

    try:
        while not shutdown:
            cycle += 1
            logger.info("--- Cycle %d ---", cycle)

            steps = brain.learn_from_repo(
                repo_path,
                max_chunks=args.max_chunks,
            )

            # SNN stimuli
            for step in steps:
                _emit_snn_stimulus(
                    args.snn_dir,
                    step.chunk_summary,
                    step.directive,
                    step.step_index,
                )

            # Dashboard update
            if dashboard is not None:
                dashboard.draw(brain)

            # Batch report
            state = brain.get_learning_state()
            print(f"\033[1;33m--- Cycle {cycle} Report ---\033[0m")
            print(f"  Chunks: {len(steps)}  Spikes: {sum(s.n_spikes for s in steps)}")
            print(f"  ATP: {state['avg_atp']:.4f}  Entanglement: {state['avg_entanglement']:.6f}")

            # Persist state after every cycle
            if args.state_file:
                brain.save_state(args.state_file)

            if shutdown:
                break

            logger.info("Sleeping %ds before next cycle...", args.sleep)
            for _ in range(args.sleep):
                if shutdown:
                    break
                time.sleep(1)

    finally:
        # Always save state on exit
        if args.state_file:
            brain.save_state(args.state_file)
            print(f"\n\033[1;32mState saved to {args.state_file}\033[0m")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print saved learning state."""
    if not args.state_file or not os.path.exists(args.state_file):
        print("No saved state found.")
        print(f"  Expected: {args.state_file}")
        return 1

    try:
        fsize = os.path.getsize(args.state_file)
        if fsize == 0:
            print("No saved state found (file is empty).")
            print(f"  Path: {args.state_file}")
            return 1
        with open(args.state_file) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Cannot read state file: {exc}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"GOTM Brain State: {args.state_file}")
    print(f"{'=' * 60}")
    print(f"  Neurons:         {state.get('n_neurons', '?')}")
    print(f"  Total steps:     {state.get('total_steps', 0)}")
    print(f"  Backend:         {state.get('bridge_backend', '?')}")
    print(f"  History length:  {len(state.get('history', []))}")

    pool = state.get("pool_state", {})
    emap = pool.get("entanglement_map", [])
    if emap:
        arr = np.array(emap)
        print(f"  Avg entangle:    {np.mean(arr):.6f}")
        print(f"  Max entangle:    {np.max(arr):.6f}")
        print(f"  Measurements:    {pool.get('measurement_count', 0)}")

    neurons = state.get("neuron_states", [])
    if neurons:
        atps = [n.get("atp_level", 0) for n in neurons]
        spikes = sum(n.get("total_spikes", 0) for n in neurons)
        failures = sum(n.get("metabolic_failures", 0) for n in neurons)
        print(f"  Total spikes:    {spikes}")
        print(f"  Metabolic fails: {failures}")
        print(f"  Avg ATP:         {np.mean(atps):.4f}")

    # Last 5 history entries
    history = state.get("history", [])
    if history:
        print("\n  Last 5 learning steps:")
        for h in history[-5:]:
            print(
                f"    [{h['step']:4d}] {h['directive']:10s} "
                f"spikes={h['n_spikes']:3d} ATP={h['avg_atp']:.3f} "
                f"ent={h['avg_entanglement']:.5f}"
            )

    print(f"{'=' * 60}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m sc_neurocore.quantum_cognition",
        description="GOTM Quantum Cognition Brain — learning system CLI",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- learn ---
    p_learn = sub.add_parser("learn", help="One-shot learning from a repository")
    p_learn.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help=f"Repository path (default: {_DEFAULT_GOTM_PATH})",
    )
    p_learn.add_argument("--max-chunks", type=int, default=None, help="Max chunks to process")
    p_learn.add_argument(
        "--n-neurons", type=int, default=32, help="Number of neurons (default: 32)"
    )
    p_learn.add_argument("--seed", type=int, default=42, help="Random seed")
    p_learn.add_argument("--model", default=None, help="LLM model alias (e.g. qwq:32b, gemma4:e4b)")
    p_learn.add_argument(
        "--state-file", default=_DEFAULT_STATE_FILE, help="Path to save/restore brain state"
    )
    p_learn.add_argument("--snn-dir", default=_DEFAULT_SNN_DIR, help="SNN stimuli output directory")

    # --- daemon ---
    p_daemon = sub.add_parser("daemon", help="Continuous learning daemon")
    p_daemon.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help=f"Repository path (default: {_DEFAULT_GOTM_PATH})",
    )
    p_daemon.add_argument(
        "--max-chunks", type=int, default=50, help="Max chunks per cycle (default: 50)"
    )
    p_daemon.add_argument(
        "--n-neurons", type=int, default=32, help="Number of neurons (default: 32)"
    )
    p_daemon.add_argument("--seed", type=int, default=42, help="Random seed")
    p_daemon.add_argument("--model", default=None, help="LLM model alias")
    p_daemon.add_argument(
        "--state-file", default=_DEFAULT_STATE_FILE, help="Path to save/restore brain state"
    )
    p_daemon.add_argument(
        "--snn-dir", default=_DEFAULT_SNN_DIR, help="SNN stimuli output directory"
    )
    p_daemon.add_argument(
        "--sleep", type=int, default=5, help="Seconds between cycles (default: 5)"
    )
    p_daemon.add_argument("--dashboard", action="store_true", help="Show ANSI terminal dashboard")

    # --- status ---
    p_status = sub.add_parser("status", help="Print saved learning state")
    p_status.add_argument(
        "--state-file", default=_DEFAULT_STATE_FILE, help="Path to brain state file"
    )

    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "learn":
        return cmd_learn(args)
    elif args.command == "daemon":
        return cmd_daemon(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
