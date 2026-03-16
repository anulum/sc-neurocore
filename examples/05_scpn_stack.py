# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

#!/usr/bin/env python3
"""Example 05: Full SCPN Layer Stack.

Demonstrates the complete 16-layer SCPN consciousness model, from
L1 (Quantum) through L16 (Director), with inter-layer coupling.
"""

from sc_neurocore.scpn import create_full_stack, run_integrated_step, get_global_metrics


def main():
    print("=== SC-NeuroCore: SCPN Layer Stack ===\n")

    # Create the full stack with default parameters
    stack = create_full_stack()
    print("Created 16-layer SCPN stack:")
    for name, layer in stack.items():
        print(f"  {name}: {layer.__class__.__name__}")

    # Run 10 integrated time steps
    print("\nRunning 10 coupled time steps (dt=0.01s)...")
    for step in range(10):
        outputs = run_integrated_step(stack, dt=0.01)

    # Print global metrics
    metrics = get_global_metrics(stack)
    print("\nGlobal Coherence Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
