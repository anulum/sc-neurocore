#!/usr/bin/env python3
"""Example 05: Verilog HDL Generation.

Demonstrates generating a Verilog top-level module from a network
description using the HDL generation utilities.
"""

from sc_neurocore.hdl_gen import VerilogGenerator

def main():
    print("=== SC-NeuroCore: Verilog HDL Generation ===\n")

    gen = VerilogGenerator(module_name="sc_example_top")

    # Add layers to the network description
    gen.add_layer("Dense", "hidden_0", {"n_neurons": 16})
    gen.add_layer("Dense", "hidden_1", {"n_neurons": 8})
    gen.add_layer("Dense", "output",   {"n_neurons": 4})

    # Generate Verilog code
    verilog = gen.generate()

    print("Generated Verilog:")
    print("-" * 60)
    print(verilog)
    print("-" * 60)
    print(f"\nTotal lines: {len(verilog.splitlines())}")
    print("Done.")


if __name__ == "__main__":
    main()
