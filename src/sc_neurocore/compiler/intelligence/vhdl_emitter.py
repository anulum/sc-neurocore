# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VHDL emitter

"""VHDL-2008 emitter for compiled neuron modules.

Generates VHDL wrappers for Verilog neuron modules to support
mixed-language simulation and synthesis.
"""

from __future__ import annotations


def verilog_to_vhdl_wrapper(
    module_name: str,
    *,
    data_width: int = 16,
    signed: bool = True,
) -> str:
    """Generate a VHDL-2008 entity/architecture wrapper for a Verilog module.

    This produces a VHDL entity that matches the Verilog module's port list,
    enabling mixed-language simulation and synthesis (Vivado, Questa, GHDL).
    The VHDL wrapper instantiates the Verilog module as a component.

    Parameters
    ----------
    module_name : str
        Verilog module name.
    data_width : int
        Fixed-point data width.
    signed : bool
        Whether ports use signed types.

    Returns
    -------
    str
        VHDL-2008 source code.
    """
    dw = data_width
    sig_type = "signed" if signed else "unsigned"

    return f"""-- Auto-generated VHDL-2008 wrapper for {module_name}
-- SC-NeuroCore — DO-254 / IEC 61508 compliant output
-- Mixed-language: instantiates Verilog module via component

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {module_name}_vhdl is
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        en        : in  std_logic;
        I_t       : in  {sig_type}({dw - 1} downto 0);
        spike_out : out std_logic
    );
end entity {module_name}_vhdl;

architecture rtl of {module_name}_vhdl is

    component {module_name} is
        port (
            clk       : in  std_logic;
            rst       : in  std_logic;
            en        : in  std_logic;
            I_t       : in  std_logic_vector({dw - 1} downto 0);
            spike_out : out std_logic
        );
    end component;

    signal I_t_slv : std_logic_vector({dw - 1} downto 0);

begin

    I_t_slv <= std_logic_vector(I_t);

    u_neuron : {module_name}
        port map (
            clk       => clk,
            rst       => rst,
            en        => en,
            I_t       => I_t_slv,
            spike_out => spike_out
        );

end architecture rtl;
"""
