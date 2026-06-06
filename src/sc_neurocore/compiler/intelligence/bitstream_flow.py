# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream automation

"""Bitstream automation utilities for open-source FPGA flows.

Generates Makefiles for Yosys + nextpnr synthesis, placement, and routing.
"""

from __future__ import annotations

from typing import Literal


def generate_oss_makefile(
    module_name: str,
    *,
    target: Literal["ice40", "ecp5"] = "ice40",
    device: str = "hx8k",
    package: str = "ct256",
    freq_mhz: float = 12.0,
    verilog_files: list[str] | None = None,
    pcf_file: str | None = None,
) -> str:
    """Generate a Makefile for open-source FPGA synthesis (Yosys + nextpnr).

    Parameters
    ----------
    module_name : str
        Top-level module name.
    target : str
        ``"ice40"`` or ``"ecp5"``.
    device : str
        Device string (e.g. ``"hx8k"``, ``"um5g-85k"``).
    package : str
        Package (e.g. ``"ct256"``, ``"CABGA381"``).
    freq_mhz : float
        Target frequency.
    verilog_files : list, optional
        Verilog source files.
    pcf_file : str, optional
        Pin constraint file.

    Returns
    -------
    str
        Complete Makefile content.
    """
    if verilog_files is None:
        verilog_files = [f"{module_name}.v"]

    srcs = " ".join(verilog_files)

    if target == "ice40":
        return f"""# Auto-generated Makefile for {module_name} (iCE40)
# Tools: Yosys + nextpnr-ice40 + icepack

TOP = {module_name}
DEVICE = {device}
PACKAGE = {package}
FREQ = {freq_mhz}
SRCS = {srcs}
PCF = {pcf_file or module_name + ".pcf"}

all: $(TOP).bin

$(TOP).json: $(SRCS)
\tyosys -p "synth_ice40 -top $(TOP) -json $@" $(SRCS)

$(TOP).asc: $(TOP).json $(PCF)
\tnextpnr-ice40 --$(DEVICE) --package $(PACKAGE) --pcf $(PCF) --json $< --asc $@ --freq $(FREQ)

$(TOP).bin: $(TOP).asc
\ticepack $< $@

prog: $(TOP).bin
\ticeprog $<

clean:
\trm -f $(TOP).json $(TOP).asc $(TOP).bin

.PHONY: all prog clean
"""
    elif target == "ecp5":
        return f"""# Auto-generated Makefile for {module_name} (ECP5)
# Tools: Yosys + nextpnr-ecp5 + ecppack

TOP = {module_name}
DEVICE = {device}
PACKAGE = {package}
FREQ = {freq_mhz}
SRCS = {srcs}
LPF = {pcf_file or module_name + ".lpf"}

all: $(TOP).bit

$(TOP).json: $(SRCS)
\tyosys -p "synth_ecp5 -top $(TOP) -json $@" $(SRCS)

$(TOP).config: $(TOP).json $(LPF)
\tnextpnr-ecp5 --$(DEVICE) --package $(PACKAGE) --lpf $(LPF) --json $< --textcfg $@ --freq $(FREQ)

$(TOP).bit: $(TOP).config
\tecppack $< $@

prog: $(TOP).bit
\topendla prog $(TOP).bit

clean:
\trm -f $(TOP).json $(TOP).config $(TOP).bit

.PHONY: all prog clean
"""
    raise ValueError(f"Unsupported OSS target: {target!r}")
