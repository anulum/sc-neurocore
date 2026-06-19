# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# SC-NeuroCore — IP-XACT packaging for Vivado IP Integrator

"""Generate IP-XACT (IEEE 1685) component XML for FPGA IP integration.

Produces a ``component.xml`` that allows the neuron module to appear as a
drag-and-drop block in Xilinx Vivado IP Integrator, with auto-connected
AXI ports, clock, and reset.

Usage::

    from sc_neurocore.hdl_gen.ip_xact import generate_ip_xact

    xml = generate_ip_xact(
        module_name="sc_lif",
        vendor="anulum.li",
        version="1.0",
        data_width=16,
        params={"P_V_REST": 16, "P_V_THRESH": 16},
        bus="axi_lite",
    )
"""

from __future__ import annotations

import importlib
from typing import Literal
from xml.etree.ElementTree import Element, SubElement, tostring  # nosec B405

parseString = importlib.import_module("defusedxml.minidom").parseString


def generate_ip_xact(
    module_name: str,
    *,
    vendor: str = "anulum.li",
    library: str = "sc_neurocore",
    version: str = "1.0",
    data_width: int = 16,
    params: dict[str, int] | None = None,
    bus: Literal["axi_lite", "wishbone", "none"] = "none",
) -> str:
    """Generate IP-XACT component XML.

    Parameters
    ----------
    module_name : str
        Top-level Verilog module name.
    vendor : str
        IP vendor identifier.
    library : str
        IP library name.
    version : str
        IP version string.
    data_width : int
        Neuron data width.
    params : dict, optional
        Verilog parameters.
    bus : str
        Bus interface type.

    Returns
    -------
    str
        IP-XACT XML string.
    """
    ns = "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"
    comp = Element(
        "spirit:component",
        {
            "xmlns:spirit": ns,
            "xmlns:xilinx": "http://www.xilinx.com",
        },
    )

    # Identity
    SubElement(comp, "spirit:vendor").text = vendor
    SubElement(comp, "spirit:library").text = library
    SubElement(comp, "spirit:name").text = module_name
    SubElement(comp, "spirit:version").text = version

    # Bus interfaces
    bus_ifs = SubElement(comp, "spirit:busInterfaces")

    # Clock
    clk_if = SubElement(bus_ifs, "spirit:busInterface")
    SubElement(clk_if, "spirit:name").text = "clk"
    bt = SubElement(clk_if, "spirit:busType")
    bt.set("spirit:vendor", "xilinx.com")
    bt.set("spirit:library", "signal")
    bt.set("spirit:name", "clock")
    bt.set("spirit:version", "1.0")
    SubElement(clk_if, "spirit:slave")
    pm = SubElement(clk_if, "spirit:portMaps")
    p = SubElement(pm, "spirit:portMap")
    lp = SubElement(p, "spirit:logicalPort")
    SubElement(lp, "spirit:name").text = "CLK"
    pp = SubElement(p, "spirit:physicalPort")
    SubElement(pp, "spirit:name").text = "clk"

    # Reset
    rst_if = SubElement(bus_ifs, "spirit:busInterface")
    SubElement(rst_if, "spirit:name").text = "rst"
    bt2 = SubElement(rst_if, "spirit:busType")
    bt2.set("spirit:vendor", "xilinx.com")
    bt2.set("spirit:library", "signal")
    bt2.set("spirit:name", "reset")
    bt2.set("spirit:version", "1.0")
    SubElement(rst_if, "spirit:slave")

    # AXI-Lite interface
    if bus == "axi_lite":
        axi_if = SubElement(bus_ifs, "spirit:busInterface")
        SubElement(axi_if, "spirit:name").text = "S_AXI"
        axi_bt = SubElement(axi_if, "spirit:busType")
        axi_bt.set("spirit:vendor", "xilinx.com")
        axi_bt.set("spirit:library", "interface")
        axi_bt.set("spirit:name", "aximm")
        axi_bt.set("spirit:version", "1.0")
        SubElement(axi_if, "spirit:slave")
        mem = SubElement(axi_if, "spirit:memoryMapRef")
        mem.set("spirit:memoryMapRef", f"{module_name}_mmap")

    # Model
    model = SubElement(comp, "spirit:model")
    views = SubElement(model, "spirit:views")
    view = SubElement(views, "spirit:view")
    SubElement(view, "spirit:name").text = "xilinx_synthesis"
    SubElement(view, "spirit:envIdentifier").text = ":vivado.xilinx.com:synthesis"
    lang = SubElement(view, "spirit:language")
    lang.text = "verilog"
    fsets = SubElement(view, "spirit:fileSetRef")
    SubElement(fsets, "spirit:localName").text = "xilinx_synthesis_view_fileset"

    # Ports
    ports = SubElement(model, "spirit:ports")
    _add_port(ports, "clk", "in", 1)
    _add_port(ports, "rst", "in", 1)
    _add_port(ports, "en", "in", 1)
    _add_port(ports, "I_t", "in", data_width)
    _add_port(ports, "spike_out", "out", 1)

    # File sets
    file_sets = SubElement(comp, "spirit:fileSets")
    fs = SubElement(file_sets, "spirit:fileSet")
    SubElement(fs, "spirit:name").text = "xilinx_synthesis_view_fileset"
    f = SubElement(fs, "spirit:file")
    SubElement(f, "spirit:name").text = f"{module_name}.v"
    SubElement(f, "spirit:fileType").text = "verilogSource"

    # Parameters
    if params:
        parameters = SubElement(comp, "spirit:parameters")
        for pname, pwidth in params.items():
            param = SubElement(parameters, "spirit:parameter")
            SubElement(param, "spirit:name").text = pname
            val = SubElement(param, "spirit:value")
            val.text = "0"
            val.set("spirit:format", "long")
            val.set("spirit:resolve", "user")

    raw = tostring(comp, encoding="unicode")
    pretty_xml: str = parseString(raw).toprettyxml(indent="  ")
    return pretty_xml


def _add_port(
    parent: Element,
    name: str,
    direction: str,
    width: int,
) -> None:
    """Add a port element to the ports section.

    Parameters
    ----------
    parent : Element
        Parent ports element.
    name : str
        Port name.
    direction : str
        ``"in"`` or ``"out"``.
    width : int
        Bit width.
    """
    port = SubElement(parent, "spirit:port")
    SubElement(port, "spirit:name").text = name
    wire = SubElement(port, "spirit:wire")
    SubElement(wire, "spirit:direction").text = direction
    if width > 1:
        vec = SubElement(wire, "spirit:vector")
        left = SubElement(vec, "spirit:left")
        left.text = str(width - 1)
        right = SubElement(vec, "spirit:right")
        right.text = "0"
