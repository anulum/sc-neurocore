# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for IP-XACT (IEEE 1685) component XML generation

"""Module-specific tests for :func:`sc_neurocore.hdl_gen.ip_xact.generate_ip_xact`.

Exercises every branch of the IP-XACT emitter — the AXI-Lite interface, the
optional parameter block, and the port vector geometry — and asserts the output
against the parsed element tree rather than raw substrings alone, so a structural
regression (a wrongly nested port, a dropped namespace) is caught rather than a
mere text drift.
"""

from __future__ import annotations

from defusedxml.ElementTree import fromstring  # type: ignore[import-untyped]  # no type stubs
import pytest

from sc_neurocore.hdl_gen.ip_xact import generate_ip_xact

#: The IP-XACT / SPIRIT 1685-2009 namespace the emitter binds to the ``spirit`` prefix.
_SPIRIT_NS = "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"


def _q(tag: str) -> str:
    """Return ``tag`` qualified with the SPIRIT namespace for ElementTree lookups."""
    return f"{{{_SPIRIT_NS}}}{tag}"


def _identity(xml: str) -> dict[str, str | None]:
    """Return the vendor/library/name/version identity block from an IP-XACT string."""
    root = fromstring(xml)
    return {
        field: (root.find(_q(field)).text if root.find(_q(field)) is not None else None)
        for field in ("vendor", "library", "name", "version")
    }


def _ports(xml: str) -> dict[str, tuple[str | None, str | None, str | None]]:
    """Map each port name to ``(direction, vector-left, vector-right)``.

    ``vector-left`` / ``vector-right`` are ``None`` for a scalar (1-bit) port, which
    the emitter renders without a ``spirit:vector`` element.
    """
    root = fromstring(xml)
    model = root.find(_q("model"))
    assert model is not None
    ports_el = model.find(_q("ports"))
    assert ports_el is not None
    result: dict[str, tuple[str | None, str | None, str | None]] = {}
    for port in ports_el.findall(_q("port")):
        name = port.find(_q("name"))
        wire = port.find(_q("wire"))
        assert name is not None and name.text is not None
        assert wire is not None
        direction = wire.find(_q("direction"))
        vector = wire.find(_q("vector"))
        left = right = None
        if vector is not None:
            left_el = vector.find(_q("left"))
            right_el = vector.find(_q("right"))
            left = left_el.text if left_el is not None else None
            right = right_el.text if right_el is not None else None
        result[name.text] = (
            direction.text if direction is not None else None,
            left,
            right,
        )
    return result


def test_default_component_is_wellformed_and_carries_default_identity() -> None:
    """A bare call yields parseable XML with the documented default identity."""
    xml = generate_ip_xact("sc_lif")

    # Well-formed: parses without raising, and is the pretty-printed SPIRIT root.
    assert xml.startswith('<?xml version="1.0" ?>')
    root = fromstring(xml)
    assert root.tag == _q("component")
    # ElementTree consumes ``xmlns:*`` as namespace declarations, so assert the
    # Xilinx binding on the serialised text (the SPIRIT binding is proven by the
    # namespaced root tag above).
    assert 'xmlns:xilinx="http://www.xilinx.com"' in xml

    assert _identity(xml) == {
        "vendor": "anulum.li",
        "library": "sc_neurocore",
        "name": "sc_lif",
        "version": "1.0",
    }


def test_default_ports_cover_scalar_and_vector_geometry() -> None:
    """The fixed port set renders scalars bare and the data port as a Q-width vector."""
    ports = _ports(generate_ip_xact("sc_lif", data_width=16))

    assert ports["clk"] == ("in", None, None)
    assert ports["rst"] == ("in", None, None)
    assert ports["en"] == ("in", None, None)
    assert ports["spike_out"] == ("out", None, None)
    # data_width 16 → a [15:0] vector on the current-input port.
    assert ports["I_t"] == ("in", "15", "0")


def test_data_width_one_emits_a_scalar_current_port() -> None:
    """A unit data width collapses ``I_t`` to a scalar — the ``width == 1`` branch."""
    ports = _ports(generate_ip_xact("sc_lif", data_width=1))
    assert ports["I_t"] == ("in", None, None)


@pytest.mark.parametrize("data_width", [4, 8, 24, 32])
def test_vector_left_tracks_data_width(data_width: int) -> None:
    """The data-port vector's left index is ``data_width - 1`` for any multi-bit width."""
    ports = _ports(generate_ip_xact("sc_lif", data_width=data_width))
    assert ports["I_t"] == ("in", str(data_width - 1), "0")


def test_axi_lite_bus_adds_a_slave_memory_mapped_interface() -> None:
    """``bus="axi_lite"`` appends an S_AXI aximm slave with a module-scoped map ref."""
    xml = generate_ip_xact("sc_lif", bus="axi_lite")
    root = fromstring(xml)
    bus_ifs = root.find(_q("busInterfaces"))
    assert bus_ifs is not None
    names = {
        el.find(_q("name")).text
        for el in bus_ifs.findall(_q("busInterface"))
        if el.find(_q("name")) is not None
    }
    assert {"clk", "rst", "S_AXI"} == names

    axi = next(
        el
        for el in bus_ifs.findall(_q("busInterface"))
        if el.find(_q("name")) is not None and el.find(_q("name")).text == "S_AXI"
    )
    bus_type = axi.find(_q("busType"))
    assert bus_type is not None
    assert bus_type.get(_q("name")) == "aximm"
    assert axi.find(_q("slave")) is not None
    mem = axi.find(_q("memoryMapRef"))
    assert mem is not None
    assert mem.get(_q("memoryMapRef")) == "sc_lif_mmap"


def test_non_axi_buses_omit_the_axi_interface() -> None:
    """The default and Wishbone buses leave only the clock and reset interfaces."""
    for xml in (generate_ip_xact("sc_lif"), generate_ip_xact("sc_lif", bus="wishbone")):
        root = fromstring(xml)
        bus_ifs = root.find(_q("busInterfaces"))
        assert bus_ifs is not None
        names = {
            el.find(_q("name")).text
            for el in bus_ifs.findall(_q("busInterface"))
            if el.find(_q("name")) is not None
        }
        assert names == {"clk", "rst"}


def test_parameters_block_renders_each_named_parameter() -> None:
    """A non-empty ``params`` map emits one user-resolvable long parameter per key."""
    xml = generate_ip_xact("sc_lif", params={"P_V_REST": 16, "P_V_THRESH": 16})
    root = fromstring(xml)
    params_el = root.find(_q("parameters"))
    assert params_el is not None

    rendered: dict[str, tuple[str | None, str | None, str | None]] = {}
    for param in params_el.findall(_q("parameter")):
        name = param.find(_q("name"))
        value = param.find(_q("value"))
        assert name is not None and name.text is not None
        assert value is not None
        rendered[name.text] = (
            value.text,
            value.get(_q("format")),
            value.get(_q("resolve")),
        )

    assert rendered == {
        "P_V_REST": ("0", "long", "user"),
        "P_V_THRESH": ("0", "long", "user"),
    }


def test_empty_params_map_omits_the_parameters_block() -> None:
    """An empty ``params`` dict is falsy, so no ``spirit:parameters`` element appears."""
    root = fromstring(generate_ip_xact("sc_lif", params={}))
    assert root.find(_q("parameters")) is None


def test_default_component_has_no_parameters_block() -> None:
    """Omitting ``params`` (``None``) likewise yields no parameters element."""
    root = fromstring(generate_ip_xact("sc_lif"))
    assert root.find(_q("parameters")) is None


def test_custom_identity_is_reflected_verbatim() -> None:
    """Vendor, library, version, and module name flow through unchanged."""
    xml = generate_ip_xact(
        "sc_izhikevich",
        vendor="example.org",
        library="custom_lib",
        version="2.3",
    )
    assert _identity(xml) == {
        "vendor": "example.org",
        "library": "custom_lib",
        "name": "sc_izhikevich",
        "version": "2.3",
    }


def test_synthesis_view_and_fileset_reference_the_module_source() -> None:
    """The synthesis view is Verilog and the fileset points at ``{module}.v``."""
    xml = generate_ip_xact("sc_lif")
    root = fromstring(xml)

    model = root.find(_q("model"))
    assert model is not None
    view = model.find(_q("views")).find(_q("view"))
    assert view is not None
    assert view.find(_q("language")).text == "verilog"

    file_sets = root.find(_q("fileSets"))
    assert file_sets is not None
    file_el = file_sets.find(_q("fileSet")).find(_q("file"))
    assert file_el is not None
    assert file_el.find(_q("name")).text == "sc_lif.v"
    assert file_el.find(_q("fileType")).text == "verilogSource"


def test_generation_is_deterministic() -> None:
    """Identical arguments produce byte-identical XML (no dict-ordering drift)."""
    first = generate_ip_xact("sc_lif", data_width=12, params={"A": 1, "B": 2}, bus="axi_lite")
    second = generate_ip_xact("sc_lif", data_width=12, params={"A": 1, "B": 2}, bus="axi_lite")
    assert first == second


def test_generate_ip_xact_is_exported_from_the_package() -> None:
    """The emitter is re-exported at the ``hdl_gen`` package level, like its siblings."""
    import sc_neurocore.hdl_gen as hdl_gen

    assert "generate_ip_xact" in hdl_gen.__all__
    assert hdl_gen.generate_ip_xact is generate_ip_xact
