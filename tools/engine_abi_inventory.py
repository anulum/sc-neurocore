# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed engine interface inventory

"""Capture engine exports, aliases, signatures and global reference identities.

Run capture with the interpreter that owns the wheel installation. Comparison
excludes measured paths and binary hashes but checks every interface field.
Capture resolves live global references without serialisation or deserialisation.
Comparison reads JSON. Pickle compatibility, instance state, numerical behaviour
and array admission require separate tests against the installed bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import platform
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.engine_abi_contracts import JsonValue, differences, read_inventory

MODULES = ("sc_neurocore_engine", "sc_neurocore_engine.sc_neurocore_engine")
INTROSPECTION_MEMBERS = frozenset(("__doc__", "__dict__", "__weakref__"))


def symbol_details(value: object) -> dict[str, JsonValue]:
    """Measure a live object's identity, signature and scalar constant value.

    Parameters
    ----------
    value : object
        Exported object or statically resolved class member.

    Returns
    -------
    dict of str to JsonValue
        Stable identity and signature fields without address-based repr values.
    """
    result: dict[str, JsonValue] = {"kind": type(value).__name__}
    for field in ("__module__", "__qualname__", "__name__", "__text_signature__"):
        attribute: object = getattr(value, field, None)
        result[field] = attribute if isinstance(attribute, str) else None
    if callable(value):
        try:
            result["signature"] = str(inspect.signature(value))
            result["signature_error"] = None
        except (TypeError, ValueError) as exc:
            result["signature"] = None
            result["signature_error"] = f"{type(exc).__name__}: {exc}"
    if value is None or isinstance(value, (bool, int, str)):
        result["constant"] = value
    elif isinstance(value, float):
        result["constant"] = repr(value)
    return result


def global_reference_identity(value: object) -> dict[str, JsonValue]:
    """Resolve the published module and qualified name of a live global.

    Parameters
    ----------
    value : object
        Live imported class or function.

    Returns
    -------
    dict of str to JsonValue
        Resolution identity or an observed refusal diagnostic. This does not
        claim a successful pickle roundtrip.
    """
    try:
        module: object = getattr(value, "__module__", None)
        qualified: object = getattr(value, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualified, str):
            raise ValueError("global reference requires named module and qualified name")
        restored: object = importlib.import_module(module)
        for part in qualified.split("."):
            restored = getattr(restored, part)
    except (ValueError, AttributeError, ImportError) as exc:
        return {"same_object": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "same_object": restored is value,
        "error": None,
    }


def capture_inventory(require_installed: bool = False) -> dict[str, JsonValue]:
    """Capture both actual engine namespaces and every advertised facade export.

    Parameters
    ----------
    require_installed : bool, default=False
        Require both modules to originate inside this interpreter's site-packages.

    Returns
    -------
    dict of str to JsonValue
        Versioned interface inventory and measured module/binary provenance.

    Raises
    ------
    ValueError
        The package is editable, advertises an absent name or has malformed exports.
    ImportError
        Either required engine namespace cannot load.
    """
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    installation_roots = sorted(
        {Path(sysconfig.get_path(scheme)).resolve() for scheme in ("purelib", "platlib")}
    )
    if require_installed:
        spec = importlib.util.find_spec(MODULES[0])
        spec_origin = spec.origin if spec is not None else None
        if spec_origin is None or not any(
            Path(spec_origin).resolve().is_relative_to(root) for root in installation_roots
        ):
            raise ValueError(
                f"{MODULES[0]} does not originate in installed site-packages: {spec_origin}"
            )
    loaded: dict[str, ModuleType] = {name: importlib.import_module(name) for name in MODULES}
    facade = loaded[MODULES[0]]
    declared: object = getattr(facade, "__all__", None)
    if (
        not isinstance(declared, list)
        or not declared
        or not all(isinstance(name, str) for name in declared)
    ):
        raise ValueError("engine facade __all__ must be a list of names")
    exports = [str(name) for name in declared]
    missing = sorted(name for name in exports if not hasattr(facade, name))
    if missing:
        raise ValueError(f"engine facade advertises absent exports: {missing}")
    provenance: dict[str, JsonValue] = {}
    modules: dict[str, JsonValue] = {}
    identities: dict[int, list[str]] = {}
    records: dict[str, dict[str, JsonValue]] = {}
    for name, module in loaded.items():
        origin_value: object = getattr(module, "__file__", None)
        if not isinstance(origin_value, str):
            raise ValueError(f"{name} has no file provenance")
        origin = Path(origin_value).resolve()
        installed = any(origin.is_relative_to(root) for root in installation_roots)
        if require_installed and not installed:
            raise ValueError(f"{name} does not originate in installed site-packages: {origin}")
        provenance[name] = {
            "path": str(origin),
            "sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
            "installed": installed,
        }
        symbols: dict[str, JsonValue] = {}
        if not hasattr(module, "__version__"):
            raise ValueError(f"{name} has no __version__")
        names = sorted({n for n in dir(module) if not n.startswith("_")} | {"__version__"})
        for symbol in names:
            value: object = getattr(module, symbol)
            details = symbol_details(value)
            if inspect.isclass(value):
                members: dict[str, JsonValue] = {}
                for member in sorted(dir(value)):
                    public = not member.startswith("_")
                    protocol = member.startswith("__") and member.endswith("__")
                    if (public or protocol) and member not in INTROSPECTION_MEMBERS:
                        members[member] = symbol_details(inspect.getattr_static(value, member))
                details["members"] = members
            if inspect.isclass(value) or inspect.isroutine(value):
                details["global_reference"] = global_reference_identity(value)
                qualified = f"{name}.{symbol}"
                identities.setdefault(id(value), []).append(qualified)
                records[qualified] = details
            symbols[symbol] = details
        modules[name] = symbols
    for aliases in identities.values():
        for qualified in aliases:
            records[qualified]["aliases"] = list(sorted(aliases))
    facade_exports: list[JsonValue] = list(exports)
    return {
        "schema": 1,
        "inventory": {"modules": modules, "facade_exports": facade_exports},
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "site_packages": str(site_packages),
            "installation_roots": [str(root) for root in installation_roots],
            "require_installed": require_installed,
            "modules": provenance,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Capture an interface or fail on any difference from a saved interface.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; defaults to the process arguments.

    Returns
    -------
    int
        Zero for capture/equality, one for interface drift, two for invalid input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("capture")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--require-installed", action="store_true")
    compare = modes.add_parser("compare")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "capture":
            result = capture_inventory(args.require_installed)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"Engine ABI inventory written to {args.output}")
            return 0
        changes = differences(read_inventory(args.before), read_inventory(args.after))
    except (OSError, ValueError, ImportError) as exc:
        print(f"Engine ABI inventory failed: {exc}", file=sys.stderr)
        return 2
    if changes:
        print("\n".join(changes), file=sys.stderr)
        return 1
    print("Engine ABI inventories match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
