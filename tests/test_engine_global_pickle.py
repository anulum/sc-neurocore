# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public engine global serialization contracts

"""Round-trip actual exported engine classes and functions through every pickle protocol."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
import pickle
import sys
from types import ModuleType
from typing import cast
import unittest

_CHECKOUT_GATE: Callable[[], ModuleType] | None
try:
    from tests.engine_requirement import require_engine as _checkout_require_engine
except ModuleNotFoundError as _outside_checkout:
    if _outside_checkout.name != "tests":
        raise
    _CHECKOUT_GATE = None
else:
    _CHECKOUT_GATE = _checkout_require_engine


def require_engine() -> ModuleType:
    """Require the real extension through the checkout or installed consumer gate.

    Returns
    -------
    ModuleType
        The loaded compiled engine used by the serialization contracts.

    Raises
    ------
    ImportError
        An installed extension or dependency fails to load. Outside the
        checkout, genuine absence is also a hard failure. Inside the checkout,
        the maintained gate preserves the optional-profile collection policy
        and the hosted-CI requirement. No engine errors are caught here.
    """
    if _CHECKOUT_GATE is None:
        return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return _CHECKOUT_GATE()


_NATIVE = require_engine()

_FACADE = importlib.import_module("sc_neurocore_engine")


class EngineGlobalPickleContracts(unittest.TestCase):
    """Preserve real public class and function identities during serialization."""

    def _assert_global_roundtrips(self, module: ModuleType, names: tuple[str, ...]) -> None:
        """Exercise each actual public callable through the complete pickle protocol set."""
        observed = 0
        for name in names:
            value: object = getattr(module, name)
            if not (inspect.isclass(value) or inspect.isroutine(value)):
                continue
            observed += 1
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(namespace=module.__name__, name=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(value, protocol)), value)
        self.assertGreater(observed, 0, "The namespace must expose real class/function globals")

    def test_facade_exports_preserve_global_pickle_identity(self) -> None:
        """Serialize every declared facade class/function through its actual native identity."""
        names = tuple(cast(list[str], _FACADE.__all__))
        self._assert_global_roundtrips(_FACADE, names)

    def test_native_namespace_preserves_global_pickle_identity(self) -> None:
        """Serialize all public extension classes/functions, including native-only exports."""
        names = tuple(name for name in dir(_NATIVE) if not name.startswith("_"))
        self._assert_global_roundtrips(_NATIVE, names)


if __name__ == "__main__":
    for module in (_FACADE, _NATIVE):
        if not Path(str(module.__file__)).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("Global-pickle consumer requires an installed engine")
    unittest.main()
