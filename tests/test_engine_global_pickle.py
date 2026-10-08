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
from pathlib import Path
import pickle
import sys
from types import ModuleType
from typing import cast
import unittest

try:
    from tests.engine_requirement import require_engine
except ModuleNotFoundError as _outside_checkout:
    if _outside_checkout.name != "tests":
        raise
    _NATIVE = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
else:
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
