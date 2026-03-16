# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Component registry for user-defined neurons, synapses, and layers.

Usage::

    from sc_neurocore.utils.registry import registry

    @registry.register("neuron", "MyLIF")
    class MyLIF:
        ...

    cls = registry.get("neuron", "MyLIF")
    print(registry.list("neuron"))
"""

from __future__ import annotations


class ComponentRegistry:
    """Thread-safe component registry with namespace support."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, type]] = {}

    def register(self, namespace: str, name: str | None = None):
        """Decorator to register a class under *namespace*/*name*.

        If *name* is ``None``, ``cls.__name__`` is used.
        """

        def decorator(cls: type) -> type:
            key = name or cls.__name__
            ns = self._store.setdefault(namespace, {})
            if key in ns:
                raise KeyError(f"{namespace}/{key} already registered")
            ns[key] = cls
            return cls

        return decorator

    def get(self, namespace: str, name: str) -> type:
        """Retrieve a registered class. Raises ``KeyError`` if missing."""
        try:
            return self._store[namespace][name]
        except KeyError:
            raise KeyError(f"{namespace}/{name} not registered") from None

    def list(self, namespace: str) -> list[str]:
        """Return sorted names in *namespace*."""
        return sorted(self._store.get(namespace, {}))

    def namespaces(self) -> list[str]:
        """Return all registered namespaces."""
        return sorted(self._store)

    def clear(self, namespace: str | None = None) -> None:
        """Remove all entries (or just one namespace)."""
        if namespace is None:
            self._store.clear()
        else:
            self._store.pop(namespace, None)


registry = ComponentRegistry()
