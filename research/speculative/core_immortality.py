# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, Any

logger = logging.getLogger(__name__)


class _SoulUnpickler(pickle.Unpickler):
    """Restrict soul loading to known safe types."""

    _SAFE = {
        "builtins": {
            "list",
            "dict",
            "set",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
            "bytes",
            "complex",
            "frozenset",
        },
        "collections": {"OrderedDict", "defaultdict"},
        "numpy": {"ndarray", "dtype", "float64", "float32", "int64", "int32", "array"},
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "numpy.core.numeric": {"_frombuffer", "scalar"},
        __name__: {"DigitalSoul"},
        "sc_neurocore.core.immortality": {"DigitalSoul"},
    }

    def find_class(self, module, name):
        if module in self._SAFE and (name in self._SAFE[module] or "*" in self._SAFE[module]):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden: {module}.{name}")


@dataclass
class DigitalSoul:
    """
    Handles the persistence and 'immortality' of an SC Agent.
    Captures full state (weights, traces, parameters) for restoration.
    """

    agent_id: str
    state_data: Dict[str, Any] = field(default_factory=dict)

    def capture_agent(self, orchestrator):
        """
        Extracts state from all modules registered in the orchestrator.
        """
        logger.info("Soul: Capturing state for Agent '%s'...", self.agent_id)
        for name, module in orchestrator.modules.items():
            if hasattr(module, "get_weights"):
                self.state_data[f"{name}_weights"] = module.get_weights()
            elif hasattr(module, "weights"):
                self.state_data[f"{name}_weights"] = module.weights

            if hasattr(module, "get_state"):
                self.state_data[f"{name}_state"] = module.get_state()

        logger.info("Soul: Captured %d state components.", len(self.state_data))

    def save_soul(self, filepath: str):
        """
        Serializes the soul to a file.
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        except OSError as exc:
            logger.error("Soul: Failed to save to %s: %s", filepath, exc)
            raise
        logger.info("Soul: Saved to %s", filepath)

    @classmethod
    def load_soul(cls, filepath: str) -> "DigitalSoul":
        """
        Restores a soul from a file.
        """
        try:
            with open(filepath, "rb") as f:
                return _SoulUnpickler(f).load()
        except OSError as exc:
            logger.error("Soul: Failed to load from %s: %s", filepath, exc)
            raise

    def reincarnate(self, orchestrator):
        """
        Injects the soul data back into an existing orchestrator's modules.
        """
        logger.info("Soul: Reincarnating Agent '%s'...", self.agent_id)
        for name, module in orchestrator.modules.items():
            w_key = f"{name}_weights"
            s_key = f"{name}_state"

            if w_key in self.state_data:
                if hasattr(module, "weights"):
                    module.weights = self.state_data[w_key]
                if hasattr(module, "_refresh_packed_weights"):
                    module._refresh_packed_weights()

            if s_key in self.state_data:
                if hasattr(module, "v"):  # Typical for neurons
                    module.v = self.state_data[s_key].get("v", 0.0)
        logger.info("Soul: Reincarnation complete.")
