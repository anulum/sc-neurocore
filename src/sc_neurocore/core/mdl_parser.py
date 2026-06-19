# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parser for Mind Description Language (MDL)

"""Mind Description Language helpers for serialising orchestrator state."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MDLSpecification:
    """Serializable MDL payload containing architecture and state sections."""

    version: str = "1.0"
    agent_name: str = "Unknown"
    architecture: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


class MindDescriptionLanguage:
    """Parser for the Mind Description Language (MDL).

    A universal, substrate-independent format for archiving an agent's
    architecture and state.
    """

    @staticmethod
    def encode(orchestrator: Any, agent_name: str) -> str:
        """Export the orchestrator state to a YAML MDL string."""
        architecture = {}
        state = {}

        for name, module in orchestrator.modules.items():
            # Abstract representation
            architecture[name] = {"type": module.__class__.__name__, "module": module.__module__}

            if hasattr(module, "get_state"):
                state[name] = module.get_state()
            elif hasattr(module, "weights"):
                # Convert numpy to list for YAML
                state[name] = {"weights": module.weights.tolist()}

        mdl = MDLSpecification(agent_name=agent_name, architecture=architecture, state=state)
        return str(yaml.dump(asdict(mdl), sort_keys=False))

    @staticmethod
    def decode(mdl_string: str) -> Dict[str, Any]:
        """Parse an MDL string back into a dictionary for reconstruction."""
        data = yaml.safe_load(mdl_string)
        if not isinstance(data, dict):
            raise ValueError("MDL string must decode into a mapping")
        decoded: dict[str, Any] = data
        logger.info(
            "MDL: Decoded mind of '%s' (v%s)",
            decoded.get("agent_name", "Unknown"),
            decoded.get("version"),
        )
        return decoded
