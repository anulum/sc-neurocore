# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parser for Mind Description Language (MDL)

from typing import Any
import logging
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class MDLSpecification:
    version: str = "1.0"
    agent_name: str = "Unknown"
    architecture: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


class MindDescriptionLanguage:
    """
    Parser for Mind Description Language (MDL).
    A universal, substrate-independent format for archiving consciousness.
    """

    @staticmethod
    def encode(orchestrator, agent_name: str) -> str:
        """
        Exports the Orchestrator state to YAML MDL.
        """
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
        return yaml.dump(asdict(mdl), sort_keys=False)

    @staticmethod
    def decode(mdl_string: str) -> Dict[str, Any]:
        """
        Parses MDL back to a dictionary (for reconstruction).
        """
        data = yaml.safe_load(mdl_string)
        logger.info(
            "MDL: Decoded mind of '%s' (v%s)",
            data.get("agent_name", "Unknown"),
            data.get("version"),
        )
        return data
