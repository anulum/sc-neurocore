# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Central Orchestrator for sc-neurocore Agents

"""TensorStream-aware orchestrator for sequencing registered processing modules."""

from __future__ import annotations

from typing import Any, Optional
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from .tensor_stream import TensorStream

logger = logging.getLogger(__name__)


@dataclass
class CognitiveOrchestrator:
    """Central orchestrator that sequences registered modules into a pipeline.

    Connects disparate processing modules and routes a :class:`TensorStream`
    through them in execution order.
    """

    modules: Dict[str, Any] = field(default_factory=dict)
    active_goals: List[str] = field(default_factory=list)
    attention_focus: Optional[str] = None

    def register_module(self, name: str, module_obj: Any) -> None:
        """Register a named module object for later pipeline execution."""
        self.modules[name] = module_obj

    def set_attention(self, module_name: str) -> None:
        """Focus orchestrator resources on a specific module."""
        if module_name in self.modules:
            self.attention_focus = module_name
            logger.info("Orchestrator: Attention focused on '%s'.", module_name)

    def execute_pipeline(self, pipeline: List[str], initial_input: TensorStream) -> TensorStream:
        """Execute a sequence of modules, handling TensorStream conversions.

        Parameters
        ----------
        pipeline : list of str
            Ordered module names to execute; unknown names are skipped.
        initial_input : TensorStream
            Input stream fed to the first module in the pipeline.

        Returns
        -------
        TensorStream
            The stream produced by the final executed module.
        """
        current_stream = initial_input

        for module_name in pipeline:
            if module_name not in self.modules:
                logger.warning("Module %s not found.", module_name)
                continue

            module = self.modules[module_name]

            # Smart dispatch based on module type/method
            if hasattr(module, "forward"):
                # Many layers use 'forward'
                # Check what input it expects (rough heuristic)
                if "Quantum" in module.__class__.__name__:
                    input_data = current_stream.to_bitstream()
                else:
                    input_data = current_stream.to_prob()

                output_data = module.forward(input_data)

                # Wrap output back to stream
                if isinstance(output_data, np.ndarray):
                    if np.iscomplexobj(output_data):
                        current_stream = TensorStream(output_data, "quantum")
                    elif output_data.dtype == np.uint8:
                        current_stream = TensorStream(output_data, "bitstream")
                    else:
                        current_stream = TensorStream(output_data, "prob")

            elif hasattr(module, "step"):
                # Simple neurons or CPGs
                # Process scalar or vector step
                val = current_stream.to_prob()
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    res = np.array([module.step(v) for v in val.flatten()])
                else:
                    res = module.step(val)
                current_stream = TensorStream.from_prob(res)

        return current_stream
