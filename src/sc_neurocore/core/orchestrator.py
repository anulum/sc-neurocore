
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from .tensor_stream import TensorStream

logger = logging.getLogger(__name__)

@dataclass
class CognitiveOrchestrator:
    """
    Central Orchestrator for sc-neurocore Agents.
    Connects disparate modules into a functional pipeline.
    """
    modules: Dict[str, Any] = field(default_factory=dict)
    active_goals: List[str] = field(default_factory=list)
    attention_focus: Optional[str] = None
    
    def register_module(self, name: str, module_obj: Any):
        self.modules[name] = module_obj
        
    def set_attention(self, module_name: str):
        """Focuses resources on a specific module."""
        if module_name in self.modules:
            self.attention_focus = module_name
            logger.info("Orchestrator: Attention focused on '%s'.", module_name)

    def execute_pipeline(self, pipeline: List[str], initial_input: TensorStream) -> TensorStream:
        """
        Executes a sequence of modules.
        Automatically handles TensorStream conversions.
        """
        current_stream = initial_input
        
        for module_name in pipeline:
            if module_name not in self.modules:
                logger.warning("Module %s not found.", module_name)
                continue
                
            module = self.modules[module_name]
            
            # Smart dispatch based on module type/method
            if hasattr(module, 'forward'):
                # Many layers use 'forward'
                # Check what input it expects (rough heuristic)
                if 'Quantum' in module.__class__.__name__:
                    input_data = current_stream.to_bitstream()
                else:
                    input_data = current_stream.to_prob()
                    
                output_data = module.forward(input_data)
                
                # Wrap output back to stream
                if isinstance(output_data, np.ndarray):
                    if np.iscomplexobj(output_data):
                        current_stream = TensorStream(output_data, 'quantum')
                    elif output_data.dtype == np.uint8:
                        current_stream = TensorStream(output_data, 'bitstream')
                    else:
                        current_stream = TensorStream(output_data, 'prob')
                        
            elif hasattr(module, 'step'):
                # Simple neurons or CPGs
                # Process scalar or vector step
                val = current_stream.to_prob()
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    res = np.array([module.step(v) for v in val.flatten()])
                else:
                    res = module.step(val)
                current_stream = TensorStream.from_prob(res)
                
        return current_stream
