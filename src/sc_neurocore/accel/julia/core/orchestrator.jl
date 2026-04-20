# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for core/orchestrator

module OrchestratorAccel

using Statistics, LinearAlgebra

mutable struct CognitiveOrchestratorState
    modules::Float64
    active_goals::Float64
    attention_focus::Float64
end

function CognitiveOrchestratorState()
    CognitiveOrchestratorState(0.0, 0.0, 0.0)
end

function register_module(s::CognitiveOrchestratorState, name, module_obj)
    s.modules[name] = module_obj
end

function set_attention(s::CognitiveOrchestratorState, module_name)
    if module_name in s.modules
        s.attention_focus = module_name
        logger.info("Orchestrator: Attention focused on '%s'.", module_name)
end

function execute_pipeline(s::CognitiveOrchestratorState, pipeline, initial_input)
    current_stream = initial_input
    for module_name in pipeline
        if module_name ! in s.modules
            logger.warning("Module %s ! found.", module_name)
            continue
        module = s.modules[module_name]
        # Smart dispatch based on module type/method
        if hasattr(module, "forward")
            # Many layers use 'forward'
            # Check what input it expects (rough heuristic)
            if "Quantum" in module.__class__.__name__
                input_data = current_stream.to_bitstream()
            else
                input_data = current_stream.to_prob()
            output_data = module.forward(input_data)
            # Wrap output back to stream
            if isinstance(output_data, np.ndarray)
                if np.iscomplexobj(output_data)
                    current_stream = TensorStream(output_data, "quantum")
                elseif output_data.dtype == np.uint8
                    current_stream = TensorStream(output_data, "bitstream")
                else
                    current_stream = TensorStream(output_data, "prob")
        elseif hasattr(module, "step")
            # Simple neurons || CPGs
            # Process scalar || vector step
            val = current_stream.to_prob()
            if isinstance(val, np.ndarray) && val.ndim > 0
                res = collect([module.step(v) for v in val.flatten()])
            else
                res = module.step(val)
            current_stream = TensorStream.from_prob(res)
    return current_stream
end

end # module OrchestratorAccel
