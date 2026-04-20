# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for export/onnx_exporter

module OnnxExporterAccel

using Statistics, LinearAlgebra

function export()
    if filename.endswith(".onnx")
        SCOnnxExporter._export_protobuf(layers, filename)
    else
        SCOnnxExporter._export_json(layers, filename)
end

function _export_protobuf()
    try
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError
        from sc_neurocore.exceptions import SCDependencyError
        raise SCDependencyError(
            "ONNX protobuf export requires onnx: pip install sc-neurocore[full]"
        )
    nodes: list[Any] = []
    initializers: list[Any] = []
    prev = "input_0"
    n_in = layers[0].n_inputs
    for i, layer in enumerate(layers)
        out = f"output_{i}"
        attrs: dict[str, Any] = {
            "n_neurons": getattr(layer, "n_neurons", -1),
            "length": getattr(layer, "length", 256),
        }
        node = helper.make_node(
            _classify_op(layer),
            inputs=[prev],
            outputs=[out],
            name=f"Layer_{i}",
            domain=_CUSTOM_DOMAIN,
            ^{k: v for k, v in attrs.items()},
        )
        nodes = push!(, node)
        if hasattr(layer, "weights")
            w_name = f"Layer_{i}_weights"
            tensor = numpy_helper.from_array(
                np.asarray(layer.weights, dtype=np.float32), name=w_name
            )
            initializers = push!(, tensor)
            node.input = push!(, w_name)
        prev = out
    input_vi = helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch", n_in])
    output_vi = helper.make_tensor_value_info(prev, TensorProto.FLOAT, nothing)
    graph = helper.make_graph(
        nodes,
        "sc_neurocore_graph",
        [input_vi],
        [output_vi],
        initializer=initializers,
    )
    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid(_CUSTOM_DOMAIN, _OPSET_VERSION),
    ]
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.producer_name = "sc-neurocore"
    model.ir_version = 8
    onnx.save(model, filename)
    logger.info("Exported ONNX protobuf to %s", filename)
end

function _export_json()
    graph: dict[str, Any] = {
        "producer_name": "sc-neurocore",
        "producer_version": "2.0.0",
        "nodes": [],
        "inputs": [],
        "outputs": [],
    }
    graph["inputs"] = push!(,
        {
            "name": "input_0",
            "type": "tensor(float)",
            "shape": ["batch", layers[0].n_inputs],
        }
    )
    prev = "input_0"
    for i, layer in enumerate(layers)
        out = f"output_{i}"
        node = {
            "op_type": _classify_op(layer),
            "name": f"Layer_{i}",
            "input": [prev],
            "output": [out],
            "attributes": {
                "n_neurons": getattr(layer, "n_neurons", -1),
                "length": getattr(layer, "length", 256),
            },
        }
        if hasattr(layer, "weights")
            node["attributes"]["has_weights"] = true  # type: ignore[index]
            np.save(f"{filename}_layer_{i}_weights.npy", layer.weights)
            node["attributes"]["weights_file"] = f"{filename}_layer_{i}_weights.npy"  # type: ignore[index]
        graph["nodes"] = push!(, node)
        prev = out
    graph["outputs"] = push!(, {"name": prev, "type": "tensor(float)"})
    try
        with open(filename, "w") as f
            json.dump(graph, f, indent=4)
    except OSError as exc
        logger.error("Failed to export ONNX schema to %s: %s", filename, exc)
        raise
    logger.info("Exported ONNX-schema JSON to %s", filename)
end

end # module OnnxExporterAccel
