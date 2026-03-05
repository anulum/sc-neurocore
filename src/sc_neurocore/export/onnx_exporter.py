# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any, Optional
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SCOnnxExporter:
    """
    Exports SC Networks to ONNX-compatible JSON schema (or Protobuf if libs available).
    Standard ONNX doesn't support 'StochasticBitstream' types natively.
    We map SC layers to integer operations (MatMulInteger) or custom domains.
    """

    @staticmethod
    def export(layers: list[Any], filename: str):  # type: ignore
        """
        Export layer list to a JSON definition.
        """
        graph = {
            "producer_name": "sc-neurocore",
            "producer_version": "2.0.0",
            "nodes": [],
            "inputs": [],
            "outputs": [],
        }

        # Define Input
        # Assuming input 0 is the start
        graph["inputs"].append(  # type: ignore
            {"name": "input_0", "type": "tensor(float)", "shape": ["batch", layers[0].n_inputs]}
        )

        previous_output = "input_0"

        for i, layer in enumerate(layers):
            node_name = f"Layer_{i}"
            output_name = f"output_{i}"

            # Detect Layer Type
            layer_type = layer.__class__.__name__

            node = {
                "op_type": (
                    "SC_Dense"
                    if "Dense" in layer_type or "Vectorized" in layer_type
                    else "SC_Custom"
                ),
                "name": node_name,
                "input": [previous_output],
                "output": [output_name],
                "attributes": {
                    "n_neurons": getattr(layer, "n_neurons", -1),
                    "length": getattr(layer, "length", 256),
                },
            }

            # Embed Weights if possible
            if hasattr(layer, "weights"):
                # We can save weights as an initializer in a real ONNX
                # Here we just flag it
                node["attributes"]["has_weights"] = True  # type: ignore
                # We don't dump large weights to JSON usually,
                # but we could save a reference to an external .npy
                np.save(f"{filename}_layer_{i}_weights.npy", layer.weights)
                node["attributes"]["weights_file"] = f"{filename}_layer_{i}_weights.npy"  # type: ignore

            graph["nodes"].append(node)  # type: ignore
            previous_output = output_name

        # Define Final Output
        graph["outputs"].append({"name": previous_output, "type": "tensor(float)"})  # type: ignore

        try:
            with open(filename, "w") as f:
                json.dump(graph, f, indent=4)
        except OSError as exc:
            logger.error("Failed to export ONNX schema to %s: %s", filename, exc)
            raise

        logger.info("Exported ONNX-Schema JSON to %s", filename)
        logger.info("Note: Weights saved as .npy sidecars.")
