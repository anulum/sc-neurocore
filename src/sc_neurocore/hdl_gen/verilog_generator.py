from typing import Any, Optional
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class VerilogGenerator:
    """
    Generates Top-Level Verilog for a defined SC Network.
    """

    def __init__(self, module_name="sc_network_top"):  # type: ignore
        self.module_name = module_name
        self.layers = []
        self.wires = []
        self.instances = []

    def add_layer(self, layer_type: str, name: str, params: Dict[str, Any]):  # type: ignore
        self.layers.append({"type": layer_type, "name": name, "params": params})

    def generate(self) -> str:
        """
        Emits Verilog code.
        """
        code = f"module {self.module_name} (\n"
        code += "    input wire clk,\n"
        code += "    input wire rst_n,\n"
        # Determine I/O from first/last layer logic (simplified)
        code += "    input wire [7:0] input_bus,\n"
        code += "    output wire [7:0] output_bus\n"
        code += ");\n\n"

        code += "    // Internal Signals\n"
        # Generate wires for connections
        for i in range(len(self.layers) - 1):
            code += f"    wire [7:0] layer_{i}_to_{i+1};\n"

        code += "\n"

        # Instantiate Layers
        for i, layer in enumerate(self.layers):
            l_type = layer["type"]
            l_name = layer["name"]

            # Simple Dense Layer instantiation logic
            if l_type == "Dense":
                code += f"    // Layer {i}: {l_name}\n"
                code += "    sc_dense_layer_core #(\n"
                code += f"        .NUM_NEURONS({layer['params'].get('n_neurons', 10)})\n"
                code += f"    ) {l_name}_inst (\n"
                code += "        .clk(clk),\n"
                code += "        .rst_n(rst_n),\n"

                # Connect Input
                if i == 0:
                    code += "        .input_bus(input_bus),\n"
                else:
                    code += f"        .input_bus(layer_{i-1}_to_{i}),\n"

                # Connect Output
                if i == len(self.layers) - 1:
                    code += "        .output_bus(output_bus)\n"
                else:
                    code += f"        .output_bus(layer_{i}_to_{i+1})\n"

                code += "    );\n\n"

        code += "endmodule\n"
        return code

    def save_to_file(self, path: str):  # type: ignore
        try:
            with open(path, "w") as f:
                f.write(self.generate())
        except OSError as exc:
            logger.error("Failed to write Verilog to %s: %s", path, exc)
            raise
