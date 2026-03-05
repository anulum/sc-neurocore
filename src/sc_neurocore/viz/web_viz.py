# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class WebVisualizer:
    """
    Generates a standalone HTML file to visualize the SC Network.
    """

    @staticmethod
    def generate_html(layers: list[Any], filename="network_viz.html"):  # type: ignore
        # 1. Build Graph Data
        nodes = []
        links = []

        # Input Node
        nodes.append({"id": "Input", "group": 0})

        for i, layer in enumerate(layers):
            layer_name = f"L{i}_{layer.__class__.__name__}"

            # Layer Node (representing the whole layer for simplicity)
            nodes.append(
                {"id": layer_name, "group": i + 1, "neurons": getattr(layer, "n_neurons", "?")}
            )

            # Link from prev
            prev = "Input" if i == 0 else f"L{i-1}_{layers[i-1].__class__.__name__}"
            links.append({"source": prev, "target": layer_name, "value": 1})

        data = {"nodes": nodes, "links": links}
        json_str = json.dumps(data)

        # 2. Embed in HTML Template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SC-NeuroCore Visualization</title>
    <style>
        body {{ font-family: sans-serif; background: #111; color: #eee; }}
        #graph {{ width: 800px; height: 600px; border: 1px solid #333; margin: 20px auto; }}
        .node {{ fill: #69b3a2; stroke: #fff; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        text {{ fill: #eee; font-size: 12px; }}
    </style>
</head>
<body>
    <h1 style="text-align:center">SC-NeuroCore Topology</h1>
    <div id="graph_container" style="text-align:center">
        <canvas id="graph" width="800" height="600"></canvas>
    </div>
    <script>
        const graphData = {json_str};
        const canvas = document.getElementById('graph');
        const ctx = canvas.getContext('2d');
        
        // Simple Force Layout Simulation
        const nodes = graphData.nodes;
        const links = graphData.links;
        
        // Initial random positions
        nodes.forEach(n => {{
            n.x = Math.random() * 800;
            n.y = Math.random() * 600;
            n.vx = 0; n.vy = 0;
        }});
        
        function draw() {{
            ctx.fillStyle = "#111";
            ctx.fillRect(0,0,800,600);
            
            // Draw Links
            ctx.strokeStyle = "#555";
            links.forEach(l => {{
                const src = nodes.find(n => n.id === l.source);
                const tgt = nodes.find(n => n.id === l.target);
                if(src && tgt) {{
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(tgt.x, tgt.y);
                    ctx.stroke();
                }}
            }});
            
            // Draw Nodes
            nodes.forEach(n => {{
                ctx.fillStyle = n.group === 0 ? "#ff5555" : "#55aaff";
                ctx.beginPath();
                ctx.arc(n.x, n.y, 10, 0, 2*Math.PI);
                ctx.fill();
                ctx.fillStyle = "#fff";
                ctx.fillText(n.id, n.x + 12, n.y + 4);
                if(n.neurons) ctx.fillText(n.neurons + " neurons", n.x + 12, n.y + 16);
            }});
        }}
        
        function update() {{
            // Very simple layout: Linear positioning by group
            const groups = [...new Set(nodes.map(n => n.group))];
            const layerHeight = 500 / groups.length;
            
            nodes.forEach(n => {{
                // Target Y based on group
                const ty = 50 + n.group * 100;
                const tx = 400; // Center X
                
                n.x += (tx - n.x) * 0.1;
                n.y += (ty - n.y) * 0.1;
            }});
            
            draw();
            requestAnimationFrame(update);
        }}
        
        update();
    </script>
</body>
</html>
        """

        with open(filename, "w") as f:
            f.write(html_content)
        logger.info("Generated Visualization: %s", filename)
