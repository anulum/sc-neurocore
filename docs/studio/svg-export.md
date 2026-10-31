# SVG Vector Export

Export simulation traces as publication-quality SVG vector graphics.
The SVG includes axes, grid lines, colour-coded state variable traces,
spike markers, a legend, and a model name watermark. Suitable for
LaTeX documents, Inkscape editing, and journal figures.

## Export Formats

The Studio supports three data export formats:

| Format | Button | Contents | Use Case |
|--------|--------|----------|----------|
| **SVG** | SVG | Vector traces with axes and legend | Paper figures, presentations |
| **CSV** | CSV | Full-resolution time, scalar state and drive columns, when available | External tools (MATLAB, R, pandas) |
| **JSON** | JSON | Complete response, including raw availability and display projection metadata | Reproducibility, archival |

CSV uses every raw sample, with time and drive rounded to four decimal places
and scalar states to six. A large run may exceed the server's raw-data limit
while still returning a reduced trace for display. In that case, **CSV** and
**Compute STA** show the server's explanation and refuse the operation. They
do not substitute display samples for the full trace. Shorten the simulation
or reduce its state count, then run it again to obtain full-resolution data.

**JSON** remains available and preserves the response's `raw.included: false`
declaration and reason. **SVG** remains available as a figure of the display
trace; its plotted vertices are not a full-resolution data export. Older saved
results without a raw-availability block retain their original array exports.

## Frontend SVG Generation

Click the **SVG** button in the header toolbar after running a simulation.
The browser generates the SVG client-side from the current simulation
result — no server round-trip needed.

The generated SVG:

- **800×400** viewport (configurable via API)
- Dark background (`#0d1117`) matching the Studio theme
- 5 horizontal grid lines for visual reference
- Colour-coded polylines: `#4fc3f7` (cyan), `#81c784` (green),
  `#ffb74d` (orange), `#e57373` (red), `#ce93d8` (purple)
- Spike markers as short red vertical lines at the top
- X-axis label: "time (ms)", Y-axis label: "mV"
- Monospace Y-axis tick labels (5 ticks)
- Legend with colour swatches and variable names
- Model name watermark (top right, subtle)

## Downsampling

Traces with more than 2000 data points are stride-sampled to keep
the SVG file size reasonable (~50–100 KB) while preserving the overall
trace shape. A 100ms simulation at dt=0.01 (10,000 points) produces
2000 SVG polyline vertices.

## Backend API

For headless/automated export (CLI scripts, CI pipelines, paper builds):

```
POST /api/export/svg
Content-Type: application/json

{
  "name": "AdExNeuron",
  "params": {},
  "dt": 0.025,
  "duration": 100.0,
  "current": 0.5,
  "protocol": "step"
}
```

Returns `image/svg+xml` response body. Save directly to file:

```bash
curl -X POST http://localhost:8001/api/export/svg \
  -H "Content-Type: application/json" \
  -d '{"name":"AdExNeuron","dt":0.025,"duration":100,"current":0.5,"protocol":"step"}' \
  -o adex_trace.svg
```

## Python API

The SVG generator is also available as a Python function:

```python
from sc_neurocore.studio.svg_export import traces_to_svg

svg = traces_to_svg(
    time=[0.0, 0.1, 0.2, ...],
    states={"v": [-65.0, -64.8, ...], "w": [0.0, 0.01, ...]},
    spikes=[42, 87, 134],
    model_name="AdExNeuron",
    dt=0.1,
    width=800,
    height=400,
)

with open("figure.svg", "w") as f:
    f.write(svg)
```

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time` | `list[float]` | required | Time points (ms) |
| `states` | `dict[str, list[float]]` | required | State variable traces |
| `spikes` | `list[int]` | `None` | Spike indices into time array |
| `model_name` | `str` | `""` | Watermark text (top right) |
| `dt` | `float` | `0.1` | Timestep for spike index fallback |
| `width` | `int` | `800` | SVG width in pixels |
| `height` | `int` | `400` | SVG height in pixels |

Returns a complete SVG string. Empty data produces a no-data
SVG with "No data" text.

## Editing in Inkscape

The generated SVG uses standard elements (`polyline`, `line`, `text`,
`rect`) with no embedded scripts or fonts. Open in Inkscape to:

- Change colours for journal requirements (e.g. colourblind-safe palette)
- Add annotations or arrows
- Adjust font sizes for specific column widths
- Convert to PDF/EPS for LaTeX `\includegraphics`

## Implementation

- `src/sc_neurocore/studio/svg_export.py` — `traces_to_svg()` generator
- `src/sc_neurocore/studio/app.py` — `POST /api/export/svg` endpoint
- `studio/frontend/src/stores/studio.ts` — `exportSVG()` action
- `tests/test_studio_svg_export.py` — 14 tests
