# Export

Model export to standard interchange formats.

- `ONNXExporter` — Export SC networks to ONNX (`.onnx` protobuf) or JSON (`.json` legacy). SC layers map to custom domain `sc_neurocore` with op types `SC_Dense` and `SC_Custom`. Standard layers use ONNX standard ops.

Two export paths:
- `.onnx` — Standard ONNX ModelProto via the `onnx` library (optional dependency)
- `.json` — Dependency-free JSON schema for lightweight deployment

```python
from sc_neurocore.export import ONNXExporter

exporter = ONNXExporter()
exporter.export(model, "model.onnx")
```

::: sc_neurocore.export.onnx_exporter
    options:
      show_root_heading: true
