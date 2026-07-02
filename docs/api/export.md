# Export

Model export to standard interchange formats.

Two ONNX-oriented export paths are maintained:

- `SCOnnxExporter` — file-oriented exporter for SC networks that writes ONNX
  protobuf files when the optional `onnx` dependency is installed and JSON
  sidecars for lightweight deployment.
- `sc_neurocore.export.onnx_export.ONNXExporter` — dependency-free graph
  exporter for SC-IR-style nodes. It emits a JSON-serializable `ONNXGraph`
  envelope using the custom `sc.neurocore` domain.

```python
from sc_neurocore.export import SCOnnxExporter

exporter = SCOnnxExporter()
exporter.export(model, "model.onnx")
```

For dependency-free graph export, use the SC-IR graph exporter directly:

```python
from sc_neurocore.export.onnx_export import ONNXExporter

graph = ONNXExporter().export(ir_graph, {"input_a": (128, 1024)})
payload = graph.to_dict()
```

Final graph output metadata follows the actual final emitted node. For example,
a final `SC_POPCOUNT` node produces an ONNX `int32` tensor (`elem_type=6`),
while stochastic bitstream outputs remain `bool` tensors (`elem_type=9`).
Mapped SC-IR node types that do not have a shape inference rule fail closed
instead of silently emitting a guessed `(1,)` output.

For MLIR/SSA lowering, use the compiler exporter on the same graph-style
surface:

```python
from sc_neurocore.export.compiler_export import CompilerExporter

mlir_text = CompilerExporter().export_to_mlir(ir_graph, {"input_a": (128, 1024)})
```

`CompilerExporter` supports the `mlir` target and validates the graph before
emission. Empty graphs, duplicate node IDs, duplicate output edges, unsupported
node types, wrong node arity, missing external input shapes, non-positive tensor
dimensions, and output names that collide with graph inputs raise `ValueError`
before SSA text is emitted. MLIR-facing input names are validated with the
shared HDL identifier guard; invalid identifiers fail closed instead of being
rewritten.

::: sc_neurocore.export.onnx_exporter
    options:
      show_root_heading: true

::: sc_neurocore.export.onnx_export
    options:
      show_root_heading: true

::: sc_neurocore.export.compiler_export
    options:
      show_root_heading: true
