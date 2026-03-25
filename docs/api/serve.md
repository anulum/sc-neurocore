# Streaming Server

Real-time SNN inference server for streaming spike events.

- `SNNServer` — Accept spike events over WebSocket or TCP, run inference on a loaded model, stream output spikes back. Supports batched and single-event modes.

Designed for closed-loop BCI, robotic control, and real-time neural decoding.

```python
from sc_neurocore.serve import SNNServer

server = SNNServer(model=my_snn, port=8080)
server.run()
```

::: sc_neurocore.serve
    options:
      show_root_heading: true
