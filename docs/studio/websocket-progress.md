# WebSocket Progress Streaming

Long-running Studio operations (characterisation, heatmap, model scan)
stream progress updates via WebSocket. A progress bar appears in the UI
showing the current step, percentage, and description.

## Supported Operations

| Operation | Steps | Typical Duration | What It Does |
|-----------|:-----:|:----------------:|-------------|
| `characterize` | ~50 | 5-30s | f-I curve (20 pts) + sensitivity (15 params × 2) |
| `heatmap` | N×M | 10-60s | Two-parameter sweep (e.g. 15×15 = 225 sims) |
| `scan` | 118 | 30-120s | Classify all 118 models by firing pattern |

## Protocol

Connect via WebSocket to `/ws/progress`, send a JSON request, receive
progress frames:

```
Client → Server: {"op": "characterize", "config": {"name": "AdExNeuron", "current": 15.0}}

Server → Client: {"type": "progress", "step": "trace", "pct": 0, "msg": "Running default simulation"}
Server → Client: {"type": "progress", "step": "fi_curve", "pct": 5, "msg": "f-I curve 1/20"}
Server → Client: {"type": "progress", "step": "fi_curve", "pct": 10, "msg": "f-I curve 2/20"}
...
Server → Client: {"type": "progress", "step": "sensitivity", "pct": 60, "msg": "Sensitivity 3/15: tau_m"}
...
Server → Client: {"type": "complete", "pct": 100, "result": {...}}
```

## Message Types

| Type | Fields | Meaning |
|------|--------|---------|
| `progress` | pct, step, msg | Operation in progress |
| `complete` | pct=100, result | Operation finished successfully |
| `error` | msg | Operation failed |
| `heartbeat` | — | Connection alive, worker still running |

## Frontend Integration

The progress bar appears below the header when `isSimulating` is true
and `progressMsg` is non-empty. It shows:

- Current step description (e.g. "f-I curve 12/20")
- Percentage bar (0-100%)
- Percentage number

When WebSocket is unavailable (e.g. behind a proxy that strips WS),
the frontend falls back to the standard HTTP endpoint without progress.

## API

### WebSocket: /ws/progress

**Request:**
```json
{
  "op": "characterize",
  "config": {
    "name": "AdExNeuron",
    "params": {"a": 4.0, "b": 0.0805},
    "current": 15.0,
    "duration": 200.0,
    "dt": 0.1
  }
}
```

**Operations:**
- `characterize` — full model characterisation with f-I + sensitivity
- `heatmap` — 2D parameter sweep (requires `param_x`, `param_y`, `x_min`/`x_max`/`x_steps`, `y_min`/`y_max`/`y_steps` in config)
- `scan` — scan all 118 models (no config needed)

## Implementation

The backend runs the long operation in a daemon thread, pushing progress
messages to a `queue.Queue`. The async WebSocket handler polls the queue
and forwards messages to the client. This avoids blocking the event loop
while the CPU-intensive simulation runs.

```
Client ←WebSocket→ AsyncIO handler ←Queue→ Daemon thread (simulation)
```
