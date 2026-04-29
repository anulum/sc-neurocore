# Compiler Service Contract

`sc_neurocore.compiler_service` defines the local contract for a future
compiler service that coordinates surrogate SC optimisation, digital-twin
replay, and live FPGA update packages. It is a deterministic boundary layer:
it does not open sockets, run vendor tools, or claim deployed service
infrastructure.

```python
from sc_neurocore.compiler_service import (
    CompilerServiceRequest,
    DigitalTwinSyncContract,
    build_compiler_service_response,
)
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import TargetHardwareProfile

target = TargetHardwareProfile(
    name="pynq-z2",
    budget=HardwareBudget(max_luts=12_000, max_power_mw=2_500.0),
)
request = CompilerServiceRequest(
    request_id="req-001",
    target=target,
    network=(LayerProfile("hidden", mac_count=256),),
    changed_fields=("weights", "lfsr_seeds"),
    twin_sync=DigitalTwinSyncContract(session_id="twin-a"),
)
response = build_compiler_service_response(request)
```

## Update Classes

- `hot_swap` — register/configuration updates such as weights, thresholds,
  LFSR seeds, bitstream length, decorrelator, or routing table.
- `partial_reconfiguration` — overlay-level changes such as tile mapping or
  AER route overlays.
- `full_resynthesis_required` — HDL, clock, PDK, I/O shape, layer-count, or
  top-module changes.

Every package includes validation gates for digital-twin replay, telemetry
bounds, and rollback packaging before live hardware application.

::: sc_neurocore.compiler_service
    options:
      show_root_heading: true
      members:
        - LiveUpdateKind
        - DigitalTwinSyncContract
        - LiveUpdatePolicy
        - CompilerServiceRequest
        - LiveUpdatePackage
        - CompilerServiceResponse
        - build_compiler_service_contract
        - plan_live_update
        - build_compiler_service_response
