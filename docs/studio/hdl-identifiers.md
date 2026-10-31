# HDL identifier contract

Studio validates names used in generated HDL through the public library API:

```python
from sc_neurocore.hdl_gen import sanitize_ident

assert sanitize_ident("neuron_state_0", context="state variable") == "neuron_state_0"

try:
    sanitize_ident("module", context="module name")
except ValueError as refusal:
    assert str(refusal) == "Invalid module name: 'module'"
else:
    raise AssertionError("Reserved names must be refused")
```

Accepted names contain one to 64 ASCII characters. They start with a letter
or underscore; subsequent characters may also be digits. The generator's
reserved-word set is refused with a case-sensitive comparison. The API
returns the original name and does not replace characters, truncate, escape
or normalise it. Invalid names raise `ValueError`, with the supplied context
in the message.

Project loading, native bit-true execution and co-simulation use this same
validator. A refused project identifier must be corrected before generation;
changing a name silently could change circuit connections or replay identity.
The identifier contract does not attest numerical accuracy, synthesis
support, resource use or execution on a physical device.
