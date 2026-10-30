NIR Bridge
==========

SC-NeuroCore provides bidirectional conversion between
`NIR <https://neuroir.org/>`_ graphs and SC-NeuroCore networks,
with explicit execution settings and a validated hardware lowering path.

.. contents:: On this page
   :local:

Quick Start
-----------

.. code-block:: python

   import nir
   from sc_neurocore.nir_bridge import from_nir

   graph = nir.read("model.nir")
   network = from_nir(graph)
   results = network.run({"input": np.array([1.0, 0.5])}, steps=100)

Module: ``sc_neurocore.nir_bridge``
-----------------------------------

.. automodule:: sc_neurocore.nir_bridge
   :members:
   :show-inheritance:

Parser
------

.. automodule:: sc_neurocore.nir_bridge.parser
   :members:
   :show-inheritance:

Node Map
--------

.. automodule:: sc_neurocore.nir_bridge.node_map
   :members:
   :show-inheritance:

Export
------

.. automodule:: sc_neurocore.nir_bridge.export
   :members:
   :show-inheritance:

Framework inference
-------------------

The package facade exposes ``from_snntorch`` and ``to_snntorch`` lazily.
The qualified lane uses separate pinned profiles, CPU float64 vectors and
real snnTorch cells. Unsupported reset, topology and stochastic bitstream
contracts are refused. See the NIR integration guide for the exact preserved
semantics and the distinction between native Sinabs and SC reference execution.
