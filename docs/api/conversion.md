# ANN-to-SNN Conversion

Convert trained PyTorch ANNs to rate-coded spiking neural networks.

## Contract

The conversion package is an optional PyTorch surface. The base package can be
imported without PyTorch; resolving `convert`, `ConvertedSNN`, or
`QCFSActivation` requires a PyTorch-capable environment.

- `convert(model, calibration_data=None, T=16, percentile=99.9)` extracts
  `Linear` and `Conv2d` weights, calibrates activation thresholds from ReLU
  layers when calibration data is supplied, and returns a deterministic
  `ConvertedSNN`.
- `ConvertedSNN.run(x)` rate-codes NumPy input with a fixed RNG seed and returns
  output spike counts for one vector or a batch.
- `ConvertedSNN.classify(x)` returns the argmax class index from output spike
  counts.
- `QCFSActivation` replaces ReLU during conversion-aware training by clipping
  activations to `[0, theta]` and quantising them to `T + 1` spike-rate levels
  with a straight-through gradient.

## Verification

The public conversion files are covered by the scoped NumPy-docstring policy:

- `src/sc_neurocore/conversion/__init__.py`
- `src/sc_neurocore/conversion/ann_to_snn.py`
- `src/sc_neurocore/conversion/qcfs.py`

Focused production tests live in `tests/test_conversion.py` and
`tests/test_conversion_ann_snn.py`. They exercise real PyTorch modules,
conversion calibration, `ConvertedSNN.run`, `ConvertedSNN.classify`, QCFS range
and gradient behaviour, and the layer-extraction contract.

## Converter

::: sc_neurocore.conversion.ann_to_snn
    options:
      show_root_heading: true
      members:
        - convert
        - ConvertedSNN

## QCFS Activation

::: sc_neurocore.conversion.qcfs
    options:
      show_root_heading: true
      members:
        - QCFSActivation
