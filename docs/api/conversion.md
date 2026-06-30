# ANN-to-SNN Conversion

Convert trained PyTorch ANNs to rate-coded spiking neural networks.

## Contract

The conversion package is an optional PyTorch surface. The base package can be
imported without PyTorch; resolving `convert`, `ConvertedSNN`, or
`QCFSActivation` requires a PyTorch-capable environment.

- `convert(model, calibration_data=None, T=None, percentile=99.9)` extracts
  `Linear` and `Conv2d` weights and selects a conversion route from the model's
  activations, returning a deterministic `ConvertedSNN`:
  - **Threshold-balancing route** (ReLU models, Diehl et al. 2015) — calibrates
    activation thresholds from ReLU layers when calibration data is supplied,
    starts each IF neuron from rest, and defaults `T` to 16.
  - **QCFS route** (QCFS-trained models, Bu et al. 2022) — takes each
    `QCFSActivation`'s *learned* threshold directly, pre-loads each IF neuron to
    `theta / 2` (the optimal shift that cancels the quantisation bias), ignores
    calibration data, and adopts the layers' trained step budget when `T` is
    left unset.
- `ConvertedSNN.run(x)` rate-codes NumPy input with a fixed RNG seed and returns
  output spike counts for one vector or a batch. `initial_membrane_fraction`
  controls the per-layer membrane pre-load (`0.0` rest, `0.5` QCFS shift).
- `ConvertedSNN.classify(x)` returns the argmax class index from output spike
  counts.
- `QCFSActivation` replaces ReLU during conversion-aware training by clipping
  activations to `[0, theta]` and quantising them to `T + 1` spike-rate levels
  with a straight-through gradient.
- `replace_relu_with_qcfs(model, T=8, theta=1.0, learn_theta=True)` substitutes
  every `ReLU`/`ReLU6` in a model (recursing through submodules) with a
  `QCFSActivation`, preparing the network for QCFS conversion-aware fine-tuning.

## Verification

The public conversion files are covered by the scoped NumPy-docstring policy:

- `src/sc_neurocore/conversion/__init__.py`
- `src/sc_neurocore/conversion/ann_to_snn.py`
- `src/sc_neurocore/conversion/qcfs.py`

Focused production tests live in `tests/test_conversion.py` and
`tests/test_conversion_ann_snn.py`. They exercise real PyTorch modules, the
threshold-balancing and QCFS conversion routes, `ConvertedSNN.run`,
`ConvertedSNN.classify`, the membrane shift, the ReLU→QCFS substitution helper,
QCFS range and gradient behaviour, and the layer-extraction contract.

## Converter

::: sc_neurocore.conversion.ann_to_snn
    options:
      show_root_heading: true
      members:
        - convert
        - ConvertedSNN
        - replace_relu_with_qcfs

## QCFS Activation

::: sc_neurocore.conversion.qcfs
    options:
      show_root_heading: true
      members:
        - QCFSActivation
