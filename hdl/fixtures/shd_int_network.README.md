<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Integer SHD network for the bit-exact core -->

# Integer SHD network

`shd_int_network.json` is the trained SHD network in the integer form of
`tools/shd_int_reference.py`. It loads with `IntNetwork.from_json` and runs on
`hdl/sc_shd_int_core.v`, which computes it bit for bit.

**Network:** 140 counts → 128 → 128 → 20, with integer axonal delays 0..30 in
both hidden layers, int8 weights and the standard LIF neuron.

| Field | Value |
|---|---|
| sha256 of the exported JSON | `6bb6f1387253ea5db78358d9d7076c71424e8668b9cd65e70ad14b189b0bf450` (the file adds one trailing newline) |
| `frac_bits` | 0 |
| thresholds | 72 (layer 1), 51 (layer 2) |
| membrane widths | 22 and 17 bits |
| non-zero weights | 13 652, 12 265 and 1 638 per layer |

## Provenance

**Training:** `data/masquelier_shd/train_dcls_max.py`.
- DCLS "max" delays, with SIG annealed by cosine from 15 to 0.
- Standard LIF neurons, seed 0.
- L1 weight penalty 3e-5, then iterative epsilon pruning with fine-tuning.
- The checkpoint is the one chosen by validation accuracy measured with rounded
  delays and SIG = 0.
- Checkpoint sha256: `b37d345a400c71796274e002a3811cbe422a0c30c684805208cc76adbad0251f`.

**Export:** `python tools/export_shd_int_network.py TRAINING_REPO RUN_DIR best.pth 0 OUT_JSON`.
1. Each delay layer is evaluated at SIG = 0 with rounded positions, and every
   channel's kernel must be a single tap.
2. The per-channel kernel gains fold into the next layer's weights.
3. Each layer is quantised to int8 with one scale.
4. The thresholds become `ceil(2^F / s)`.

Running the export again on the checkpoint reproduces the JSON byte for byte; the file adds one trailing newline.

**Reproduction environment:** the export was run again on 2026-11-07 on CPU with
Python 3.12.3, torch 2.15.0, spikingjelly 0.0.0.0.14, dcls 0.1.1 and numpy 2.5.4.
Its output had the sha256 above. The exporter needs these packages and the
training repository, so CI cannot run it; `tests/test_shd_int_network.py` pins
the committed artefact instead.

**Input:** 140 channels of 4-bit counts per 2 ms step. Each count is the sum of
five neighbouring SHD channels, framed as in the training repository's loader.
