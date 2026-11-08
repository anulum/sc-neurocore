#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Export a DCLS-max SHD checkpoint to the integer network
"""Export a DCLS-max SHD checkpoint (``train_dcls_max.py``) to ``tools/shd_int_reference.py``.

The delay layers are evaluated with rounded positions and SIG = 0, and each channel's kernel
is read as the layer's impulse response; every kernel must be a single tap (one integer delay
per input channel). The per-channel kernel gains fold into the next dense layer's columns, each
dense layer is quantised to int8 with one per-tensor scale (round half to even), and the LIF
thresholds become ``ceil(2^F / s)`` in membrane units of ``s * 2^-F``.

Needs torch, spikingjelly, DCLS and the training repository; run with that repository as the
first argument:

    export_shd_int_network.py TRAINING_REPO RUN_DIR CHECKPOINT FRAC_BITS OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

REPO, RUN, CKPT, FRAC, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
os.environ["WANDB_MODE"] = "disabled"

import numpy as np  # noqa: E402
import shd_int_reference as ref  # noqa: E402
import torch  # noqa: E402
from spikingjelly.activation_based import neuron  # noqa: E402

os.chdir(REPO)
from configs.config_SHD import Config  # noqa: E402
from src.modules import dcls_module  # noqa: E402
from src.SHD.snn import SNN_axonal_feedforward_delays  # noqa: E402


class CompatibleLIFNode(neuron.LIFNode):
    """The training script's standard LIF with the Vmin-LIF constructor signature."""

    def __init__(
        self,
        tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        v_inf=None,
        beta_v_inf=None,
        surrogate_function=None,
        detach_reset=False,
        step_mode="s",
        backend="torch",
        store_v_seq=False,
    ):
        del v_inf, beta_v_inf
        super().__init__(
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )


def main() -> None:
    """Export the checkpoint named on the command line and print its digests as JSON.

    The printed record holds the checkpoint and network SHA-256 digests, the per-layer
    int8 scales, the two LIF thresholds and the two membrane widths.

    Raises
    ------
    SystemExit
        If the run does not use the standard LIF neuron.
    AssertionError
        If the model structure differs from the integer network or a delay kernel is
        not a single tap.
    """
    with open(os.path.join(RUN, "config.json")) as f:
        meta = json.load(f)
    if meta.get("neuron_module") != "standard_lif":
        raise SystemExit("the integer LIF models the standard LIF only")
    config = Config()
    config.DCLSversion = "max"
    config.hidden_layers = [ref.N_HIDDEN, ref.N_HIDDEN]
    config.neuron_module = CompatibleLIFNode
    ckpt_path = os.path.join(RUN, CKPT)
    model = SNN_axonal_feedforward_delays(config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True)["net"])
    model.eval()

    dcls = [m for m in model.layers if isinstance(m, dcls_module)]
    linear = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    neurons = [m for m in model.layers if isinstance(m, neuron.LIFNode)]
    assert len(dcls) == 2 and len(linear) == 3 and len(neurons) == 2
    for m in dcls:
        assert tuple(m.P.shape) == (1, m.in_channels, 1, 1), "delays must be one per input channel"
    for m in linear:
        assert m.bias is None
    for m in neurons:
        assert m.decay_input is False and m.v_reset == 0.0 and m.v_threshold == 1.0 and m.tau == 4.0

    delays, gains = [], []
    with torch.no_grad():
        for m in dcls:
            m.P.data.round_()
            m.SIG.data.fill_(0.0)
            x = torch.zeros(64, 1, m.in_channels)
            x[0, 0, :] = 1.0
            y = m(x)[:, 0, :].numpy().astype(np.float64)
            d = y.argmax(0)
            g = y[d, np.arange(m.in_channels)]
            rest = y.copy()
            rest[d, np.arange(m.in_channels)] = 0.0
            assert np.abs(rest).max() < 1e-6, "a delay kernel is not a single tap"
            delays.append(d.astype(np.int64))
            gains.append(g)
    w = [m.weight.detach().numpy().astype(np.float64) for m in linear]
    folded = [w[0] * gains[0][None, :], w[1] * gains[1][None, :], w[2]]
    scales = [float(np.abs(f).max() / ref.WEIGHT_MAX) for f in folded]
    q = [np.rint(f / s).astype(np.int64) for f, s in zip(folded, scales)]
    net = ref.IntNetwork(
        frac_bits=FRAC,
        d1=delays[0],
        d2=delays[1],
        q1=q[0],
        q2=q[1],
        q3=q[2],
        theta1=math.ceil(2**FRAC / scales[0]),
        theta2=math.ceil(2**FRAC / scales[1]),
    )
    with open(OUT, "w") as f:
        f.write(net.to_json())
    with open(ckpt_path, "rb") as f:
        ckpt_digest = hashlib.sha256(f.read()).hexdigest()
    with open(OUT, "rb") as f:
        net_digest = hashlib.sha256(f.read()).hexdigest()
    print(
        json.dumps(
            {
                "checkpoint_sha256": ckpt_digest,
                "network_sha256": net_digest,
                "scales": scales,
                "theta": [net.theta1, net.theta2],
                "membrane_width": [net.membrane_width(1), net.membrane_width(2)],
            }
        )
    )


if __name__ == "__main__":
    main()
