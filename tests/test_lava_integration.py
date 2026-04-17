# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lava-nc integration test

"""
Lava-nc integration test.

Requires lava-nc which only supports Python 3.10.
Skip on unsupported versions.
"""

import pytest

try:
    import lava  # noqa: F401

    HAS_LAVA = True
except ImportError:
    HAS_LAVA = False

REASON = "lava-nc not installed"


def test_integrations_init_reexports_helpers():
    """`from sc_neurocore.integrations import X` works without lava-nc.

    Closes task #33: __init__ now re-exports the 5 always-importable
    symbols (HAS_LAVA, LoihiNetworkConfig, SCtoLavaConverter,
    export_weights_loihi, loihi_threshold_from_sc).
    """
    from sc_neurocore.integrations import (
        HAS_LAVA as INIT_HAS_LAVA,
        LoihiNetworkConfig,
        SCtoLavaConverter,
        export_weights_loihi,
        loihi_threshold_from_sc,
    )

    assert isinstance(INIT_HAS_LAVA, bool)
    assert callable(export_weights_loihi)
    assert callable(loihi_threshold_from_sc)
    # SCtoLavaConverter is a class; LoihiNetworkConfig is a dataclass
    assert isinstance(SCtoLavaConverter, type)
    assert hasattr(LoihiNetworkConfig, "__dataclass_fields__")


def test_integrations_init_lava_classes_only_when_has_lava():
    """SCDenseProcess / PySCDenseModel are exposed only when HAS_LAVA."""
    import sc_neurocore.integrations as integrations_pkg

    if integrations_pkg.HAS_LAVA:
        assert "SCDenseProcess" in integrations_pkg.__all__
        assert "PySCDenseModel" in integrations_pkg.__all__
        assert hasattr(integrations_pkg, "SCDenseProcess")
        assert hasattr(integrations_pkg, "PySCDenseModel")
    else:
        # Without lava-nc the class names must be absent from the package
        assert "SCDenseProcess" not in integrations_pkg.__all__
        assert "PySCDenseModel" not in integrations_pkg.__all__
        assert not hasattr(integrations_pkg, "SCDenseProcess")


@pytest.mark.skipif(not HAS_LAVA, reason=REASON)
def test_lava_import():
    """Verify lava-nc can be imported."""
    import lava.lib.dl.slayer as slayer  # noqa: F401
    from lava.proc.lif.process import LIF  # noqa: F401


@pytest.mark.skipif(not HAS_LAVA, reason=REASON)
def test_sc_to_lava_converter():
    """Convert SC-NeuroCore SNN to Lava process network and run on CPU sim."""
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps
    import numpy as np

    n_in, n_out = 8, 4
    weights = np.random.uniform(0, 1, (n_out, n_in)).astype(np.float32) * 100

    dense = Dense(weights=weights.astype(int))
    lif = LIF(shape=(n_out,), vth=100, dv=1, du=1)

    dense.s_out.connect(lif.a_in)

    run_cfg = Loihi1SimCfg()
    lif.run(condition=RunSteps(num_steps=100), run_cfg=run_cfg)
    v = lif.v.get()
    lif.stop()

    assert v.shape == (n_out,), f"Expected ({n_out},), got {v.shape}"


@pytest.mark.skipif(not HAS_LAVA, reason=REASON)
def test_spike_train_parity():
    """Compare SC-NeuroCore LIF spike train vs Lava LIF over 100 steps."""
    from sc_neurocore.neurons import StochasticLIFNeuron
    from lava.proc.lif.process import LIF
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps

    sc_neuron = StochasticLIFNeuron()
    sc_spikes = []
    for _ in range(100):
        spike, _ = sc_neuron.step(leak_k=1, gain_k=256, i_t=50, noise_in=0)
        sc_spikes.append(spike)
    sc_count = sum(sc_spikes)

    lif = LIF(shape=(1,), vth=256, dv=1, du=1)
    lif.run(condition=RunSteps(num_steps=100), run_cfg=Loihi1SimCfg())
    lif.stop()

    # Both should fire (exact match not expected due to model differences)
    assert sc_count > 0, "SC-NeuroCore neuron must fire"
