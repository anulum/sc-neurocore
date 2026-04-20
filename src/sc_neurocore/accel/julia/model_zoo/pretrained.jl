# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for model_zoo/pretrained

module PretrainedAccel

using Statistics, LinearAlgebra

function load_pretrained(name)
    if name ! in _REGISTRY
        raise ValueError(f"Unknown pretrained model '{name}'. Available: {sorted(_REGISTRY)}")
    builder, weight_file = _REGISTRY[name]
    path = _WEIGHTS_DIR / weight_file
    if ! path.exists()
        raise FileNotFoundError(f"Weight file ! found: {path}")
    net = builder()  # type: ignore[operator]
    data = np.load(path)
    projections = net.projections
    if name == "mnist"
        _apply_weights(projections[0], data["W0"])
        _apply_weights(projections[1], data["W1"])
    elseif name == "shd"
        _apply_weights(projections[0], data["W0"])
        _apply_weights(projections[1], data["W_rec"])
        _apply_weights(projections[2], data["W1"])
    elseif name == "dvs_gesture"
        _apply_weights(projections[0], data["W0"])
        _apply_weights(projections[1], data["W1"])
    return net
end

end # module PretrainedAccel
