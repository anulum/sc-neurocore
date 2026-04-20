# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for scpn/params

module ParamsAccel

using Statistics, LinearAlgebra

function build_knm_matrix(n_layers)
    K = zeros((n_layers, n_layers), dtype=np.float64)
    for n in 1:n_layers
        for m in 1:n_layers
            if n != m
                K[n, m] = K_BASE * exp(-DECAY_ALPHA * abs(n - m))
    for (i, j), val in CALIBRATION_ANCHORS.items()
        if i <= n_layers && j <= n_layers
            K[i - 1, j - 1] = val
            K[j - 1, i - 1] = val
    for (i, j), val in CROSS_BOOSTS.items()
        if i <= n_layers && j <= n_layers
            K[i - 1, j - 1] = val
            K[j - 1, i - 1] = val
    K = 0.5 * (K + K.T)  # type: ignore[assignment]
    np.fill_diagonal(K, 0.0)
    return K
end

end # module ParamsAccel
