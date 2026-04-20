# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for neurons/base

module BaseAccel

using Statistics, LinearAlgebra

function step(input_current)
    error("not implemented")
end

function reset_state()
    error("not implemented")
end

function get_state()
    error("not implemented")
end

end # module BaseAccel
