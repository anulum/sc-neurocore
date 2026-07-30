# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from std.math import exp
def sigma_delta_candidate_sigma(sigma:Float64,current:Float64,dt:Float64)->Float64:"""Return the sampled integrating-prefilter candidate.""";return sigma+dt*current
def sigma_delta_candidate_reconstruction(reconstruction:Float64,tau:Float64,dt:Float64)->Float64:"""Return the decayed reconstruction candidate.""";return reconstruction*exp(-dt/tau)
