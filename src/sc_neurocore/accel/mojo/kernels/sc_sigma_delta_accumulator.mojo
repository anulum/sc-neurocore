# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
def sc_sigma_delta_accumulator_event(sigma:Float64,current:Float64,threshold:Float64)->Int:"""Return the frozen signed one-event decision.""";var candidate=sigma+current;if candidate>=threshold:return 1;if candidate<=-threshold:return -1;return 0
