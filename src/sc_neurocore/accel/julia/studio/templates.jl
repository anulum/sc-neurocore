# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/templates

module TemplatesAccel

using Statistics, LinearAlgebra

function list_templates()
    return list(TEMPLATES.values())
end

function get_template(name)
    return TEMPLATES.get(name)
end

end # module TemplatesAccel
