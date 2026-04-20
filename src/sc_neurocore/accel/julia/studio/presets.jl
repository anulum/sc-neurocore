# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/presets

module PresetsAccel

using Statistics, LinearAlgebra

function list_presets()
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "description": p["description"],
            "suggested_view": p.get("suggested_view", "trace"),
        }
        for p in PRESETS
    ]
end

function get_preset(preset_id)
    return next((p for p in PRESETS if p["id"] == preset_id), nothing)
end

end # module PresetsAccel
