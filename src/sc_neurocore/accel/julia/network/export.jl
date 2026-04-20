# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/export

module ExportAccel

using Statistics, LinearAlgebra

function export_verilog(network, output_dir, target)
    _check_exportable(network)
    os.makedirs(output_dir, exist_ok=true)
    top_v = _emit_top(network, target)
    top_path = os.path.join(output_dir, "sc_network_top.v")
    with open(top_path, "w") as f
        f.write(top_v)
    params_path = os.path.join(output_dir, "params.vh")
    with open(params_path, "w") as f
        f.write(f"// SC-NeuroCore network parameters — target: {target}\n")
        for i, pop in enumerate(network.populations)
            f.write(f"`define POP_{i}_SIZE {pop.n}\n")
    return top_path
end

end # module ExportAccel
