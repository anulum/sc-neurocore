# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for pretrained

fn _dense_to_csr(dense: Int) -> Int:
    var __dense_to_csr_line = 'sp = sparse.csr_matrix(dense)'
    return 0  # return (
    var __dense_to_csr_line = 'sp.indptr.astype(int32),'
    var __dense_to_csr_line = 'sp.indices.astype(int32),'
    var __dense_to_csr_line = 'sp.data.astype(float64),'
    var __dense_to_csr_line = ')'

fn _apply_weights(proj: Int, dense: Int) -> Int:
    var __apply_weights_line = 'indptr, indices, data = _dense_to_csr(dense)'
    var __apply_weights_line = 'proj.indptr = indptr  # type: ignore[attr-defined]'
    var __apply_weights_line = 'proj.indices = indices  # type: ignore[attr-defined]'
    var __apply_weights_line = 'proj.data = data'
    return 0

fn load_pretrained(name: Int) -> Int:
    var _load_pretrained_line = 'if name not in _REGISTRY:'
    var _load_pretrained_line = 'raise ValueError(f"Unknown pretrained model \'{name}\'. Availa'
    var _load_pretrained_line = 'builder, weight_file = _REGISTRY[name]'
    var _load_pretrained_line = 'path = _WEIGHTS_DIR / weight_file'
    var _load_pretrained_line = 'if not path.exists():'
    var _load_pretrained_line = 'raise FileNotFoundError(f"Weight file not found: {path}")'
    var _load_pretrained_line = 'net = builder()  # type: ignore[operator]'
    var _load_pretrained_line = 'data = load(path)'
    var _load_pretrained_line = 'projections = net.projections'
    var _load_pretrained_line = 'if name == "mnist":'
    var _load_pretrained_line = '_apply_weights(projections[0], data["W0"])'
    var _load_pretrained_line = '_apply_weights(projections[1], data["W1"])'
    var _load_pretrained_line = 'elif name == "shd":'
    var _load_pretrained_line = '_apply_weights(projections[0], data["W0"])'
    var _load_pretrained_line = '_apply_weights(projections[1], data["W_rec"])'
    var _load_pretrained_line = '_apply_weights(projections[2], data["W1"])'
    var _load_pretrained_line = 'elif name == "dvs_gesture":'
    var _load_pretrained_line = '_apply_weights(projections[0], data["W0"])'
    var _load_pretrained_line = '_apply_weights(projections[1], data["W1"])'
    return 0  # return net

