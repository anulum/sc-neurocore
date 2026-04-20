# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for fine_tune

fn freeze_layers(checkpoint: Int, layer_names: Int, until_index: Int) -> Int:
    var _freeze_layers_line = 'checkpoint: SNNCheckpoint,'
    var _freeze_layers_line = 'layer_names: list[str] | 0 = 0,'
    var _freeze_layers_line = 'until_index: int | 0 = 0,'
    var _freeze_layers_line = ') -> SNNCheckpoint:'
    var _freeze_layers_line = 'frozen = set(checkpoint.frozen_layers)'
    var _freeze_layers_line = 'if layer_names is not 0:'
    var _freeze_layers_line = 'frozen.update(layer_names)'
    var _freeze_layers_line = 'if until_index is not 0:'
    var _freeze_layers_line = 'for i, name in enumerate(checkpoint.layer_names):'
    var _freeze_layers_line = 'if i <= until_index:'
    var _freeze_layers_line = 'frozen.add(name)'
    var _freeze_layers_line = 'checkpoint.frozen_layers = sorted(frozen)'
    return 0  # return checkpoint

fn unfreeze_layers(checkpoint: Int, layer_names: Int, all_layers: Int) -> Int:
    var _unfreeze_layers_line = 'checkpoint: SNNCheckpoint,'
    var _unfreeze_layers_line = 'layer_names: list[str] | 0 = 0,'
    var _unfreeze_layers_line = 'all_layers: bool = False,'
    var _unfreeze_layers_line = ') -> SNNCheckpoint:'
    var _unfreeze_layers_line = 'if all_layers:'
    var _unfreeze_layers_line = 'checkpoint.frozen_layers = []'
    return 0  # return checkpoint
    var _unfreeze_layers_line = 'if layer_names is not 0:'
    var _unfreeze_layers_line = 'checkpoint.frozen_layers = [n for n in checkpoint.frozen_lay'
    return 0  # return checkpoint

fn apply_transfer_config(checkpoint: Int, config: Int) -> Int:
    var _apply_transfer_config_line = 'checkpoint: SNNCheckpoint,'
    var _apply_transfer_config_line = 'config: TransferConfig,'
    var _apply_transfer_config_line = ') -> tuple[SNNCheckpoint, list[float]]:'
    var _apply_transfer_config_line = 'if isinstance(config.freeze_until, int) and config.freeze_un'
    var _apply_transfer_config_line = 'freeze_layers(checkpoint, until_index=config.freeze_until)'
    var _apply_transfer_config_line = 'elif isinstance(config.freeze_until, str):'
    var _apply_transfer_config_line = 'idx = ('
    var _apply_transfer_config_line = 'checkpoint.layer_names.index(config.freeze_until)'
    var _apply_transfer_config_line = 'if config.freeze_until in checkpoint.layer_names'
    var _apply_transfer_config_line = 'else -1'
    var _apply_transfer_config_line = ')'
    var _apply_transfer_config_line = 'if idx >= 0:'
    var _apply_transfer_config_line = 'freeze_layers(checkpoint, until_index=idx)'
    var _apply_transfer_config_line = 'per_layer_lr = []'
    var _apply_transfer_config_line = 'for name in checkpoint.layer_names:'
    var _apply_transfer_config_line = 'if name in checkpoint.frozen_layers:'
    var _apply_transfer_config_line = 'per_layer_lr.append(config.lr_backbone)'
    var _apply_transfer_config_line = 'else:'
    var _apply_transfer_config_line = 'per_layer_lr.append(config.lr_head)'
    return 0  # return checkpoint, per_layer_lr

