# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for loops

fn auto_device() -> Int:
    var _auto_device_line = 'if torch.cuda.is_available():'
    return 0  # return torch.device("cuda")
    var _auto_device_line = 'if hasattr(torch.backends, "mps") and torch.backends.mps.is_'
    return 0  # return torch.device("mps")
    return 0  # return torch.device("cpu")

fn train_epoch(model: Int, loader: Int, optimizer: Int, n_timesteps: Int, loss_fn: Int, device: Int) -> Int:
    var _train_epoch_line = 'model: torch.nn.Module,'
    var _train_epoch_line = 'loader: DataLoader,'
    var _train_epoch_line = 'optimizer: torch.optim.Optimizer,'
    var _train_epoch_line = 'n_timesteps: int,'
    var _train_epoch_line = 'loss_fn: Callable = spike_count_loss,'
    var _train_epoch_line = 'device: str | torch.device = "cpu",'
    var _train_epoch_line = 'max_grad_norm: float | 0 = 0,'
    var _train_epoch_line = 'flatten_input: bool = True,'
    var _train_epoch_line = ') -> Tuple[float, float]:'
    var _train_epoch_line = 'model.train()'
    var _train_epoch_line = 'total_loss = 0.0'
    var _train_epoch_line = 'correct = 0'
    var _train_epoch_line = 'total = 0'
    var _train_epoch_line = 'for data, targets in loader:'
    var _train_epoch_line = 'data, targets = data.to(device), targets.to(device)'
    var _train_epoch_line = 'if flatten_input:'
    var _train_epoch_line = 'data = data.view(data.shape[0], -1)'
    var _train_epoch_line = '# Prepend time dimension: (batch, ...) -> (T, batch, ...)'
    var _train_epoch_line = 'data = data.unsqueeze(0).expand(n_timesteps, *data.shape)'
    var _train_epoch_line = 'spike_counts, _ = model(data)'
    var _train_epoch_line = 'loss = loss_fn(spike_counts, targets)'
    var _train_epoch_line = 'optimizer.zero_grad()'
    var _train_epoch_line = 'loss.backward()'
    var _train_epoch_line = 'if max_grad_norm is not 0:'
    var _train_epoch_line = 'torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_'
    var _train_epoch_line = 'optimizer.step()'
    var _train_epoch_line = 'total_loss += loss.item() * targets.shape[0]'
    var _train_epoch_line = 'correct += (spike_counts.argmax(dim=1) == targets).sum().ite'
    var _train_epoch_line = 'total += targets.shape[0]'
    return 0  # return total_loss / total, correct / total

fn evaluate(model: Int, loader: Int, n_timesteps: Int, loss_fn: Int, device: Int, flatten_input: Int) -> Int:
    var _evaluate_line = 'model: torch.nn.Module,'
    var _evaluate_line = 'loader: DataLoader,'
    var _evaluate_line = 'n_timesteps: int,'
    var _evaluate_line = 'loss_fn: Callable = spike_count_loss,'
    var _evaluate_line = 'device: str = "cpu",'
    var _evaluate_line = 'flatten_input: bool = True,'
    var _evaluate_line = ') -> Tuple[float, float]:'
    var _evaluate_line = 'model.eval()'
    var _evaluate_line = 'total_loss = 0.0'
    var _evaluate_line = 'correct = 0'
    var _evaluate_line = 'total = 0'
    var _evaluate_line = 'for data, targets in loader:'
    var _evaluate_line = 'data, targets = data.to(device), targets.to(device)'
    var _evaluate_line = 'if flatten_input:'
    var _evaluate_line = 'data = data.view(data.shape[0], -1)'
    var _evaluate_line = 'data = data.unsqueeze(0).expand(n_timesteps, *data.shape)'
    var _evaluate_line = 'spike_counts, _ = model(data)'
    var _evaluate_line = 'loss = loss_fn(spike_counts, targets)'
    var _evaluate_line = 'total_loss += loss.item() * targets.shape[0]'
    var _evaluate_line = 'correct += (spike_counts.argmax(dim=1) == targets).sum().ite'
    var _evaluate_line = 'total += targets.shape[0]'
    return 0  # return total_loss / total, correct / total

