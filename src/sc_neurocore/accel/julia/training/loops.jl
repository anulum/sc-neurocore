# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for training/loops

module LoopsAccel

using Statistics, LinearAlgebra

function auto_device()
    if torch.cuda.is_available()
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") && torch.backends.mps.is_available()
        return torch.device("mps")
    return torch.device("cpu")
end

function train_epoch(model, loader, optimizer, n_timesteps, loss_fn, device, max_grad_norm, flatten_input)
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_timesteps: int,
    loss_fn: Callable = spike_count_loss,
    device: str | torch.device = "cpu",
    max_grad_norm: float | nothing = nothing,
    flatten_input: bool = true,
    ) -> Tuple[float, float]
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, targets in loader
        data, targets = data.to(device), targets.to(device)
        if flatten_input
            data = data.view(data.shape[0], -1)
        # Prepend time dimension: (batch, ...) -> (T, batch, ...)
        data = data.unsqueeze(0).expand(n_timesteps, *data.shape)
        spike_counts, _ = model(data)
        loss = loss_fn(spike_counts, targets)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is ! nothing
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * targets.shape[0]
        correct += (spike_counts.argmax(dim=1) == targets).sum().item()
        total += targets.shape[0]
    return total_loss / total, correct / total
end

function evaluate(model, loader, n_timesteps, loss_fn, device, flatten_input)
    model: torch.nn.Module,
    loader: DataLoader,
    n_timesteps: int,
    loss_fn: Callable = spike_count_loss,
    device: str = "cpu",
    flatten_input: bool = true,
    ) -> Tuple[float, float]
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, targets in loader
        data, targets = data.to(device), targets.to(device)
        if flatten_input
            data = data.view(data.shape[0], -1)
        data = data.unsqueeze(0).expand(n_timesteps, *data.shape)
        spike_counts, _ = model(data)
        loss = loss_fn(spike_counts, targets)
        total_loss += loss.item() * targets.shape[0]
        correct += (spike_counts.argmax(dim=1) == targets).sum().item()
        total += targets.shape[0]
    return total_loss / total, correct / total
end

end # module LoopsAccel
