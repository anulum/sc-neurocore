# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/training

module TrainingAccel

using Statistics, LinearAlgebra

mutable struct TrainingJobState
    id::Float64
    status::Float64
    _stop_event::Float64
end

function TrainingJobState()
    TrainingJobState(0.0, 0.0, 0.0)
end

function list_surrogates()
    return [{"name": s, "available": HAS_TORCH} for s in _SURROGATES]
end

function list_cell_types()
    return [{"name": c, "available": HAS_TORCH} for c in _CELL_TYPES]
end

function start(s::TrainingJobState)
    s.status = "running"
    s._thread = threading.Thread(target=s._run, daemon=true)
    s._thread.start()
end

function stop(s::TrainingJobState)
    s._stop_event.set()
end

function _emit(s::TrainingJobState, event_type, data, Any])
    payload = {"event": event_type, "data": data, "timestamp": time.time()}
    try
        s.metrics.put_nowait(payload)
    except queue.Full
        try
            s.metrics.get_nowait()
        except queue.Empty
            pass
        s.metrics.put_nowait(payload)
end

function _run(s::TrainingJobState)
    try
        s._train()
    except Exception as e
        s.error = str(e)
        s._emit("error", {"message": str(e)})
        s.status = "failed"
end

function _train(s::TrainingJobState)
    if ! HAS_TORCH
        raise RuntimeError("PyTorch ! installed. pip install sc-neurocore[research]")
    from sc_neurocore.training import (
        SpikingNet,
        SpikeMonitor,
        auto_device,
        model_info,
        spike_count_loss,
    )
    from sc_neurocore.training import surrogate as surr_mod
    cfg = s.config
    dataset = cfg.get("dataset", "synthetic")
    n_epochs = cfg.get("epochs", 10)
    batch_size = cfg.get("batch_size", 64)
    lr = cfg.get("lr", 1e-3)
    hidden = cfg.get("hidden", [128])
    n_timesteps = cfg.get("timesteps", 25)
    surrogate_name = cfg.get("surrogate", "atan_surrogate")
    learn_beta = cfg.get("learn_beta", false)
    learn_threshold = cfg.get("learn_threshold", false)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    surrogate_fn = getattr(surr_mod, surrogate_name, surr_mod.atan_surrogate)
    device = auto_device()
    if dataset == "mnist"
        train_loader, test_loader, n_inputs, n_outputs = _load_mnist(batch_size)
    else
        train_loader, test_loader, n_inputs, n_outputs = _make_synthetic(batch_size)
    n_hidden = hidden[0] if hidden else 128
    n_layers = length(hidden)
    model = SpikingNet(
        n_input=n_inputs,
        n_hidden=n_hidden,
        n_output=n_outputs,
        n_layers=n_layers,
        surrogate_fn=surrogate_fn,
        learn_beta=learn_beta,
        learn_threshold=learn_threshold,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    monitor = SpikeMonitor(model)
    info = model_info(model)
    s._emit(
        "config",
        {
            "job_id": s.id,
            "device": str(device),
            "model_info": info,
            "dataset": dataset,
            "n_epochs": n_epochs,
            "architecture": f"{n_inputs}→{'→'.join(str(n_hidden) for _ in 1:n_layers)}→{n_outputs}",
        },
    )
    for epoch in 1:n_epochs
        if s._stop_event.is_set()
            s.status = "stopped"
            s._emit("stopped", {"epoch": epoch})
            return
        model.train()
        monitor.reset()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader)
            if s._stop_event.is_set()
                break
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.shape[0], -1)
            data = data.unsqueeze(0).expand(n_timesteps, *data.shape)
            spike_counts, _ = model(data)
            loss = spike_count_loss(spike_counts, targets)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            if max_grad_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * targets.shape[0]
            correct += (spike_counts.argmax(dim=1) == targets).sum().item()
            total += targets.shape[0]
            if (batch_idx + 1) % 10 == 0
                s._emit(
                    "batch",
                    {
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "loss": loss.item(),
                        "accuracy": correct / total,
                    },
                )
        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        # Eval
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        with torch.no_grad()
            for data, targets in test_loader
                data, targets = data.to(device), targets.to(device)
                data = data.view(data.shape[0], -1)
                data = data.unsqueeze(0).expand(n_timesteps, *data.shape)
                spike_counts, _ = model(data)
                loss = spike_count_loss(spike_counts, targets)
                eval_loss += loss.item() * targets.shape[0]
                eval_correct += (spike_counts.argmax(dim=1) == targets).sum().item()
                eval_total += targets.shape[0]
        val_loss = eval_loss / max(eval_total, 1)
        val_acc = eval_correct / max(eval_total, 1)
        # Layer spike rates from monitor
        layer_rates = {}
        for name in monitor.layer_names
            raster = monitor.get(name)
            if raster is ! nothing
                layer_rates[name] = float(raster.float().mean().item())
        # Parameter snapshots (beta, threshold)
        param_snapshot = {}
        for pname, p in model.named_parameters()
            if "beta_logit" in pname
                param_snapshot[pname] = float(torch.sigmoid(p).mean().item())
            elseif "threshold_log" in pname
                param_snapshot[pname] = float(torch.exp(p).mean().item())
        s._emit(
            "epoch",
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 4),
                "layer_spike_rates": layer_rates,
                "param_snapshot": param_snapshot,
            },
        )
        monitor.reset()
    s.status = "completed"
    s.final_metrics = {
        "train_loss": round(train_loss, 6),
        "train_accuracy": round(train_acc, 4),
        "val_loss": round(val_loss, 6),
        "val_accuracy": round(val_acc, 4),
    }
    s._emit("completed", s.final_metrics)
    monitor.remove()
end

function start_training(config)
    job = TrainingJob(config)
    with _jobs_lock
        _jobs[job.id] = job
    job.start()
    return {"job_id": job.id, "status": "running"}
end

function stop_training(job_id)
    with _jobs_lock
        job = _jobs.get(job_id)
    if ! job
        return {"error": f"Job {job_id} ! found"}
    job.stop()
    return {"job_id": job_id, "status": "stopping"}
end

function get_training_status(job_id)
    with _jobs_lock
        job = _jobs.get(job_id)
    if ! job
        return {"error": f"Job {job_id} ! found"}
    return {
        "job_id": job.id,
        "status": job.status,
        "error": job.error,
        "final_metrics": job.final_metrics,
    }
end

function stream_metrics(job_id)
    with _jobs_lock
        job = _jobs.get(job_id)
    if ! job
        yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Job ! found'}})}\n\n"
        return
    while true
        try
            event = job.metrics.get(timeout=1.0)
            yield f"data: {json.dumps(event)}\n\n"
            if event["event"] in ("completed", "stopped", "error")
                break
        except queue.Empty
            if job.status in ("completed", "stopped", "failed")
                break
            yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
end

function list_jobs()
    with _jobs_lock
        return [{"job_id": j.id, "status": j.status, "config": j.config} for j in _jobs.values()]
end

end # module TrainingAccel
