# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kaggle MNIST all-fixes notebook export

# SC MNIST: all 4 fixes applied
# Fix 1: Remove calibration 3x overshoot
# Fix 2: Hook LIFCell output (not Linear) for calibration
# Fix 3: Per-trial MAC accumulation (correct bipolar structure)
# Fix 4: Per-row weight normalization (preserve inter-weight ratios)

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("=" * 70)
print("SC MNIST: All 4 Fixes")
print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
print(f"PyTorch: {torch.__version__}")
print("=" * 70)
sys.stdout.flush()


# ---- Surrogate + LIF ----
def atan_surr(x, a=2.0):
    return 0.5 + (1.0 / 3.14159265) * torch.atan(a * x)


class LIF(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.register_buffer("b", torch.tensor(beta))
        self.register_buffer("th", torch.tensor(1.0))

    def forward(self, cur, v):
        vn = self.b * v + cur
        sp = atan_surr(vn - self.th)
        vn = vn - sp.detach() * self.th
        return sp, vn


class SCAwareLin(nn.Module):
    def __init__(self, inf, outf, L=256):
        super().__init__()
        self.lin = nn.Linear(inf, outf)
        self.L = L
        with torch.no_grad():
            self.lin.weight.clamp_(-1, 1)

    def forward(self, x):
        w = self.lin.weight.clamp(-1, 1)
        if self.training:
            p = (w + 1) / 2
            w = w + torch.randn_like(w) * (p * (1 - p) / self.L).sqrt()
        return nn.functional.linear(x, w, self.lin.bias)


class SCANet(nn.Module):
    def __init__(self, ni, nh, no, nl=1, L=1024, beta=0.9):
        super().__init__()
        self.no = no
        sz = [ni] + [nh] * nl + [no]
        self.lins = nn.ModuleList(SCAwareLin(sz[i], sz[i + 1], L) for i in range(len(sz) - 1))
        self.lifs = nn.ModuleList(LIF(beta) for _ in range(len(sz) - 1))

    def forward(self, x):
        T, B, _ = x.shape
        d = x.device
        v = [torch.zeros(B, l.lin.out_features, device=d) for l in self.lins]
        ss = torch.zeros(B, self.no, device=d)
        for t in range(T):
            h = x[t]
            for i in range(len(self.lins)):
                h = self.lins[i](h)
                h, v[i] = self.lifs[i](h, v[i])
            ss += h
        return ss


# ---- FIX 3: Per-trial bipolar MAC ----
def bipolar_mac_fixed(inputs, weights, L, rng):
    """Correct bipolar MAC: accumulate per-trial, then average over L."""
    N, M = len(inputs), weights.shape[0]
    inp_p = np.clip((inputs + 1) / 2, 0, 1)
    w_p = np.clip((weights + 1) / 2, 0, 1)

    # Generate all bitstreams at once
    inp_bits = (rng.random((N, L)) < inp_p[:, None]).astype(np.uint8)

    outputs = np.zeros(M)
    for j in range(M):
        w_bits = (rng.random((N, L)) < w_p[j, :, None]).astype(np.uint8)
        # Per-trial: sum XNOR across all N inputs, decode to bipolar
        xnor = (inp_bits == w_bits).astype(np.float32)  # (N, L)
        # For each trial t: dot_t = sum_i(2*XNOR[i,t] - 1) = 2*sum_i(XNOR[i,t]) - N
        trial_dots = 2.0 * xnor.sum(axis=0) - N  # (L,) bipolar dot products per trial
        # Average over L trials for noise reduction
        outputs[j] = trial_dots.mean()

    return outputs


# ---- FIX 4: Per-row weight normalization ----
def extract_layers(model):
    layers = []
    for lin_mod in model.lins:
        w = lin_mod.lin.weight.detach().clamp(-1, 1).cpu().numpy()
        b = lin_mod.lin.bias.detach().cpu().numpy() if lin_mod.lin.bias is not None else None
        # Per-row normalization: each output neuron normalized independently
        row_max = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-8)
        w_norm = w / row_max
        layers.append({"w": w_norm, "b": b, "row_scale": row_max.flatten()})
    return layers


# ---- FIX 2: Hook LIFCell, not Linear ----
def calibrate_fixed(model, test_data, T=25):
    model.eval()
    cal = {}
    acts = {}
    hooks = []
    # Hook the LIFCell outputs, not the Linear outputs
    for i, lif in enumerate(model.lifs):
        acts[i] = []

        def mk(idx):
            def hook(m, inp, out):
                # LIFCell returns (spike, v_next) — capture spike
                spike = out[0] if isinstance(out, tuple) else out
                acts[idx].append(spike.detach().cpu())

            return hook

        hooks.append(lif.register_forward_hook(mk(i)))

    with torch.no_grad():
        for idx in range(min(200, len(test_data))):
            img, _ = test_data[idx]
            x = img.view(1, -1).unsqueeze(0).expand(T, 1, 784)
            model(x)
    for h in hooks:
        h.remove()
    for i, a in acts.items():
        if a:
            v = torch.cat([x.flatten() for x in a]).numpy()
            cal[i] = {"mean": float(v.mean()), "std": max(float(v.std()), 1e-8)}
    return cal


# ---- SC inference with all fixes ----
def sc_infer_fixed(img_flat, layers, L, rng, cal):
    x = img_flat.copy().astype(np.float64)
    x = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-8) - 1.0

    for li, lay in enumerate(layers):
        w, bias, row_scale = lay["w"], lay["b"], lay["row_scale"]
        no, ni = w.shape
        if len(x) < ni:
            xp = np.zeros(ni)
            xp[: len(x)] = x
            x = xp
        elif len(x) > ni:
            x = x[:ni]

        # FIX 3: per-trial MAC
        out = bipolar_mac_fixed(np.clip(x, -1, 1), w, L, rng)

        # Undo per-row normalization (FIX 4)
        out = out * row_scale

        # Add bias
        if bias is not None:
            out += bias

        # FIX 1 + FIX 2: calibration without 3x, using LIF output stats
        if li in cal:
            c = cal[li]
            # Normalize to zero-mean, unit-variance (no 3x overshoot)
            out = (out - c["mean"]) / c["std"]
            # Squash to [-1, 1] with tanh-like soft clipping
            out = np.tanh(out * 0.5)

        # ReLU for hidden layers
        if li < len(layers) - 1:
            out = np.maximum(out, 0.0)

        x = np.clip(out, -1.0, 1.0)
    return x


def sc_eval(model, test_data, L, n_samples=300):
    layers = extract_layers(model)
    cal = calibrate_fixed(model, test_data)
    print(f"    Calibration: {cal}")
    sys.stdout.flush()
    rng = np.random.default_rng(42)
    correct, n = 0, min(n_samples, len(test_data))
    for i in range(n):
        img, label = test_data[i]
        out = sc_infer_fixed(img.numpy().flatten(), layers, L, rng, cal)
        if int(np.argmax(out)) == label:
            correct += 1
    return correct / n


# ---- Training ----
def train_model(model, tr_ld, te_ld, epochs, T, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        cr, tot = 0, 0
        for d, tgt in tr_ld:
            x = d.view(d.size(0), -1).unsqueeze(0).expand(T, d.size(0), 784)
            sp = model(x)
            loss = lf(sp, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cr += (sp.argmax(1) == tgt).sum().item()
            tot += tgt.size(0)
        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for d, tgt in te_ld:
                x = d.view(d.size(0), -1).unsqueeze(0).expand(T, d.size(0), 784)
                tc += (model(x).argmax(1) == tgt).sum().item()
                tt += tgt.size(0)
        acc = tc / tt
        best = max(best, acc)
        print(
            f"  [{tag}] Ep {ep + 1}/{epochs}: train={cr / tot:.3f} test={acc:.3f} ({time.time() - t0:.1f}s)"
        )
        sys.stdout.flush()
    return best


# ---- Main ----
def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("/kaggle/working/data", train=True, download=True, transform=tf)
    te = datasets.MNIST("/kaggle/working/data", train=False, download=True, transform=tf)
    tr_ld = DataLoader(tr, batch_size=128, shuffle=True)
    te_ld = DataLoader(te, batch_size=128, shuffle=False)
    T, t0 = 25, time.time()

    print("\n--- SC-Aware L=1024 (best from previous) ---")
    model = SCANet(784, 128, 10, nl=1, L=1024, beta=0.9)
    float_acc = train_model(model, tr_ld, te_ld, 10, T, "SC1024")
    print(f"  Float: {float_acc:.4f}")

    print("\n--- SC Inference (all 4 fixes) ---")
    results = {}
    for L in [128, 256, 512, 1024, 2048]:
        t1 = time.time()
        acc = sc_eval(model, te, L, n_samples=300)
        el = time.time() - t1
        results[L] = {"accuracy": round(acc, 4), "time_s": round(el, 1)}
        print(f"  L={L:5d}: {acc:.2%} ({el:.1f}s)")
        sys.stdout.flush()

    total = time.time() - t0
    print("\n" + "=" * 70)
    print("RESULTS (all 4 fixes)")
    print("=" * 70)
    print(f"  Float: {float_acc:.2%}")
    for L, r in sorted(results.items()):
        print(f"  L={L:5d}: {r['accuracy']:.2%} (drop: {float_acc - r['accuracy']:.2%})")
    print(f"  Time: {total:.0f}s")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixes": ["calibration_no_3x", "hook_lif_not_linear", "per_trial_mac", "per_row_norm"],
        "float_accuracy": round(float_acc, 4),
        "sc_results": results,
        "total_s": round(total, 1),
    }
    p = Path("/kaggle/working/sc_all_fixes_results.json")
    if not p.parent.exists():
        p = Path("sc_all_fixes_results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {p}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
