# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kaggle spike-threshold notebook export

# SC MNIST: Fix 5 — LIF spike thresholding between SC layers
# The float network uses LIF(spike/no-spike) between layers.
# SC inference was passing continuous [-1,1] values — wrong representation.
# Fix: apply spike-rate encoding between SC layers to match float dynamics.

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
print("SC MNIST: Fix 5 — LIF spike thresholding")
print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
print(f"PyTorch: {torch.__version__}")
print("=" * 70)
sys.stdout.flush()


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
    def __init__(self, inf, outf, L=1024):
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


# ---- Bipolar MAC (per-trial, from all-fixes) ----
def bipolar_mac(inputs, weights, L, rng):
    N, M = len(inputs), weights.shape[0]
    inp_p = np.clip((inputs + 1) / 2, 0, 1)
    w_p = np.clip((weights + 1) / 2, 0, 1)
    inp_bits = (rng.random((N, L)) < inp_p[:, None]).astype(np.uint8)
    outputs = np.zeros(M)
    for j in range(M):
        w_bits = (rng.random((N, L)) < w_p[j, :, None]).astype(np.uint8)
        xnor = (inp_bits == w_bits).astype(np.float32)
        trial_dots = 2.0 * xnor.sum(axis=0) - N
        outputs[j] = trial_dots.mean()
    return outputs


# ---- SC inference with LIF dynamics between layers ----
def sc_infer_with_lif(img_flat, layers, L, rng, beta=0.9, T_lif=25):
    """SC inference that matches float network's LIF dynamics.

    For each hidden layer:
    1. SC bipolar MAC (weighted sum)
    2. Run LIF neuron for T_lif steps with MAC output as current
    3. Output = spike rate (proportion of timesteps that spiked)
    4. Encode spike rate as bipolar value for next layer

    For output layer:
    - SC bipolar MAC only, no LIF (use raw weighted sum for argmax)
    """
    # Normalise input to [0, 1] (matching MNIST pixel range after transforms)
    x = img_flat.copy().astype(np.float64)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 1e-8:
        x = (x - x_min) / (x_max - x_min)
    # Convert to bipolar [-1, 1] for SC
    x_bp = 2.0 * x - 1.0

    for li, lay in enumerate(layers):
        w, bias, row_scale = lay["w"], lay["b"], lay["row_scale"]
        no, ni = w.shape

        # Pad/truncate input
        if len(x_bp) < ni:
            xp = np.zeros(ni)
            xp[: len(x_bp)] = x_bp
            x_bp = xp
        elif len(x_bp) > ni:
            x_bp = x_bp[:ni]

        # SC bipolar MAC
        mac_out = bipolar_mac(np.clip(x_bp, -1, 1), w, L, rng)

        # Undo per-row normalization to get float-scale currents
        mac_out = mac_out * row_scale

        # Add bias
        if bias is not None:
            mac_out += bias

        if li < len(layers) - 1:
            # HIDDEN LAYER: run LIF to get spike rates
            # This matches what the float network does between layers
            spike_counts = np.zeros(no)
            v = np.zeros(no)
            for t in range(T_lif):
                v = beta * v + mac_out
                spikes = (v >= 1.0).astype(np.float64)
                v = v - spikes * 1.0
                spike_counts += spikes

            # Spike rate in [0, 1]
            spike_rate = spike_counts / T_lif
            # Convert to bipolar for next SC layer
            x_bp = 2.0 * spike_rate - 1.0
        else:
            # OUTPUT LAYER: use raw MAC output for classification
            x_bp = mac_out

    return x_bp  # Raw output layer values for argmax


def extract_layers(model):
    layers = []
    for lin_mod in model.lins:
        w = lin_mod.lin.weight.detach().clamp(-1, 1).cpu().numpy()
        b = lin_mod.lin.bias.detach().cpu().numpy() if lin_mod.lin.bias is not None else None
        row_max = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-8)
        layers.append({"w": w / row_max, "b": b, "row_scale": row_max.flatten()})
    return layers


def sc_eval(model, test_data, L, n_samples=300, T_lif=25):
    layers = extract_layers(model)
    rng = np.random.default_rng(42)
    correct, n = 0, min(n_samples, len(test_data))
    for i in range(n):
        img, label = test_data[i]
        out = sc_infer_with_lif(img.numpy().flatten(), layers, L, rng, T_lif=T_lif)
        if int(np.argmax(out)) == label:
            correct += 1
    return correct / n


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


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("/kaggle/working/data", train=True, download=True, transform=tf)
    te = datasets.MNIST("/kaggle/working/data", train=False, download=True, transform=tf)
    tr_ld = DataLoader(tr, batch_size=128, shuffle=True)
    te_ld = DataLoader(te, batch_size=128, shuffle=False)
    T, t0 = 25, time.time()

    print("\n--- Train SC-Aware L=1024 ---")
    model = SCANet(784, 128, 10, nl=1, L=1024, beta=0.9)
    float_acc = train_model(model, tr_ld, te_ld, 10, T, "SC1024")
    print(f"  Float: {float_acc:.4f}")

    print("\n--- SC Inference WITH LIF dynamics (Fix 5) ---")
    results = {}
    for L in [128, 256, 512, 1024, 2048]:
        t1 = time.time()
        acc = sc_eval(model, te, L, n_samples=300, T_lif=T)
        el = time.time() - t1
        results[L] = {"accuracy": round(acc, 4), "time_s": round(el, 1)}
        print(f"  L={L:5d}: {acc:.2%} ({el:.1f}s)")
        sys.stdout.flush()

    print("\n--- Compare: SC inference WITHOUT LIF (previous method) ---")
    prev_results = {}
    # Quick comparison at L=1024 only using old method (no LIF between layers)
    layers_raw = extract_layers(model)
    rng = np.random.default_rng(42)
    correct, n = 0, 300
    for i in range(n):
        img, label = te[i]
        x = img.numpy().flatten().astype(np.float64)
        x = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-8) - 1.0
        for li, lay in enumerate(layers_raw):
            w, bias, rs = lay["w"], lay["b"], lay["row_scale"]
            no, ni = w.shape
            if len(x) < ni:
                xp = np.zeros(ni)
                xp[: len(x)] = x
                x = xp
            elif len(x) > ni:
                x = x[:ni]
            out = bipolar_mac(np.clip(x, -1, 1), w, 1024, rng)
            out = out * rs
            if bias is not None:
                out += bias
            if li < len(layers_raw) - 1:
                out = np.maximum(out, 0.0)
                mx = max(abs(out).max(), 1e-8)
                x = np.clip(out / mx, -1, 1)
            else:
                x = out
        if int(np.argmax(x)) == label:
            correct += 1
    no_lif_acc = correct / n
    print(f"  L=1024 without LIF: {no_lif_acc:.2%}")

    total = time.time() - t0
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Float: {float_acc:.2%}")
    print("  SC with LIF dynamics:")
    for L, r in sorted(results.items()):
        print(f"    L={L:5d}: {r['accuracy']:.2%}")
    print(f"  SC without LIF (L=1024): {no_lif_acc:.2%}")
    print(f"  Time: {total:.0f}s")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fix": "lif_spike_threshold_between_layers",
        "float_accuracy": round(float_acc, 4),
        "sc_with_lif": results,
        "sc_without_lif_L1024": round(no_lif_acc, 4),
        "total_s": round(total, 1),
    }
    p = Path("/kaggle/working/sc_spike_threshold_results.json")
    if not p.parent.exists():
        p = Path("sc_spike_threshold_results.json")
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
