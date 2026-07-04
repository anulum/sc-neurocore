# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kaggle SC-aware MNIST notebook export

# SC-Aware MNIST: fully self-contained, zero pip installs
# Compares standard vs SC-aware SNN training + bipolar SC inference

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
print("SC-Aware MNIST (standalone, no pip install)")
print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
print(f"PyTorch: {torch.__version__}")
print("=" * 70)
sys.stdout.flush()


# ---- Surrogate gradient ----
def atan_surrogate(x, alpha=2.0):
    return 0.5 + (1.0 / 3.14159265) * torch.atan(alpha * x)


# ---- LIF Cell ----
class LIFCell(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.register_buffer("beta_val", torch.tensor(beta))
        self.register_buffer("th_val", torch.tensor(1.0))

    def forward(self, current, v):
        v_next = self.beta_val * v + current
        spike = atan_surrogate(v_next - self.th_val)
        v_next = v_next - spike.detach() * self.th_val
        return spike, v_next


# ---- Standard SNN ----
class StandardNet(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_layers=1, beta=0.9):
        super().__init__()
        self.n_out = n_out
        sizes = [n_in] + [n_hid] * n_layers + [n_out]
        self.lins = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))
        self.lifs = nn.ModuleList(LIFCell(beta) for _ in range(len(sizes) - 1))

    def forward(self, x):
        T, B, _ = x.shape
        dev = x.device
        v = [torch.zeros(B, l.out_features, device=dev) for l in self.lins]
        s_sum = torch.zeros(B, self.n_out, device=dev)
        for t in range(T):
            h = x[t]
            for i in range(len(self.lins)):
                h = self.lins[i](h)
                h, v[i] = self.lifs[i](h, v[i])
            s_sum += h
        return s_sum


# ---- SC-Aware SNN (noise injection during training) ----
class SCAwareLin(nn.Module):
    def __init__(self, in_f, out_f, L=256):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
        self.L = L
        with torch.no_grad():
            self.lin.weight.clamp_(-1.0, 1.0)

    def forward(self, x):
        w = self.lin.weight.clamp(-1.0, 1.0)
        if self.training:
            p = (w + 1.0) / 2.0
            noise = torch.randn_like(w) * (p * (1.0 - p) / self.L).sqrt()
            w = w + noise
        return nn.functional.linear(x, w, self.lin.bias)


class SCAwareNet(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_layers=1, L=256, beta=0.9):
        super().__init__()
        self.n_out = n_out
        sizes = [n_in] + [n_hid] * n_layers + [n_out]
        self.lins = nn.ModuleList(
            SCAwareLin(sizes[i], sizes[i + 1], L=L) for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(LIFCell(beta) for _ in range(len(sizes) - 1))

    def forward(self, x):
        T, B, _ = x.shape
        dev = x.device
        v = [torch.zeros(B, l.lin.out_features, device=dev) for l in self.lins]
        s_sum = torch.zeros(B, self.n_out, device=dev)
        for t in range(T):
            h = x[t]
            for i in range(len(self.lins)):
                h = self.lins[i](h)
                h, v[i] = self.lifs[i](h, v[i])
            s_sum += h
        return s_sum


# ---- Bipolar SC Inference ----
def bipolar_sc_infer(image_flat, layers, L, rng, calibration):
    x = image_flat.copy().astype(np.float64)
    x = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-8) - 1.0

    for li, lay in enumerate(layers):
        w, bias, scale = lay["w"], lay["b"], lay["s"]
        n_out, n_in = w.shape
        if len(x) < n_in:
            xp = np.zeros(n_in)
            xp[: len(x)] = x
            x = xp
        elif len(x) > n_in:
            x = x[:n_in]

        x_bp = np.clip(x, -1, 1)
        inp_p = np.clip((x_bp + 1) / 2, 0, 1)
        inp_bits = (rng.random((n_in, L)) < inp_p[:, None]).astype(np.uint8)

        out = np.zeros(n_out)
        for j in range(n_out):
            wp = np.clip((w[j] + 1) / 2, 0, 1)
            wb = (rng.random((n_in, L)) < wp[:, None]).astype(np.uint8)
            per_in = 2.0 * (inp_bits == wb).astype(np.float32).mean(axis=1) - 1.0
            out[j] = per_in.sum()

        out = out * scale
        if bias is not None:
            out += bias

        if li in calibration and calibration[li]["std"] > 1e-8:
            out = (out - calibration[li]["mean"]) / (3.0 * calibration[li]["std"])

        if li < len(layers) - 1:
            out = np.maximum(out, 0.0)
        x = np.clip(out, -1.0, 1.0)
    return x


def extract_bipolar_layers(model):
    layers = []
    lins = model.lins if hasattr(model, "lins") else model.lins
    for lin_mod in lins:
        w_tensor = lin_mod.weight if hasattr(lin_mod, "weight") else lin_mod.lin.weight
        b_tensor = lin_mod.bias if hasattr(lin_mod, "bias") else lin_mod.lin.bias
        w = w_tensor.detach().clamp(-1, 1).cpu().numpy()
        mx = max(np.abs(w).max(), 1e-8)
        b = b_tensor.detach().cpu().numpy() if b_tensor is not None else None
        layers.append({"w": w / mx, "b": b, "s": float(mx)})
    return layers


def calibrate(model, test_data, T=25):
    model.eval()
    cal = {}
    acts = {}
    hooks = []
    lins = model.lins
    for i, lin in enumerate(lins):
        acts[i] = []

        def mk(idx):
            def hook(m, inp, out):
                acts[idx].append(out.detach().cpu())

            return hook

        hooks.append(lin.register_forward_hook(mk(i)))

    with torch.no_grad():
        for idx in range(min(100, len(test_data))):
            img, _ = test_data[idx]
            x = img.view(1, -1).unsqueeze(0).expand(T, 1, 784)
            model(x)
    for h in hooks:
        h.remove()
    for i, a in acts.items():
        if a:
            v = torch.cat(a).numpy()
            cal[i] = {"mean": float(v.mean()), "std": float(v.std())}
    return cal


def sc_eval(model, test_data, L, n_samples=300):
    layers = extract_bipolar_layers(model)
    cal = calibrate(model, test_data)
    rng = np.random.default_rng(42)
    correct = 0
    n = min(n_samples, len(test_data))
    for i in range(n):
        img, label = test_data[i]
        out = bipolar_sc_infer(img.numpy().flatten(), layers, L, rng, cal)
        if int(np.argmax(out)) == label:
            correct += 1
    return correct / n


# ---- Training loop ----
def train(model, train_ld, test_ld, epochs, T, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        cr, tot = 0, 0
        for data, tgt in train_ld:
            x = data.view(data.size(0), -1).unsqueeze(0).expand(T, data.size(0), 784)
            sp = model(x)
            loss = loss_fn(sp, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cr += (sp.argmax(1) == tgt).sum().item()
            tot += tgt.size(0)
        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for data, tgt in test_ld:
                x = data.view(data.size(0), -1).unsqueeze(0).expand(T, data.size(0), 784)
                tc += (model(x).argmax(1) == tgt).sum().item()
                tt += tgt.size(0)
        tacc = tc / tt
        best = max(best, tacc)
        print(
            f"  [{tag}] Ep {ep + 1}/{epochs}: train={cr / tot:.3f} test={tacc:.3f} ({time.time() - t0:.1f}s)"
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
    T = 25
    t0 = time.time()
    results = {}

    print("\n--- Standard SNN ---")
    m_std = StandardNet(784, 128, 10)
    std_acc = train(m_std, tr_ld, te_ld, 10, T, "Std")
    print(f"  Float: {std_acc:.4f}")

    print("\n--- SC-Aware L=256 ---")
    m_256 = SCAwareNet(784, 128, 10, L=256)
    a256 = train(m_256, tr_ld, te_ld, 10, T, "SC256")
    print(f"  Float: {a256:.4f}")

    print("\n--- SC-Aware L=1024 ---")
    m_1k = SCAwareNet(784, 128, 10, L=1024)
    a1k = train(m_1k, tr_ld, te_ld, 10, T, "SC1k")
    print(f"  Float: {a1k:.4f}")

    print("\n--- Bipolar SC Inference ---")
    for name, model in [("standard", m_std), ("sc_L256", m_256), ("sc_L1024", m_1k)]:
        mr = {}
        for L in [256, 512, 1024]:
            t1 = time.time()
            acc = sc_eval(model, te, L, n_samples=200)
            el = time.time() - t1
            mr[L] = {"accuracy": round(acc, 4), "time_s": round(el, 1)}
            print(f"  [{name}] L={L}: {acc:.2%} ({el:.1f}s)")
            sys.stdout.flush()
        results[name] = mr

    total = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Float: std={std_acc:.2%} sc256={a256:.2%} sc1024={a1k:.2%}")
    for nm in results:
        print(f"  SC L=1024 [{nm}]: {results[nm][1024]['accuracy']:.2%}")
    print(f"  Time: {total:.0f}s")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "float": {"standard": round(std_acc, 4), "sc256": round(a256, 4), "sc1024": round(a1k, 4)},
        "sc_inference": results,
        "total_s": round(total, 1),
    }
    p = Path("/kaggle/working/sc_aware_results.json")
    if not p.parent.exists():
        p = Path("sc_aware_results.json")
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
