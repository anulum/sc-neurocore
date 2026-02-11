#!/usr/bin/env python3
# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact us: www.anulum.li  protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AFFERO GENERAL PUBLIC LICENSE v3
# Commercial Licensing: Available

"""HDC Symbolic Query Demo — "Capital of France?"

Demonstrates the Hyper-Dimensional Computing (HDC/VSA) kernel
backed by SC-NeuroCore's SIMD-accelerated BitStreamTensor.

Encoding scheme (Multiply-Add-Permute):
  record = sum_i (role_i * value_i)

Query:
  answer_hat = memory * role_query  (unbind)
  → find closest atom in codebook
"""

from sc_neurocore_engine import HDCVector

DIM = 10_000  # 10,000-bit hypervectors


def main():
    print(f"SC-NeuroCore HDC Symbolic Query Demo (D={DIM})")
    print("=" * 55)

    # ── Step 1: Create atomic symbols ─────────────────────────────

    # Role vectors
    role_country = HDCVector(DIM, seed=1000)
    role_capital = HDCVector(DIM, seed=1001)

    # Country atoms
    france = HDCVector(DIM, seed=2000)
    germany = HDCVector(DIM, seed=2001)
    japan = HDCVector(DIM, seed=2002)

    # Capital atoms
    paris = HDCVector(DIM, seed=3000)
    berlin = HDCVector(DIM, seed=3001)
    tokyo = HDCVector(DIM, seed=3002)

    atoms = {
        "France": france, "Germany": germany, "Japan": japan,
        "Paris": paris, "Berlin": berlin, "Tokyo": tokyo,
    }

    print("\nAtomic symbols created.")
    print(f"  Pairwise similarity (should be ~0.50):")
    print(f"    France vs Germany: {france.similarity(germany):.3f}")
    print(f"    Paris vs Berlin:   {paris.similarity(berlin):.3f}")

    # ── Step 2: Encode records ────────────────────────────────────

    # record_france = (role_country * France) + (role_capital * Paris)
    rec_france = HDCVector.bundle([
        role_country * france,
        role_capital * paris,
    ])
    rec_germany = HDCVector.bundle([
        role_country * germany,
        role_capital * berlin,
    ])
    rec_japan = HDCVector.bundle([
        role_country * japan,
        role_capital * tokyo,
    ])

    # Bundle all records into memory
    memory = HDCVector.bundle([rec_france, rec_germany, rec_japan])

    print("\nRecords encoded and bundled into memory.")

    # ── Step 3: Query "Capital of France?" ────────────────────────

    # Unbind: probe = memory * role_capital
    # Then check similarity to each country
    probe = memory * role_capital

    print("\nQuery: 'Which countries are in memory?' (unbind role_capital)")
    for name, atom in [("France", france), ("Germany", germany),
                       ("Japan", japan), ("Paris", paris)]:
        sim = probe.similarity(atom)
        print(f"  sim(probe, {name:>8s}) = {sim:.4f}")

    # ── Step 4: Query specific record ─────────────────────────────

    # To query "Capital of France?":
    # 1. Unbind France from memory: hat = memory * France
    # 2. Then unbind role_capital: answer = hat * role_capital
    # The result should be closest to Paris

    hat = memory * france
    answer = hat * role_capital

    print("\nQuery: 'Capital of France?' (memory * France * role_capital)")
    best_name, best_sim = "", -1.0
    for name, atom in atoms.items():
        sim = answer.similarity(atom)
        marker = ""
        if sim > best_sim:
            best_sim = sim
            best_name = name
        print(f"  sim(answer, {name:>8s}) = {sim:.4f}{marker}")

    print(f"\n  Best match: {best_name} (similarity {best_sim:.4f})")

    # ── Step 5: Verify bind-inverse property ──────────────────────

    a = HDCVector(DIM, seed=42)
    b = HDCVector(DIM, seed=99)
    recovered = (a * b) * b
    sim = recovered.similarity(a)
    print(f"\nBind-inverse property: sim(a, (a*b)*b) = {sim:.4f} (should be ~1.0)")

    # ── Step 6: Permute for sequence encoding ─────────────────────

    v = HDCVector(DIM, seed=42)
    p1 = v.permute(1)
    p2 = v.permute(2)
    print(f"\nPermute orthogonality:")
    print(f"  sim(v, permute(v,1)) = {v.similarity(p1):.4f}")
    print(f"  sim(v, permute(v,2)) = {v.similarity(p2):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
