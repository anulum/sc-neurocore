#!/usr/bin/env python3
# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact us: www.anulum.li  protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AFFERO GENERAL PUBLIC LICENSE v3
# Commercial Licensing: Available

"""Fault-Tolerant Binary Streams Demo — Safety-Critical Logic.

Demonstrates Module C of the v3.7 Polymorphic Engine:
Boolean logic with stochastic redundancy for error suppression.

Principle: encode a Boolean value as a redundant bitstream (all-ones or
all-zeros), then inject random bit flips to simulate radiation/noise,
and show that majority-vote decoding recovers the correct value.

Uses SC-NeuroCore's SIMD-accelerated BitStreamTensor for all operations.
"""

from sc_neurocore_engine import BitStreamTensor


def make_constant(value: bool, length: int = 1024) -> BitStreamTensor:
    """Create a constant bitstream: all-ones (True) or all-zeros (False)."""
    if value:
        words = length // 64
        tail_bits = length % 64
        data = [0xFFFF_FFFF_FFFF_FFFF] * words
        if tail_bits > 0:
            data.append((1 << tail_bits) - 1)
        return BitStreamTensor.from_packed(data, length)
    else:
        words = (length + 63) // 64
        return BitStreamTensor.from_packed([0] * words, length)


def inject_errors(tensor: BitStreamTensor, error_rate: float, seed: int) -> BitStreamTensor:
    """XOR with a random noise stream to flip bits at the given rate."""
    noise = BitStreamTensor(tensor.length, seed=seed)
    # Scale: only keep ~error_rate fraction of noise bits
    # We create a threshold mask by generating another stream at (1 - error_rate)
    # and ANDing — but for simplicity, we use the noise directly since p=0.5
    # For controlled error rates, we XOR with a low-density stream
    import random
    rng = random.Random(seed)
    bits = []
    for _ in range(tensor.length):
        bits.append(1 if rng.random() < error_rate else 0)
    # Pack manually
    words = (tensor.length + 63) // 64
    data = [0] * words
    for i, b in enumerate(bits):
        if b:
            data[i // 64] |= 1 << (i % 64)
    noise = BitStreamTensor.from_packed(data, tensor.length)
    return tensor.xor(noise)


def decode_majority(tensor: BitStreamTensor) -> bool:
    """Decode by majority vote: True if more than half the bits are set."""
    return tensor.popcount() > tensor.length // 2


def stochastic_and(a: BitStreamTensor, b: BitStreamTensor) -> BitStreamTensor:
    """Stochastic AND via bitwise AND (standard SC multiplication)."""
    # For Boolean streams, AND of all-ones and all-ones = all-ones
    # We reuse the XOR infrastructure but need AND — use from_packed
    data = [wa & wb for wa, wb in zip(a.data, b.data)]
    return BitStreamTensor.from_packed(data, a.length)


def stochastic_or(a: BitStreamTensor, b: BitStreamTensor) -> BitStreamTensor:
    """Stochastic OR via bitwise OR."""
    data = [wa | wb for wa, wb in zip(a.data, b.data)]
    return BitStreamTensor.from_packed(data, a.length)


def stochastic_not(a: BitStreamTensor) -> BitStreamTensor:
    """Stochastic NOT via bitwise complement."""
    words = (a.length + 63) // 64
    data = list(a.data)
    for i in range(len(data)):
        data[i] = ~data[i] & 0xFFFF_FFFF_FFFF_FFFF
    # Mask off trailing bits in last word
    tail = a.length % 64
    if tail > 0 and data:
        data[-1] &= (1 << tail) - 1
    return BitStreamTensor.from_packed(data, a.length)


def main():
    LENGTH = 1024  # Redundancy factor: 1024 bits per Boolean
    ERROR_RATE = 0.05  # 5% random bit-flip rate

    print(f"SC-NeuroCore Fault-Tolerant Binary Streams Demo")
    print(f"  Redundancy: {LENGTH} bits per Boolean")
    print(f"  Error rate: {ERROR_RATE*100:.1f}%")
    print("=" * 55)

    # ── Step 1: Encode Boolean values ─────────────────────────────

    val_a = True
    val_b = False
    val_c = True

    a = make_constant(val_a, LENGTH)
    b = make_constant(val_b, LENGTH)
    c = make_constant(val_c, LENGTH)

    print(f"\nEncoded: A={val_a} ({a.popcount()}/{LENGTH} ones)")
    print(f"         B={val_b} ({b.popcount()}/{LENGTH} ones)")
    print(f"         C={val_c} ({c.popcount()}/{LENGTH} ones)")

    # ── Step 2: Inject errors ─────────────────────────────────────

    a_noisy = inject_errors(a, ERROR_RATE, seed=1)
    b_noisy = inject_errors(b, ERROR_RATE, seed=2)
    c_noisy = inject_errors(c, ERROR_RATE, seed=3)

    print(f"\nAfter {ERROR_RATE*100:.1f}% noise injection:")
    print(f"  A: {a_noisy.popcount()}/{LENGTH} ones ->decode={decode_majority(a_noisy)}")
    print(f"  B: {b_noisy.popcount()}/{LENGTH} ones ->decode={decode_majority(b_noisy)}")
    print(f"  C: {c_noisy.popcount()}/{LENGTH} ones ->decode={decode_majority(c_noisy)}")

    # Verify correctness
    assert decode_majority(a_noisy) == val_a, "Error: A decoded incorrectly"
    assert decode_majority(b_noisy) == val_b, "Error: B decoded incorrectly"
    assert decode_majority(c_noisy) == val_c, "Error: C decoded incorrectly"
    print("  All decoded correctly despite noise!")

    # ── Step 3: Boolean logic on noisy streams ────────────────────

    print(f"\nBoolean logic on noisy streams:")

    # AND
    and_result = stochastic_and(a_noisy, c_noisy)
    decoded_and = decode_majority(and_result)
    expected_and = val_a and val_c
    print(f"  A AND C: {decoded_and} (expected {expected_and}) "
          f"[{and_result.popcount()}/{LENGTH} ones]")
    assert decoded_and == expected_and

    # OR
    or_result = stochastic_or(a_noisy, b_noisy)
    decoded_or = decode_majority(or_result)
    expected_or = val_a or val_b
    print(f"  A OR  B: {decoded_or} (expected {expected_or}) "
          f"[{or_result.popcount()}/{LENGTH} ones]")
    assert decoded_or == expected_or

    # NOT
    not_result = stochastic_not(b_noisy)
    decoded_not = decode_majority(not_result)
    expected_not = not val_b
    print(f"  NOT   B: {decoded_not} (expected {expected_not}) "
          f"[{not_result.popcount()}/{LENGTH} ones]")
    assert decoded_not == expected_not

    # Complex: (A AND C) OR (NOT B)
    complex_result = stochastic_or(
        stochastic_and(a_noisy, c_noisy),
        stochastic_not(b_noisy),
    )
    decoded_complex = decode_majority(complex_result)
    expected_complex = (val_a and val_c) or (not val_b)
    print(f"  (A AND C) OR (NOT B): {decoded_complex} (expected {expected_complex})")
    assert decoded_complex == expected_complex

    # ── Step 4: Error rate sweep ──────────────────────────────────

    print(f"\nError tolerance sweep (100 trials per rate):")
    for rate in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45]:
        successes = 0
        trials = 100
        for trial in range(trials):
            noisy = inject_errors(a, rate, seed=10000 + trial)
            if decode_majority(noisy) == val_a:
                successes += 1
        pct = successes / trials * 100
        print(f"  {rate*100:5.1f}% error rate ->{pct:5.1f}% correct decoding")

    print("\nDone. Stochastic redundancy provides robust error suppression!")


if __name__ == "__main__":
    main()
