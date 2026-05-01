# Rust Fuzzing Harness

This directory contains the `cargo-fuzz` harnesses for native Rust parser and
bitstream surfaces.

## Install

```bash
cargo install cargo-fuzz
```

## Targets

```bash
cargo fuzz run ir_parser
cargo fuzz run bitstream_ops
```

`ir_parser` feeds arbitrary UTF-8 text into the SC IR parser, verifies any
successfully parsed graph, prints it, and reparses the normalized form.

`bitstream_ops` checks pack/unpack, fast-pack parity, popcount, rotate,
bitwise-AND, and hamming-distance invariants over bounded bitstreams.

Keep generated corpora and crash artifacts out of release packages unless they
are intentionally minimized and documented.
