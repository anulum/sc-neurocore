<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SCThreeStatePhantomBurster

`SCThreeStatePhantomBurster` preserves the exact three-state project recurrence
formerly exposed as `BertramPhantomBurster`. It is a count-neutral compatibility
identity and makes no Bertram-paper attribution.

Its state is `(v, s1, s2)`. The potassium current uses instantaneous `n_inf(v)`;
the two slow gates use fixed 20 s and 100 s time constants. A simultaneous RK4
step at `dt=0.5 ms` advances all three states. Sampled upward crossings of
`-20 mV` emit events without reset.

The Python, Rust safety, Rust engine, Julia, Go, and Mojo files retain the old
defaults and recurrence under the new name. The paired schemas and project
receipt make the lack of literature attribution explicit.
