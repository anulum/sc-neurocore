// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Width-explicit compound CIRCT operations

//! Build compound datapaths from typed HW, Comb and Seq instructions.

/// An SSA word with its physical width and arithmetic interpretation.
#[derive(Clone)]
pub(super) struct Value {
    /// Complete MLIR SSA reference, including its leading percent sign.
    pub name: String,
    /// Physical number of bits carried by this SSA value.
    pub width: usize,
    /// Interpretation used for subsequent extension and arithmetic operations.
    pub signed: bool,
}

/// Allocate unique local SSA names while retaining every intermediate width.
pub(super) struct Net<'a> {
    text: &'a mut String,
    prefix: String,
    next: usize,
}

impl<'a> Net<'a> {
    /// Start one operation's private SSA namespace.
    pub fn new(text: &'a mut String, id: u32) -> Self {
        Self {
            text,
            prefix: format!("v{id}_compound"),
            next: 0,
        }
    }

    /// Emit one typed instruction and return its result word.
    fn instruction(&mut self, rhs: String, width: usize, signed: bool) -> Value {
        let name = format!("%{}_{}", self.prefix, self.next);
        self.next += 1;
        self.text.push_str(&format!("  {name} = {rhs}\n"));
        Value {
            name,
            width,
            signed,
        }
    }

    /// Materialise an encoded integer, without implicit scaling.
    pub fn constant(&mut self, number: i64, width: usize, signed: bool) -> Value {
        self.instruction(format!("hw.constant {number} : i{width}"), width, signed)
    }

    /// Preserve low bits when narrowing and extend using the source signedness.
    pub fn resize(&mut self, input: &Value, width: usize, signed: bool) -> Value {
        let result = if width < input.width {
            self.extract(input, 0, width)
        } else if width > input.width {
            let pad = width - input.width;
            let extension = if input.signed {
                let sign = self.extract(input, input.width - 1, 1);
                self.instruction(
                    format!("comb.replicate {} : (i1) -> i{pad}", sign.name),
                    pad,
                    false,
                )
            } else {
                self.constant(0, pad, false)
            };
            self.concat(&[extension, input.clone()])
        } else {
            input.clone()
        };
        Value { signed, ..result }
    }

    /// Extract a packed lane or selected product bits.
    pub fn extract(&mut self, input: &Value, low: usize, width: usize) -> Value {
        assert!(width > 0 && low + width <= input.width);
        self.instruction(
            format!(
                "comb.extract {} from {low} : (i{}) -> i{width}",
                input.name, input.width
            ),
            width,
            false,
        )
    }

    /// Concatenate physical words from the most significant lane to the least.
    pub fn concat(&mut self, inputs: &[Value]) -> Value {
        let names = inputs
            .iter()
            .map(|v| v.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        let types = inputs
            .iter()
            .map(|v| format!("i{}", v.width))
            .collect::<Vec<_>>()
            .join(", ");
        self.instruction(
            format!("comb.concat {names} : {types}"),
            inputs.iter().map(|v| v.width).sum(),
            false,
        )
    }

    /// Apply a same-width binary hardware instruction with wrapping arithmetic.
    pub fn binary(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        assert_eq!(a.width, b.width);
        self.instruction(
            format!("comb.{op} {}, {} : i{}", a.name, b.name, a.width),
            a.width,
            a.signed,
        )
    }

    /// Compare raw words using an explicit signed or unsigned CIRCT predicate.
    pub fn compare(&mut self, predicate: &str, a: &Value, b: &Value) -> Value {
        assert_eq!(a.width, b.width);
        self.instruction(
            format!(
                "comb.icmp {predicate} {}, {} : i{}",
                a.name, b.name, a.width
            ),
            1,
            false,
        )
    }

    /// Select identically sized physical words with a Boolean control.
    pub fn select(&mut self, condition: &Value, a: &Value, b: &Value) -> Value {
        assert_eq!(condition.width, 1);
        assert_eq!(a.width, b.width);
        self.instruction(
            format!(
                "comb.mux {}, {}, {} : i{}",
                condition.name, a.name, b.name, a.width
            ),
            a.width,
            a.signed,
        )
    }

    /// Build a complete power-of-two constant table with explicit Boolean multiplexers.
    pub fn lookup(&mut self, index: &Value, entries: &[i64], width: usize) -> Value {
        assert!(entries.len().is_power_of_two());
        let mut values = entries
            .iter()
            .map(|v| self.constant(*v, width, true))
            .collect::<Vec<_>>();
        let mut bit = 0;
        while values.len() > 1 {
            let selector = self.extract(index, bit, 1);
            values = values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| self.select(&selector, &pair[1], &pair[0]))
                .collect();
            bit += 1;
        }
        values[0].clone()
    }

    /// Declare an asynchronously reset register whose next word is defined later.
    pub fn register(&mut self, reset: &Value) -> (Value, String) {
        let next = format!("%{}_next_{}", self.prefix, self.next);
        let state = self.instruction(
            format!(
                "seq.firreg {next} clock %sc_clock reset async %sc_reset, {} : i{}",
                reset.name, reset.width
            ),
            reset.width,
            reset.signed,
        );
        (state, next)
    }

    /// Bind a deferred register input or the enclosing operation's result name.
    pub fn bind(&mut self, name: &str, value: &Value) {
        let zero = self.constant(0, value.width, value.signed);
        self.text.push_str(&format!(
            "  {name} = comb.or {}, {} : i{}\n",
            value.name, zero.name, value.width
        ));
    }
}
