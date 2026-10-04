// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — qualified Rust item and member syntax identities

//! Resolve declaration ownership through Rust syntax instead of warning text.

use proc_macro2::Span;
use quote::ToTokens;
use serde::Serialize;
use syn::spanned::Spanned;
use syn::{Fields, ForeignItem, ImplItem, Item, TraitItem};

/// One syntax-qualified declaration associated with its identifier bytes.
#[derive(Serialize)]
pub(crate) struct Symbol {
    /// Native declaration category used to match a compiler diagnostic.
    kind: &'static str,
    /// Identifier spelling or tuple-field index.
    name: String,
    /// Full enclosing module/type/trait/implementation syntax identity.
    identity: String,
    /// Inclusive identifier start in the exact parser input byte stream.
    start: usize,
    /// Exclusive identifier end in the exact parser input byte stream.
    end: usize,
}

/// Append an item identity without tying it to mutable line numbers.
fn symbol(out: &mut Vec<Symbol>, context: &str, kind: &'static str, name: &str, span: Span) {
    let range = span.byte_range();
    let identity = format!("{context}/{kind}:{name}");
    out.push(Symbol {
        kind,
        name: name.to_owned(),
        identity,
        start: range.start,
        end: range.end,
    });
}

/// Include named and positional fields under their actual type or variant owner.
fn fields(out: &mut Vec<Symbol>, context: &str, members: &Fields) {
    for (index, field) in members.iter().enumerate() {
        if let Some(name) = &field.ident {
            symbol(out, context, "field", &name.to_string(), name.span());
        } else {
            symbol(out, context, "field", &index.to_string(), field.ty.span());
        }
    }
}

/// Collect declarations using enclosing Rust modules, types and implementation syntax.
pub(crate) fn collect(items: &[Item], context: &str) -> Vec<Symbol> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Item::Mod(value) => {
                let name = value.ident.to_string();
                symbol(&mut out, context, "module", &name, value.ident.span());
                if let Some((_, items)) = &value.content {
                    out.extend(collect(items, &format!("{context}/module:{name}")));
                }
            }
            Item::Fn(value) => symbol(
                &mut out,
                context,
                "function",
                &value.sig.ident.to_string(),
                value.sig.ident.span(),
            ),
            Item::Struct(value) => {
                let name = value.ident.to_string();
                symbol(&mut out, context, "struct", &name, value.ident.span());
                fields(&mut out, &format!("{context}/struct:{name}"), &value.fields);
            }
            Item::Enum(value) => {
                let name = value.ident.to_string();
                symbol(&mut out, context, "enum", &name, value.ident.span());
                for variant in &value.variants {
                    let variant_name = variant.ident.to_string();
                    let owner = format!("{context}/enum:{name}");
                    symbol(
                        &mut out,
                        &owner,
                        "variant",
                        &variant_name,
                        variant.ident.span(),
                    );
                    fields(
                        &mut out,
                        &format!("{owner}/variant:{variant_name}"),
                        &variant.fields,
                    );
                }
            }
            Item::Union(value) => {
                let name = value.ident.to_string();
                symbol(&mut out, context, "union", &name, value.ident.span());
                fields(
                    &mut out,
                    &format!("{context}/union:{name}"),
                    &Fields::Named(value.fields.clone()),
                );
            }
            Item::Trait(value) => {
                let name = value.ident.to_string();
                symbol(&mut out, context, "trait", &name, value.ident.span());
                let owner = format!("{context}/trait:{name}");
                for member in &value.items {
                    match member {
                        TraitItem::Fn(value) => symbol(
                            &mut out,
                            &owner,
                            "method",
                            &value.sig.ident.to_string(),
                            value.sig.ident.span(),
                        ),
                        TraitItem::Const(value) => symbol(
                            &mut out,
                            &owner,
                            "constant",
                            &value.ident.to_string(),
                            value.ident.span(),
                        ),
                        TraitItem::Type(value) => symbol(
                            &mut out,
                            &owner,
                            "type",
                            &value.ident.to_string(),
                            value.ident.span(),
                        ),
                        _ => {}
                    }
                }
            }
            Item::Impl(value) => {
                let target = value.self_ty.to_token_stream().to_string();
                let implemented = value
                    .trait_
                    .as_ref()
                    .map(|(negated, path, _)| {
                        format!(
                            "{}{} for ",
                            if negated.is_some() { "!" } else { "" },
                            path.to_token_stream()
                        )
                    })
                    .unwrap_or_default();
                let owner = format!("{context}/impl:{implemented}{target}");
                for member in &value.items {
                    match member {
                        ImplItem::Fn(value) => symbol(
                            &mut out,
                            &owner,
                            "method",
                            &value.sig.ident.to_string(),
                            value.sig.ident.span(),
                        ),
                        ImplItem::Const(value) => symbol(
                            &mut out,
                            &owner,
                            "constant",
                            &value.ident.to_string(),
                            value.ident.span(),
                        ),
                        ImplItem::Type(value) => symbol(
                            &mut out,
                            &owner,
                            "type",
                            &value.ident.to_string(),
                            value.ident.span(),
                        ),
                        _ => {}
                    }
                }
            }
            Item::ForeignMod(value) => {
                let abi = value
                    .abi
                    .name
                    .as_ref()
                    .map(|name| name.value())
                    .unwrap_or_else(|| "C".into());
                let owner = format!("{context}/extern:{abi}");
                for member in &value.items {
                    match member {
                        ForeignItem::Fn(value) => symbol(
                            &mut out,
                            &owner,
                            "function",
                            &value.sig.ident.to_string(),
                            value.sig.ident.span(),
                        ),
                        ForeignItem::Static(value) => symbol(
                            &mut out,
                            &owner,
                            "static",
                            &value.ident.to_string(),
                            value.ident.span(),
                        ),
                        ForeignItem::Type(value) => symbol(
                            &mut out,
                            &owner,
                            "type",
                            &value.ident.to_string(),
                            value.ident.span(),
                        ),
                        _ => {}
                    }
                }
            }
            Item::Const(value) => symbol(
                &mut out,
                context,
                "constant",
                &value.ident.to_string(),
                value.ident.span(),
            ),
            Item::Static(value) => symbol(
                &mut out,
                context,
                "static",
                &value.ident.to_string(),
                value.ident.span(),
            ),
            Item::Type(value) => symbol(
                &mut out,
                context,
                "type",
                &value.ident.to_string(),
                value.ident.span(),
            ),
            Item::TraitAlias(value) => symbol(
                &mut out,
                context,
                "trait",
                &value.ident.to_string(),
                value.ident.span(),
            ),
            Item::ExternCrate(value) => symbol(
                &mut out,
                context,
                "extern_crate",
                &value.ident.to_string(),
                value.ident.span(),
            ),
            Item::Macro(value) => {
                if let Some(name) = &value.ident {
                    symbol(&mut out, context, "macro", &name.to_string(), name.span());
                }
            }
            _ => {}
        }
    }
    out
}
