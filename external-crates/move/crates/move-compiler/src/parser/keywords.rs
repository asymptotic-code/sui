// Copyright (c) The Diem Core Contributors
// Copyright (c) The Move Contributors
// SPDX-License-Identifier: Apache-2.0

pub const KEYWORDS: &[&str] = &[
    "abort", "acquires", "as", "break", "const", "continue", "copy", "else", "false", "friend",
    "fun", "has", "if", "let", "loop", "module", "move", "native", "public", "return", "struct",
    "true", "use", "while",
];

pub const CONTEXTUAL_KEYWORDS: &[&str] = &[
    "aborts_if",
    "aborts_with",
    "address",
    "apply",
    "assume",
    "axiom",
    "choose",
    "decreases",
    "emits",
    "ensures",
    "except",
    "forall",
    "global",
    "include",
    "internal",
    "local",
    "min",
    "modifies",
    "mut",
    "phantom",
    "post",
    "pragma",
    "requires",
    "Self",
    "schema",
    "succeeds_if",
    "to",
    "update",
    "where",
    "with",
];

pub const PRIMITIVE_TYPES: &[&str] = &["u8", "u16", "u32", "u64", "u128", "u256", "bool", "vector"];

pub const BUILTINS: &[&str] = &["assert", "freeze", "old"];
