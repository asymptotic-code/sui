/// Signed fixed-point type with 128 fractional bits.
/// Internally stores an arbitrary-precision `Integer`; actual value = val / 2^128.
module std::q128;

#[spec_only]
use std::integer::Integer;
#[spec_only]
use std::real::Real;

#[spec_only]
const SCALE: u256 = 0x1_0000_0000_0000_0000_0000_0000_0000_0000;
#[spec_only]
const HALF_SCALE: u256 = 0x8000_0000_0000_0000_0000_0000_0000_0000;

#[spec_only]
public struct Q128 has copy, drop, store { val: Integer }

// === Construction ===

#[spec_only, ext(pure)]
public fun from_integer(x: Integer): Q128 {
    Q128 { val: x.mul(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun from_u8(x: u8): Q128 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u16(x: u16): Q128 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u32(x: u32): Q128 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u64(x: u64): Q128 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u128(x: u128): Q128 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u256(x: u256): Q128 { from_integer(x.to_int()) }

#[spec_only, ext(pure)]
public fun from_real(x: Real): Q128 {
    Q128 { val: x.mul(SCALE.to_real()).to_integer() }
}

#[spec_only]
public macro fun zero(): Q128 {
    Q128 { val: std::integer::zero!() }
}

#[spec_only]
public macro fun one(): Q128 {
    from_integer(std::integer::one!())
}

#[spec_only, ext(pure)]
public fun quot(num: Integer, den: Integer): Q128 {
    Q128 { val: num.mul(SCALE.to_int()).div(den) }
}

// === Accessors ===

#[spec_only, ext(pure)]
public fun raw(q: Q128): Integer { q.val }

#[spec_only, ext(pure)]
public fun from_raw(val: Integer): Q128 { Q128 { val } }

// === Arithmetic ===

#[spec_only, ext(pure)]
public fun add(a: Q128, b: Q128): Q128 {
    Q128 { val: a.val.add(b.val) }
}

#[spec_only, ext(pure)]
public fun sub(a: Q128, b: Q128): Q128 {
    Q128 { val: a.val.sub(b.val) }
}

#[spec_only, ext(pure)]
public fun mul(a: Q128, b: Q128): Q128 {
    Q128 { val: a.val.mul(b.val).div(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun div(a: Q128, b: Q128): Q128 {
    Q128 { val: a.val.mul(SCALE.to_int()).div(b.val) }
}

#[spec_only, ext(pure)]
public fun neg(a: Q128): Q128 {
    Q128 { val: a.val.neg() }
}

#[spec_only, ext(pure)]
public fun abs(a: Q128): Q128 {
    Q128 { val: a.val.abs() }
}

#[spec_only, ext(pure)]
public fun sqrt(a: Q128): Q128 {
    Q128 { val: a.val.mul(SCALE.to_int()).sqrt() }
}

#[spec_only, ext(pure)]
public fun pow(x: Q128, n: Integer): Q128 {
    Q128 { val: std::macros::q_pow!(x.val, n, SCALE.to_int()) }
}

// === Comparisons ===

#[spec_only, ext(pure)]
public fun lt(a: Q128, b: Q128): bool { a.val.lt(b.val) }

#[spec_only, ext(pure)]
public fun gt(a: Q128, b: Q128): bool { a.val.gt(b.val) }

#[spec_only, ext(pure)]
public fun lte(a: Q128, b: Q128): bool { a.val.lte(b.val) }

#[spec_only, ext(pure)]
public fun gte(a: Q128, b: Q128): bool { a.val.gte(b.val) }

// === Rounding / Conversion ===

#[spec_only, ext(pure)]
public fun floor(q: Q128): Integer {
    q.val.div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun ceil(q: Q128): Integer {
    q.val.div_round_up(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun round(q: Q128): Integer {
    q.val.add(HALF_SCALE.to_int()).div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun to_int(q: Q128): Integer { floor(q) }

#[spec_only, ext(pure)]
public fun to_real(q: Q128): Real {
    q.val.to_real().div(SCALE.to_real())
}

// === Predicates ===

#[spec_only]
public macro fun is_zero($q: Q128): bool {
    let q = $q;
    q.val == std::integer::zero!()
}

#[spec_only, ext(pure)]
public fun is_pos(q: Q128): bool { q.val.is_pos() }

#[spec_only, ext(pure)]
public fun is_neg(q: Q128): bool { q.val.is_neg() }

#[spec_only, ext(pure)]
public fun is_int(q: Q128): bool {
    q.val.mod(SCALE.to_int()) == 0u64.to_int()
}

#[spec_only, ext(pure)]
public fun is_uq64_128(x: Q128): bool {
    !x.is_neg() && x.lt(0x1_0000_0000_0000_0000u256.to_q128())
}

#[spec_only, ext(pure)]
public fun is_uq128_128(x: Q128): bool {
    !x.is_neg() && x.lt(0x1_0000_0000_0000_0000_0000_0000_0000_0000u256.to_q128())
}
