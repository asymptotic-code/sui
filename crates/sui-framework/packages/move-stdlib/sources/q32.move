/// Signed fixed-point type with 32 fractional bits.
/// Internally stores an arbitrary-precision `Integer`; actual value = val / 2^32.
module std::q32;

#[spec_only]
use std::integer::Integer;
#[spec_only]
use std::real::Real;

#[spec_only]
const SCALE: u64 = 0x1_0000_0000;
#[spec_only]
const HALF_SCALE: u64 = 0x8000_0000;

#[spec_only]
public struct Q32 has copy, drop, store { val: Integer }

// === Construction ===

#[spec_only, ext(pure)]
public fun from_integer(x: Integer): Q32 {
    Q32 { val: x.mul(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun from_u8(x: u8): Q32 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u16(x: u16): Q32 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u32(x: u32): Q32 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u64(x: u64): Q32 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u128(x: u128): Q32 { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u256(x: u256): Q32 { from_integer(x.to_int()) }

#[spec_only, ext(pure)]
public fun from_real(x: Real): Q32 {
    Q32 { val: x.mul(SCALE.to_real()).to_integer() }
}

#[spec_only]
public macro fun zero(): Q32 {
    Q32 { val: std::integer::zero!() }
}

#[spec_only]
public macro fun one(): Q32 {
    from_integer(std::integer::one!())
}

#[spec_only, ext(pure)]
public fun quot(num: Integer, den: Integer): Q32 {
    Q32 { val: num.mul(SCALE.to_int()).div(den) }
}

// === Accessors ===

#[spec_only, ext(pure)]
public fun raw(q: Q32): Integer { q.val }

#[spec_only, ext(pure)]
public fun from_raw(val: Integer): Q32 { Q32 { val } }

// === Arithmetic ===

#[spec_only, ext(pure)]
public fun add(a: Q32, b: Q32): Q32 {
    Q32 { val: a.val.add(b.val) }
}

#[spec_only, ext(pure)]
public fun sub(a: Q32, b: Q32): Q32 {
    Q32 { val: a.val.sub(b.val) }
}

#[spec_only, ext(pure)]
public fun mul(a: Q32, b: Q32): Q32 {
    Q32 { val: a.val.mul(b.val).div(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun div(a: Q32, b: Q32): Q32 {
    Q32 { val: a.val.mul(SCALE.to_int()).div(b.val) }
}

#[spec_only, ext(pure)]
public fun neg(a: Q32): Q32 {
    Q32 { val: a.val.neg() }
}

#[spec_only, ext(pure)]
public fun abs(a: Q32): Q32 {
    Q32 { val: a.val.abs() }
}

#[spec_only, ext(pure)]
public fun sqrt(a: Q32): Q32 {
    Q32 { val: a.val.mul(SCALE.to_int()).sqrt() }
}

#[spec_only, ext(pure)]
public fun pow(x: Q32, n: Integer): Q32 {
    Q32 { val: std::macros::q_pow!(x.val, n, SCALE.to_int()) }
}

// === Comparisons ===

#[spec_only, ext(pure)]
public fun lt(a: Q32, b: Q32): bool { a.val.lt(b.val) }

#[spec_only, ext(pure)]
public fun gt(a: Q32, b: Q32): bool { a.val.gt(b.val) }

#[spec_only, ext(pure)]
public fun lte(a: Q32, b: Q32): bool { a.val.lte(b.val) }

#[spec_only, ext(pure)]
public fun gte(a: Q32, b: Q32): bool { a.val.gte(b.val) }

// === Rounding / Conversion ===

#[spec_only, ext(pure)]
public fun floor(q: Q32): Integer {
    q.val.div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun ceil(q: Q32): Integer {
    q.val.div_round_up(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun round(q: Q32): Integer {
    q.val.add(HALF_SCALE.to_int()).div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun to_int(q: Q32): Integer { floor(q) }

#[spec_only, ext(pure)]
public fun to_real(q: Q32): Real {
    q.val.to_real().div(SCALE.to_real())
}

// === Predicates ===

#[spec_only]
public macro fun is_zero($q: Q32): bool {
    let q = $q;
    q.val == std::integer::zero!()
}

#[spec_only, ext(pure)]
public fun is_pos(q: Q32): bool { q.val.is_pos() }

#[spec_only, ext(pure)]
public fun is_neg(q: Q32): bool { q.val.is_neg() }

#[spec_only, ext(pure)]
public fun is_int(q: Q32): bool {
    q.val.mod(SCALE.to_int()) == 0u64.to_int()
}

#[spec_only, ext(pure)]
public fun is_uq32_32(x: Q32): bool {
    !x.is_neg() && x.lt(0x1_0000_0000u64.to_q32())
}

// === Fixed-point conversions ===

#[spec_only, ext(pure)]
public fun from_fp32(x: std::fixed_point32::FixedPoint32): Q32 {
    Q32 { val: x.get_raw_value().to_int() }
}

#[spec_only, ext(pure)]
public fun from_uq32_32(x: std::uq32_32::UQ32_32): Q32 {
    Q32 { val: x.to_raw().to_int() }
}
