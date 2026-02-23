/// Signed fixed-point type with 18 decimal fractional digits (WAD).
/// Internally stores an arbitrary-precision `Integer`; actual value = val / 10^18.
module std::q_wad;

#[spec_only]
use std::integer::Integer;
#[spec_only]
use std::real::Real;

#[spec_only]
const SCALE: u64 = 1_000_000_000_000_000_000;
#[spec_only]
const HALF_SCALE: u64 = 500_000_000_000_000_000;

#[spec_only]
public struct Q_wad has copy, drop, store { val: Integer }

// === Construction ===

#[spec_only, ext(pure)]
public fun from_integer(x: Integer): Q_wad {
    Q_wad { val: x.mul(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun from_u8(x: u8): Q_wad { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u16(x: u16): Q_wad { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u32(x: u32): Q_wad { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u64(x: u64): Q_wad { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u128(x: u128): Q_wad { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u256(x: u256): Q_wad { from_integer(x.to_int()) }

#[spec_only, ext(pure)]
public fun from_real(x: Real): Q_wad {
    Q_wad { val: x.mul(SCALE.to_real()).to_integer() }
}

#[spec_only]
public macro fun zero(): Q_wad {
    Q_wad { val: std::integer::zero!() }
}

#[spec_only]
public macro fun one(): Q_wad {
    from_integer(std::integer::one!())
}

#[spec_only, ext(pure)]
public fun quot(num: Integer, den: Integer): Q_wad {
    Q_wad { val: num.mul(SCALE.to_int()).div(den) }
}

// === Accessors ===

#[spec_only, ext(pure)]
public fun raw(q: Q_wad): Integer { q.val }

#[spec_only, ext(pure)]
public fun from_raw(val: Integer): Q_wad { Q_wad { val } }

// === Arithmetic ===

#[spec_only, ext(pure)]
public fun add(a: Q_wad, b: Q_wad): Q_wad {
    Q_wad { val: a.val.add(b.val) }
}

#[spec_only, ext(pure)]
public fun sub(a: Q_wad, b: Q_wad): Q_wad {
    Q_wad { val: a.val.sub(b.val) }
}

#[spec_only, ext(pure)]
public fun mul(a: Q_wad, b: Q_wad): Q_wad {
    Q_wad { val: a.val.mul(b.val).div(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun div(a: Q_wad, b: Q_wad): Q_wad {
    Q_wad { val: a.val.mul(SCALE.to_int()).div(b.val) }
}

#[spec_only, ext(pure)]
public fun neg(a: Q_wad): Q_wad {
    Q_wad { val: a.val.neg() }
}

#[spec_only, ext(pure)]
public fun abs(a: Q_wad): Q_wad {
    Q_wad { val: a.val.abs() }
}

#[spec_only, ext(pure)]
public fun sqrt(a: Q_wad): Q_wad {
    Q_wad { val: a.val.mul(SCALE.to_int()).sqrt() }
}

#[spec_only, ext(pure)]
public fun pow(x: Q_wad, n: Integer): Q_wad {
    Q_wad { val: std::macros::q_pow!(x.val, n, SCALE.to_int()) }
}

// === Comparisons ===

#[spec_only, ext(pure)]
public fun lt(a: Q_wad, b: Q_wad): bool { a.val.lt(b.val) }

#[spec_only, ext(pure)]
public fun gt(a: Q_wad, b: Q_wad): bool { a.val.gt(b.val) }

#[spec_only, ext(pure)]
public fun lte(a: Q_wad, b: Q_wad): bool { a.val.lte(b.val) }

#[spec_only, ext(pure)]
public fun gte(a: Q_wad, b: Q_wad): bool { a.val.gte(b.val) }

#[spec_only, ext(pure)]
public fun min(a: Q_wad, b: Q_wad): Q_wad {
    if (a.lt(b)) a else b
}

#[spec_only, ext(pure)]
public fun max(a: Q_wad, b: Q_wad): Q_wad {
    if (a.gt(b)) a else b
}

// === Rounding / Conversion ===

#[spec_only, ext(pure)]
public fun floor(q: Q_wad): Integer {
    q.val.div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun ceil(q: Q_wad): Integer {
    q.val.div_round_up(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun round(q: Q_wad): Integer {
    q.val.add(HALF_SCALE.to_int()).div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun to_int(q: Q_wad): Integer { floor(q) }

#[spec_only, ext(pure)]
public fun to_real(q: Q_wad): Real {
    q.val.to_real().div(SCALE.to_real())
}

// === Predicates ===

#[spec_only]
public macro fun is_zero($q: Q_wad): bool {
    let q = $q;
    q.val == std::integer::zero!()
}

#[spec_only, ext(pure)]
public fun is_pos(q: Q_wad): bool { q.val.is_pos() }

#[spec_only, ext(pure)]
public fun is_neg(q: Q_wad): bool { q.val.is_neg() }

#[spec_only, ext(pure)]
public fun is_int(q: Q_wad): bool {
    q.val.mod(SCALE.to_int()) == 0u64.to_int()
}

#[spec_only, ext(pure)]
public fun in_range_u64(x: Q_wad): bool {
    !x.is_neg() && x.val.lte(0xFFFF_FFFF_FFFF_FFFFu64.to_int())
}

#[spec_only, ext(pure)]
public fun in_range_u128(x: Q_wad): bool {
    !x.is_neg() && x.val.lte(
        0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128.to_int(),
    )
}

#[spec_only, ext(pure)]
public fun in_range_u256(x: Q_wad): bool {
    !x.is_neg() && x.val.lte(
        0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu256
            .to_int(),
    )
}
