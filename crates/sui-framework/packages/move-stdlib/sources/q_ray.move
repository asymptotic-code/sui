/// Signed fixed-point type with 27 decimal fractional digits (RAY).
/// Internally stores an arbitrary-precision `Integer`; actual value = val / 10^27.
module std::q_ray;

#[spec_only]
use std::integer::Integer;
#[spec_only]
use std::real::Real;

#[spec_only]
const SCALE: u128 = 1_000_000_000_000_000_000_000_000_000;
#[spec_only]
const HALF_SCALE: u128 = 500_000_000_000_000_000_000_000_000;

#[spec_only]
public struct Q_ray has copy, drop, store { val: Integer }

// === Construction ===

#[spec_only, ext(pure)]
public fun from_integer(x: Integer): Q_ray {
    Q_ray { val: x.mul(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun from_u8(x: u8): Q_ray { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u16(x: u16): Q_ray { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u32(x: u32): Q_ray { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u64(x: u64): Q_ray { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u128(x: u128): Q_ray { from_integer(x.to_int()) }
#[spec_only, ext(pure)]
public fun from_u256(x: u256): Q_ray { from_integer(x.to_int()) }

#[spec_only, ext(pure)]
public fun from_real(x: Real): Q_ray {
    Q_ray { val: x.mul(SCALE.to_real()).to_integer() }
}

#[spec_only]
public macro fun zero(): Q_ray {
    Q_ray { val: std::integer::zero!() }
}

#[spec_only]
public macro fun one(): Q_ray {
    from_integer(std::integer::one!())
}

#[spec_only, ext(pure)]
public fun quot(num: Integer, den: Integer): Q_ray {
    Q_ray { val: num.mul(SCALE.to_int()).div(den) }
}

// === Accessors ===

#[spec_only, ext(pure)]
public fun raw(q: Q_ray): Integer { q.val }

#[spec_only, ext(pure)]
public fun from_raw(val: Integer): Q_ray { Q_ray { val } }

// === Arithmetic ===

#[spec_only, ext(pure)]
public fun add(a: Q_ray, b: Q_ray): Q_ray {
    Q_ray { val: a.val.add(b.val) }
}

#[spec_only, ext(pure)]
public fun sub(a: Q_ray, b: Q_ray): Q_ray {
    Q_ray { val: a.val.sub(b.val) }
}

#[spec_only, ext(pure)]
public fun mul(a: Q_ray, b: Q_ray): Q_ray {
    Q_ray { val: a.val.mul(b.val).div(SCALE.to_int()) }
}

#[spec_only, ext(pure)]
public fun div(a: Q_ray, b: Q_ray): Q_ray {
    Q_ray { val: a.val.mul(SCALE.to_int()).div(b.val) }
}

#[spec_only, ext(pure)]
public fun neg(a: Q_ray): Q_ray {
    Q_ray { val: a.val.neg() }
}

#[spec_only, ext(pure)]
public fun abs(a: Q_ray): Q_ray {
    Q_ray { val: a.val.abs() }
}

#[spec_only, ext(pure)]
public fun sqrt(a: Q_ray): Q_ray {
    Q_ray { val: a.val.mul(SCALE.to_int()).sqrt() }
}

#[spec_only, ext(pure)]
public fun pow(x: Q_ray, n: Integer): Q_ray {
    Q_ray { val: std::macros::q_pow!(x.val, n, SCALE.to_int()) }
}

// === Comparisons ===

#[spec_only, ext(pure)]
public fun lt(a: Q_ray, b: Q_ray): bool { a.val.lt(b.val) }

#[spec_only, ext(pure)]
public fun gt(a: Q_ray, b: Q_ray): bool { a.val.gt(b.val) }

#[spec_only, ext(pure)]
public fun lte(a: Q_ray, b: Q_ray): bool { a.val.lte(b.val) }

#[spec_only, ext(pure)]
public fun gte(a: Q_ray, b: Q_ray): bool { a.val.gte(b.val) }

#[spec_only, ext(pure)]
public fun min(a: Q_ray, b: Q_ray): Q_ray {
    if (a.lt(b)) a else b
}

#[spec_only, ext(pure)]
public fun max(a: Q_ray, b: Q_ray): Q_ray {
    if (a.gt(b)) a else b
}

// === Rounding / Conversion ===

#[spec_only, ext(pure)]
public fun floor(q: Q_ray): Integer {
    q.val.div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun ceil(q: Q_ray): Integer {
    q.val.div_round_up(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun round(q: Q_ray): Integer {
    q.val.add(HALF_SCALE.to_int()).div(SCALE.to_int())
}

#[spec_only, ext(pure)]
public fun to_int(q: Q_ray): Integer { floor(q) }

#[spec_only, ext(pure)]
public fun to_real(q: Q_ray): Real {
    q.val.to_real().div(SCALE.to_real())
}

// === Predicates ===

#[spec_only]
public macro fun is_zero($q: Q_ray): bool {
    let q = $q;
    q.val == std::integer::zero!()
}

#[spec_only, ext(pure)]
public fun is_pos(q: Q_ray): bool { q.val.is_pos() }

#[spec_only, ext(pure)]
public fun is_neg(q: Q_ray): bool { q.val.is_neg() }

#[spec_only, ext(pure)]
public fun is_int(q: Q_ray): bool {
    q.val.mod(SCALE.to_int()) == 0u128.to_int()
}

#[spec_only, ext(pure)]
public fun in_range_u128(x: Q_ray): bool {
    !x.is_neg() && x.val.lte(
        0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128.to_int(),
    )
}

#[spec_only, ext(pure)]
public fun in_range_u256(x: Q_ray): bool {
    !x.is_neg() && x.val.lte(
        0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu256
            .to_int(),
    )
}
