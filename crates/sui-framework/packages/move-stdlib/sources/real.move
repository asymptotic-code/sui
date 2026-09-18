module std::real;

#[mode(spec), ext(spec_only)]
use std::integer::Integer;

#[mode(spec), ext(spec_only)]
native public struct Real has copy, drop, store;

#[mode(spec), ext(spec_only)]
native public fun from_integer(x: Integer): Real;
#[mode(spec), ext(spec_only)]
native public fun to_integer(x: Real): Integer;

#[mode(spec), ext(spec_only, pure)]
public fun from_u8(x: u8): Real {
    x.to_int().to_real()
}
#[mode(spec), ext(spec_only, pure)]
public fun from_u16(x: u16): Real {
    x.to_int().to_real()
}
#[mode(spec), ext(spec_only, pure)]
public fun from_u32(x: u32): Real {
    x.to_int().to_real()
}
#[mode(spec), ext(spec_only, pure)]
public fun from_u64(x: u64): Real {
    x.to_int().to_real()
}
#[mode(spec), ext(spec_only, pure)]
public fun from_u128(x: u128): Real {
    x.to_int().to_real()
}
#[mode(spec), ext(spec_only, pure)]
public fun from_u256(x: u256): Real {
    x.to_int().to_real()
}

#[mode(spec), ext(spec_only, pure)]
public fun to_u8(x: Real): u8 {
    x.to_integer().to_u8()
}
#[mode(spec), ext(spec_only, pure)]
public fun to_u16(x: Real): u16 {
    x.to_integer().to_u16()
}
#[mode(spec), ext(spec_only, pure)]
public fun to_u32(x: Real): u32 {
    x.to_integer().to_u32()
}
#[mode(spec), ext(spec_only, pure)]
public fun to_u64(x: Real): u64 {
    x.to_integer().to_u64()
}
#[mode(spec), ext(spec_only, pure)]
public fun to_u128(x: Real): u128 {
    x.to_integer().to_u128()
}
#[mode(spec), ext(spec_only, pure)]
public fun to_u256(x: Real): u256 {
    x.to_integer().to_u256()
}

#[mode(spec), ext(spec_only)]
native public fun add(x: Real, y: Real): Real;
#[mode(spec), ext(spec_only)]
native public fun sub(x: Real, y: Real): Real;
#[mode(spec), ext(spec_only)]
native public fun neg(x: Real): Real;
#[mode(spec), ext(spec_only)]
native public fun mul(x: Real, y: Real): Real;
#[mode(spec), ext(spec_only)]
native public fun div(x: Real, y: Real): Real;
#[mode(spec), ext(spec_only)]
native public fun sqrt(x: Real): Real;
#[mode(spec), ext(spec_only)]
native public fun exp(x: Real, y: Integer): Real;

#[mode(spec), ext(spec_only)]
native public fun lt(x: Real, y: Real): bool;
#[mode(spec), ext(spec_only)]
native public fun gt(x: Real, y: Real): bool;
#[mode(spec), ext(spec_only)]
native public fun lte(x: Real, y: Real): bool;
#[mode(spec), ext(spec_only)]
native public fun gte(x: Real, y: Real): bool;

#[mode(spec), ext(spec_only, pure)]
public fun min(x: Real, y: Real): Real {
    if (x.lt(y)) x else y
}

#[mode(spec), ext(spec_only, pure)]
public fun max(x: Real, y: Real): Real {
    if (x.gt(y)) x else y
}

#[mode(spec), ext(spec_only)]
public use fun std::q32::from_real as Real.to_q32;
#[mode(spec), ext(spec_only)]
public use fun std::q64::from_real as Real.to_q64;
#[mode(spec), ext(spec_only)]
public use fun std::q128::from_real as Real.to_q128;
#[mode(spec), ext(spec_only)]
public use fun std::q_wad::from_real as Real.to_q_wad;
#[mode(spec), ext(spec_only)]
public use fun std::q_ray::from_real as Real.to_q_ray;
