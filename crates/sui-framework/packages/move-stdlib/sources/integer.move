module std::integer;

#[mode(spec), ext(spec_only)]
native public struct Integer has copy, drop, store;

#[mode(spec), ext(spec_only)]
native public fun from_u8(x: u8): Integer;
#[mode(spec), ext(spec_only)]
native public fun from_u16(x: u16): Integer;
#[mode(spec), ext(spec_only)]
native public fun from_u32(x: u32): Integer;
#[mode(spec), ext(spec_only)]
native public fun from_u64(x: u64): Integer;
#[mode(spec), ext(spec_only)]
native public fun from_u128(x: u128): Integer;
#[mode(spec), ext(spec_only)]
native public fun from_u256(x: u256): Integer;

#[mode(spec), ext(spec_only)]
native public fun to_u8(x: Integer): u8;
#[mode(spec), ext(spec_only)]
native public fun to_u16(x: Integer): u16;
#[mode(spec), ext(spec_only)]
native public fun to_u32(x: Integer): u32;
#[mode(spec), ext(spec_only)]
native public fun to_u64(x: Integer): u64;
#[mode(spec), ext(spec_only)]
native public fun to_u128(x: Integer): u128;
#[mode(spec), ext(spec_only)]
native public fun to_u256(x: Integer): u256;

#[mode(spec), ext(spec_only)]
public use fun std::real::from_integer as Integer.to_real;

#[mode(spec), ext(spec_only)]
public use fun std::q32::from_integer as Integer.to_q32;
#[mode(spec), ext(spec_only)]
public use fun std::q64::from_integer as Integer.to_q64;
#[mode(spec), ext(spec_only)]
public use fun std::q128::from_integer as Integer.to_q128;
#[mode(spec), ext(spec_only)]
public use fun std::q_wad::from_integer as Integer.to_q_wad;
#[mode(spec), ext(spec_only)]
public use fun std::q_ray::from_integer as Integer.to_q_ray;

#[mode(spec), ext(spec_only)]
native public fun add(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun sub(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun neg(x: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun mul(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun div(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun mod(x: Integer, y: Integer): Integer;

#[mode(spec), ext(spec_only, pure)]
public fun div_round_up(x: Integer, y: Integer): Integer {
    let result = x.div(y);
    if (x.mod(y) != 0u64.to_int()) {
        result.add(1u64.to_int())
    } else {
        result
    }
}

#[mode(spec), ext(spec_only)]
native public fun sqrt(x: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun pow(x: Integer, y: Integer): Integer;

#[mode(spec), ext(spec_only, pure)]
public fun shl(x: Integer, y: Integer): Integer {
    x.mul(2u8.to_int().pow(y))
}
#[mode(spec), ext(spec_only, pure)]
public fun shr(x: Integer, y: Integer): Integer {
    x.div(2u8.to_int().pow(y))
}

#[mode(spec), ext(spec_only)]
native public fun bit_or(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun bit_and(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun bit_xor(x: Integer, y: Integer): Integer;
#[mode(spec), ext(spec_only)]
native public fun bit_not(x: Integer): Integer;

#[mode(spec), ext(spec_only)]
native public fun lt(x: Integer, y: Integer): bool;
#[mode(spec), ext(spec_only)]
native public fun gt(x: Integer, y: Integer): bool;
#[mode(spec), ext(spec_only)]
native public fun lte(x: Integer, y: Integer): bool;
#[mode(spec), ext(spec_only)]
native public fun gte(x: Integer, y: Integer): bool;

#[mode(spec), ext(spec_only)]
public macro fun zero(): Integer {
    0u64.to_int()
}

#[mode(spec), ext(spec_only)]
public macro fun one(): Integer {
    1u64.to_int()
}

#[mode(spec), ext(spec_only)]
public macro fun two(): Integer {
    2u64.to_int()
}

#[mode(spec), ext(spec_only)]
public macro fun is_zero($x: Integer): bool {
    let x = $x;
    x == zero!()
}

#[mode(spec), ext(spec_only, pure)]
public fun abs(x: Integer): Integer {
    if (x.is_neg()) {
        x.neg()
    } else {
        x
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun min(x: Integer, y: Integer): Integer {
    if (x.lt(y)) x else y
}

#[mode(spec), ext(spec_only, pure)]
public fun max(x: Integer, y: Integer): Integer {
    if (x.gt(y)) x else y
}

#[mode(spec), ext(spec_only, pure)]
public fun div_trunc(x: Integer, y: Integer): Integer {
    let result_abs = x.abs().div(y.abs());
    if (x.is_pos() && y.is_pos() || x.is_neg() && y.is_neg()) {
        result_abs
    } else {
        result_abs.neg()
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun mod_trunc(x: Integer, y: Integer): Integer {
    x.sub(y.mul(x.div_trunc(y)))
}

#[mode(spec), ext(spec_only, pure)]
public fun is_pos(x: Integer): bool {
    x.gte(0u64.to_int())
}

#[mode(spec), ext(spec_only, pure)]
public fun is_neg(x: Integer): bool {
    x.lt(0u64.to_int())
}

const I8_MIN_AS_U8: u8 = 0x80;
const I8_MAX_AS_U8: u8 = 0x7f;

#[mode(spec), ext(spec_only, pure)]
public fun signed_from_u8(x: u8): Integer {
    if (x <= I8_MAX_AS_U8) {
        x.to_int()
    } else {
        x.to_int().sub(std::u8::max_value!().to_int()).sub(1u64.to_int())
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun is_i8(x: Integer): bool {
    x.gte(I8_MIN_AS_U8.to_signed_int()) && x.lte(I8_MAX_AS_U8.to_signed_int())
}

const I16_MIN_AS_U16: u16 = 0x8000;
const I16_MAX_AS_U16: u16 = 0x7fff;

#[mode(spec), ext(spec_only, pure)]
public fun signed_from_u16(x: u16): Integer {
    if (x <= I16_MAX_AS_U16) {
        x.to_int()
    } else {
        x.to_int().sub(std::u16::max_value!().to_int()).sub(1u64.to_int())
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun is_i16(x: Integer): bool {
    x.gte(I16_MIN_AS_U16.to_signed_int()) && x.lte(I16_MAX_AS_U16.to_signed_int())
}

const I32_MIN_AS_U32: u32 = 0x80000000;
const I32_MAX_AS_U32: u32 = 0x7fffffff;

#[mode(spec), ext(spec_only, pure)]
public fun signed_from_u32(x: u32): Integer {
    if (x <= I32_MAX_AS_U32) {
        x.to_int()
    } else {
        x.to_int().sub(std::u32::max_value!().to_int()).sub(1u64.to_int())
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun is_i32(x: Integer): bool {
    x.gte(I32_MIN_AS_U32.to_signed_int()) && x.lte(I32_MAX_AS_U32.to_signed_int())
}

const I64_MIN_AS_U64: u64 = 0x8000000000000000;
const I64_MAX_AS_U64: u64 = 0x7fffffffffffffff;

#[mode(spec), ext(spec_only, pure)]
public fun signed_from_u64(x: u64): Integer {
    if (x <= I64_MAX_AS_U64) {
        x.to_int()
    } else {
        x.to_int().sub(std::u64::max_value!().to_int()).sub(1u64.to_int())
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun is_i64(x: Integer): bool {
    x.gte(I64_MIN_AS_U64.to_signed_int()) && x.lte(I64_MAX_AS_U64.to_signed_int())
}

const I128_MIN_AS_U128: u128 = 0x80000000000000000000000000000000;
const I128_MAX_AS_U128: u128 = 0x7fffffffffffffffffffffffffffffff;

#[mode(spec), ext(spec_only, pure)]
public fun signed_from_u128(x: u128): Integer {
    if (x <= I128_MAX_AS_U128) {
        x.to_int()
    } else {
        x.to_int().sub(std::u128::max_value!().to_int()).sub(1u64.to_int())
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun is_i128(x: Integer): bool {
    x.gte(I128_MIN_AS_U128.to_signed_int()) && x.lte(I128_MAX_AS_U128.to_signed_int())
}

const I256_MIN_AS_U256: u256 = 0x8000000000000000000000000000000000000000000000000000000000000000;
const I256_MAX_AS_U256: u256 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff;

#[mode(spec), ext(spec_only, pure)]
public fun signed_from_u256(x: u256): Integer {
    if (x <= I256_MAX_AS_U256) {
        x.to_int()
    } else {
        x.to_int().sub(std::u256::max_value!().to_int()).sub(1u64.to_int())
    }
}

#[mode(spec), ext(spec_only, pure)]
public fun is_i256(x: Integer): bool {
    x.gte(I256_MIN_AS_U256.to_signed_int()) && x.lte(I256_MAX_AS_U256.to_signed_int())
}
