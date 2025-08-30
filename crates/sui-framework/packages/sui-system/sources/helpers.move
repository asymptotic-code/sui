module sui_system::helpers;

#[spec_only]
use std::{u64,u32,u16,u8};
#[spec_only]
use prover::prover::{ensures, invariant};

#[spec_only]
/// x+y will not abort (u64)
public fun can_add_u64(x: u64, y: u64): bool {
    x.to_int().add(y.to_int()).lte(u64::max_value!().to_int())
}

#[spec_only]
/// x+y will not abort (u32)
public fun can_add_u32(x: u32, y: u32): bool {
    x.to_int().add(y.to_int()).lte(u32::max_value!().to_int())
}

#[spec_only]
/// x-y will not abort
public fun can_sub_u64(x: u64, y: u64): bool {
    y <= x
}

#[spec_only]
/// x-y will not abort
public fun can_sub_u32(x: u32, y: u32): bool {
    y <= x
}

// iteration macros have expansions that wrap up the contained loop,
// so that an invariant cannot be attached.  the versions here have a
// parameter that will give the loop invarant, which gets placed in
// the correct spot.

/// like `std::macros::range_do`, but allows for an invariant.
/// The supplied function takes one parameter, the loop index
public macro fun range_do_with_invariant<$T, $R: drop>($start: $T, $stop: $T, $f: |$T| -> $R, $g: |$T| -> bool) {
    let mut i = $start;
    let stop = $stop;
    invariant!(|| { ensures( $g(i) ); });
    while (i < stop) {
        $f(i);
        i = i + 1;
    }
}

/// like `std::macros::do`, but allows for an invariant.
/// The supplied function takes one parameter, the loop index
public macro fun do_with_invariant<$T, $R: drop>($stop: $T, $f: |$T| -> $R, $g: |$T| -> bool) {
    range_do_with_invariant!(0, $stop, $f, $g)
}

/// like `std:vectors::do`, but allows for an invariant.
/// The supplied function takes one parameter, the loop index
/// (which is also used as the index into the vector in the loop)
public macro fun vector_do_ref_with_invariant<$T, $R: drop>($v: &vector<$T>, $f: |&$T| -> $R, $g: |u64| -> bool) {
    let v = $v;
    do_with_invariant!(v.length(), |i| $f(&v[i]), |i| i <= v.length() && ($g(i)))
}

/// like `std::vectors::find_index`, but allows for an invariant.
public macro fun vector_find_index_with_invariant<$T>($v: &vector<$T>,
    $f: |&$T| -> bool,
    $g: |u64| -> bool): Option<u64> {
    let v = $v;
    'find_index: {
        do_with_invariant!(v.length(),
            (|i| if ($f(&v[i])) return 'find_index std::option::some(i)), // do this
            (|i|  i <= v.length() && ($g(i)))); // using this invariant
        std::option::none()
    }
}
