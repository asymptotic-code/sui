/// Specifications for modules used by sui_system, and in particular for functions
/// that cause problems.

module sui_system::other_specs;

use sui::tx_context::TxContext;
use std::option::{Option,some,none};
use prover::prover::{drop,ensures,requires,val};

#[spec(target=sui::tx_context::epoch)]
public fun epoch_spec(self: &TxContext): u64 {
    sui::tx_context::epoch(self)
}

#[spec(target=sui::tx_context::sender)]
public fun sender_spec(self: &TxContext): address {
    sui::tx_context::sender(self)
}

#[spec(target=std::option::destroy_some)] // see sui-prover issue #144
public fun destroy_some_spec<Element>(t: Option<Element>): Element {
    prover::prover::requires(t.is_some());
    let i = prover::prover::val(t.borrow());
    let r = t.destroy_some();
    prover::prover::ensures(&r == i);
    prover::prover::drop(i);
    r
}

// #[spec_only(inv_target=std::option::Option)] // see sui-prover issue #144
// fun Option_inv<T>(self: &Option<T>): bool {
//     if (self.is_some()) {
//         let o = val(self.borrow());
//         let x = some(o);
//         let b = self == x;
//         drop(x);
//         b
//     } else {
//         true
//     }
// }
