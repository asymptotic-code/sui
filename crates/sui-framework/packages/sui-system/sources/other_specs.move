/// Specifications for modules used by sui_system, and in particular for functions
/// that cause problems.

module sui_system::other_specs;

use sui::tx_context::TxContext;
use sui::priority_queue;
use std::u64;

use prover::prover::{drop,ensures,requires,val};

#[spec(target=sui::tx_context::epoch)]
public fun epoch_spec(self: &TxContext): u64 {
    sui::tx_context::epoch(self)
}

// see sui-prover issue #219.  When that's fixed, these two items can be removed
native fun tx_context_sender_sf(self: &TxContext): address;

#[spec(target=sui::tx_context::sender)]
public fun sender_spec(self: &TxContext): address {
    let r = sui::tx_context::sender(self);
    ensures(r == tx_context_sender_sf(self));
    r
}


#[spec(target=sui::priority_queue::new)]
public fun new_spec<T: drop>(mut entries: vector<priority_queue::Entry<T>>): priority_queue::PriorityQueue<T> {
    requires(entries.length() < u64::max_value!() - 1);
    priority_queue::new(entries)
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

// specs for Versioned

use sui::versioned::{Self,Versioned,VersionChangeCap};

#[spec(target=sui::versioned::create)]
public fun create_spec<T: store>(init_version: u64, init_value: T, ctx: &mut TxContext): Versioned {
    let r = versioned::create(init_version, init_value, ctx);
    //ensures(versioned_has_version_of_type<T>(&r));
    r
}

#[spec(target=sui::versioned::load_value)]
public fun load_value_spec<T: store>(self: &Versioned): &T {
    // requires(versioned_has_version_of_type<T>(self)); // can fail if the type T is incorrect
    versioned::load_value(self)
}

// Actually, can fail if the type T is incorrect
#[spec(target=sui::versioned::load_value_mut)]
public fun load_value_mut_spec<T: store>(self: &mut Versioned): &mut T {
    // requires(versioned_has_version_of_type<T>(self));
    versioned::load_value_mut(self)
}

// Actually, can fail if the type T is incorrect
#[spec(target=sui::versioned::remove_value_for_upgrade)]
public fun remove_value_for_upgrade_spec<T: store>(self: &mut Versioned): (T, VersionChangeCap) {
    // requires(versioned_has_version_of_type<T>(self));
    versioned::remove_value_for_upgrade(self)
}

#[spec(target=sui::versioned::upgrade)]
public fun upgrade_spec<T: store>(
    self: &mut Versioned,
    new_version: u64,
    new_value: T,
    cap: VersionChangeCap) {
        // requires(versioned_has_version_of_type<T>(self)); // can fail if the type T is incorrect
        // requires self and cap have the same id; not expressible here
        requires(self.version() < new_version); // or EInvalidUpgrade
        versioned::upgrade(self, new_version, new_value, cap)
    }

// Actually, can fail if the type T is incorrect
#[spec(target=sui::versioned::destroy)]
public fun destroy_spec<T: store>(self: Versioned): T {
    // requires(versioned_has_version_of_type<T>(self));
    versioned::destroy(self)
}
