/// Specifications for modules used by sui_system, and in particular for functions
/// that cause problems.

module sui_system::other_specs;

use sui::tx_context::TxContext;

#[spec(target=sui::tx_context::epoch)]
public fun epoch_spec(self: &TxContext): u64 {
    sui::tx_context::epoch(self)
}

#[spec(target=sui::tx_context::sender)]
public fun sender_spec(self: &TxContext): address {
    sui::tx_context::sender(self)
}
