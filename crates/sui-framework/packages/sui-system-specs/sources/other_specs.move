module sui_system_specs::other_specs;

#[spec(target = sui::tx_context::epoch)]
public fun epoch_spec(self: &TxContext): u64 {
    sui::tx_context::epoch(self)
}

#[spec(target = sui::tx_context::sender)]
public fun sender_spec(self: &TxContext): address {
    sui::tx_context::sender(self)
}
