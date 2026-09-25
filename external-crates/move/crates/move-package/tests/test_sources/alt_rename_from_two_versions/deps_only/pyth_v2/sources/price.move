module pyth::price {
    public fun version(): u64 { wormhole::state::id() + 2 }
}
