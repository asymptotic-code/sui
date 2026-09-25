module root::use_it {
    public fun f(): u64 { wormhole::state::id() }
}
