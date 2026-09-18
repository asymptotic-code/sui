// `#[mode(spec)]` code is spec code: like `#[spec_only]`, it may call functions
// that are internal to another module. Outside spec mode it is filtered out.
module 0x42::m {
    fun secret(): u64 { 7 }
}

module 0x42::n {
    #[mode(spec), ext(spec_only)]
    fun helper(): u64 { 0x42::m::secret() }

    // still an error: not spec code
    public fun not_spec(): u64 { 0x42::m::secret() }
}
