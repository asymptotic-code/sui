module root_pkg::both;

public fun sum(): u64 { pyth::price::version() + pyth_upgraded::price::version() }
