module root::use_all;

public fun tags(): u8 { a::registry::tag() + b::registry::tag() + c::registry::tag() }
