module sui_system_specs::storage_fund;

#[spec_only]
use sui_system::storage_fund::{
    StorageFund,
    total_object_storage_rebates,
    non_refundable_balance,
    total_balance,
};

#[spec_only]
use sui_system_specs::helpers::can_add_u64;

#[spec_only]
use prover::prover::requires;


#[spec(prove, no_opaque, target = sui_system::storage_fund::total_object_storage_rebates)]
public fun total_object_storage_rebates_spec(self: &StorageFund): u64 {
    total_object_storage_rebates(self)
}

#[spec(prove, target = sui_system::storage_fund::total_balance)]
public fun total_balance_spec(self: &StorageFund): u64 {
    requires(can_add_u64(total_object_storage_rebates(self), non_refundable_balance(self)));
    total_balance(self)
}
