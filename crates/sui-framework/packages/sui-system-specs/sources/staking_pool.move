#[allow(unused_const)]
module sui_system_specs::staking_pool;

#[spec_only]
use sui_system::staking_pool::{
    PoolTokenExchangeRate,
    StakedSui,
    StakingPool,
    FungibleStakedSui,
    staked_sui_amount,
    is_preactive,
    is_inactive,
    split_fungible_staked_sui,
    join_fungible_staked_sui,
    is_equal_staking_metadata,
    split,
    split_staked_sui,
    pool_token_exchange_rate_at_epoch,
    join_staked_sui,
    fungible_staked_sui_value,
    activation_epoch,
};

#[spec_only]
use sui_system_specs::helpers::can_add_u64;

#[spec_only]
use prover::prover::{requires, ensures};

#[spec_only]
use prover::ghost;

/// CLONE.
#[spec_only]
const MIN_STAKING_THRESHOLD: u64 = 1_000_000_000; // 1 SUI

#[spec(prove, no_opaque, target = sui_system::staking_pool::staked_sui_amount)]
public fun staked_sui_amount_spec(staked_sui: &StakedSui): u64 {
    staked_sui_amount(staked_sui)
}

#[spec(prove, no_opaque, target = sui_system::staking_pool::is_preactive)]
public fun is_preactive_spec(pool: &StakingPool): bool {
    is_preactive(pool)
}

#[spec(prove, no_opaque, target = sui_system::staking_pool::is_inactive)]
public fun is_inactive_spec(pool: &StakingPool): bool {
    is_inactive(pool)
}

#[spec(prove, target = sui_system::staking_pool::split_fungible_staked_sui)]
public fun split_fungible_staked_sui_spec(
    fungible_staked_sui: &mut FungibleStakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
): FungibleStakedSui {
    requires(split_amount <= fungible_staked_sui_value(fungible_staked_sui));
    split_fungible_staked_sui(fungible_staked_sui, split_amount, ctx)
}

#[spec(prove, target = sui_system::staking_pool::join_fungible_staked_sui)]
public fun join_fungible_staked_sui_spec(self: &mut FungibleStakedSui, other: FungibleStakedSui) {
    requires(self.pool_id() == other.pool_id());
    requires(can_add_u64(self.value(), other.value()));
    join_fungible_staked_sui(self, other);
}

#[spec(prove, target = sui_system::staking_pool::split)]
public fun split_spec(self: &mut StakedSui, split_amount: u64, ctx: &mut TxContext): StakedSui {
    requires(split_amount <= staked_sui_amount(self));
    requires(split_amount >= MIN_STAKING_THRESHOLD);
    requires(staked_sui_amount(self) - split_amount >= MIN_STAKING_THRESHOLD);
    split(self, split_amount, ctx)
}

#[spec(prove, no_opaque, target = sui_system::staking_pool::is_equal_staking_metadata)]
public fun is_equal_staking_metadata_spec(self: &StakedSui, other: &StakedSui): bool {
    is_equal_staking_metadata(self, other)
}

#[spec_only]
/// function needed to make the requirements of `pool_token_exchange_rate_at_epoch_spec`
/// expressible outside this module.
public fun activation_epoch_is_positive(pool: &StakingPool): bool {
    let epoch_opt = activation_epoch(pool);
    epoch_opt.is_some() &&
    *epoch_opt.borrow() > 0
}

#[spec(prove, target = sui_system::staking_pool::pool_token_exchange_rate_at_epoch)]
public fun pool_token_exchange_rate_at_epoch_spec(
    pool: &StakingPool,
    epoch: u64,
): PoolTokenExchangeRate {
    requires(! pool.is_preactive());
    // requires(*pool.activation_epoch.borrow() > 0); // visible? add to invariant?
    requires(activation_epoch_is_positive(pool));
    pool_token_exchange_rate_at_epoch(pool, epoch)
}

#[spec(prove, target = sui_system::staking_pool::split_staked_sui)]
public fun split_staked_sui_spec(stake: &mut StakedSui, split_amount: u64, ctx: &mut TxContext) {
    use specs::transfer_spec::{SpecTransferAddress,SpecTransferAddressExists};
    ghost::declare_global_mut<SpecTransferAddressExists, bool>();
    ghost::declare_global_mut<SpecTransferAddress, address>();
    // preconditions from the call to split:
    requires(split_amount <= staked_sui_amount(stake));
    requires(split_amount >= MIN_STAKING_THRESHOLD);
    requires(staked_sui_amount(stake) - split_amount >= MIN_STAKING_THRESHOLD);
    split_staked_sui(stake, split_amount, ctx);
}

#[spec(prove, target = sui_system::staking_pool::join_staked_sui)]
public fun join_staked_sui_spec(self: &mut StakedSui, other: StakedSui) {
    requires(is_equal_staking_metadata(self, &other));
    requires(can_add_u64(staked_sui_amount(self), staked_sui_amount(&other))); // or the join overflows
    join_staked_sui(self, other);
}
