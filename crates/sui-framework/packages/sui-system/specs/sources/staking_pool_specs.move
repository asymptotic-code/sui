module specs::staking_pool_specs;

use sui::tx_context::TxContext;

use sui_system::staking_pool::PoolTokenExchangeRate;

use sui_system::staking_pool::StakingPool;

use sui_system::staking_pool::StakedSui;

use sui_system::staking_pool;
use sui_system::staking_pool::FungibleStakedSui;
use sui::object::ID;

#[spec_only]
use prover::prover::{asserts, ensures};
#[spec_only]
use prover::ghost;
#[spec_only]
use specs::transfer_spec::{SpecTransferAddress, SpecTransferAddressExists};

const MIN_STAKING_THRESHOLD: u64 = 1_000_000_000;

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::fungible_staked_sui_pool_id, no_opaque)]
fun fungible_staked_sui_pool_id_spec(
    fungible_staked_sui: &FungibleStakedSui,
): ID {
    staking_pool::fungible_staked_sui_pool_id(fungible_staked_sui)
}

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::fungible_staked_sui_value, no_opaque)]
fun fungible_staked_sui_value_spec(
    fungible_staked_sui: &FungibleStakedSui,
): u64 {
    staking_pool::fungible_staked_sui_value(fungible_staked_sui)
}

// @VERIFY(⚙️/✅)
#[spec(prove, target=staking_pool::is_equal_staking_metadata)]
fun is_equal_staking_metadata_spec(
    self: &StakedSui,
    other: &StakedSui,
): bool {
    let result = staking_pool::is_equal_staking_metadata(self, other);
    ensures(result == (
        (staking_pool::pool_id(self) == staking_pool::pool_id(other)) &&
        (staking_pool::stake_activation_epoch(self) == staking_pool::stake_activation_epoch(other))
    ));
    result
}

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::is_inactive)]
fun is_inactive_spec(
    pool: &StakingPool,
): bool {
    staking_pool::is_inactive(pool)
}

// @VERIFY(⚙️/✅)
#[spec(prove, target=staking_pool::is_preactive)]
fun is_preactive_spec(
    pool: &StakingPool,
): bool {
    let result = staking_pool::is_preactive(pool);
    ensures(result == staking_pool::activation_epoch(pool).is_none());
    result
}

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::join_fungible_staked_sui)]
fun join_fungible_staked_sui_spec(
    self: &mut FungibleStakedSui,
    other: FungibleStakedSui,
) {
    asserts(staking_pool::fungible_staked_sui_pool_id(self) == staking_pool::fungible_staked_sui_pool_id(&other));
    asserts(
        staking_pool::fungible_staked_sui_value(self).to_int()
            .add(staking_pool::fungible_staked_sui_value(&other).to_int())
            .lte(std::u64::max_value!().to_int()),
    );
    staking_pool::join_fungible_staked_sui(self, other)
}

#[spec(prove, target=staking_pool::join_staked_sui, ignore_abort)]
fun join_staked_sui_spec(
    self: &mut StakedSui,
    other: StakedSui,
) {
    staking_pool::join_staked_sui(self, other)
}

#[spec(prove, target=staking_pool::pending_stake_amount, ignore_abort)]
fun pending_stake_amount_spec(
    staking_pool: &StakingPool,
): u64 {
    staking_pool::pending_stake_amount(staking_pool)
}

#[spec(prove, target=staking_pool::pending_stake_withdraw_amount, ignore_abort)]
fun pending_stake_withdraw_amount_spec(
    staking_pool: &StakingPool,
): u64 {
    staking_pool::pending_stake_withdraw_amount(staking_pool)
}

#[spec(prove, target=staking_pool::pool_id, ignore_abort, no_opaque)]
fun pool_id_spec(
    staked_sui: &StakedSui,
): ID {
    staking_pool::pool_id(staked_sui)
}

#[spec(prove, target=staking_pool::pool_token_amount, ignore_abort)]
fun pool_token_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::pool_token_amount(exchange_rate)
}

#[spec(prove, target=staking_pool::pool_token_exchange_rate_at_epoch, ignore_abort)]
fun pool_token_exchange_rate_at_epoch_spec(
    pool: &StakingPool,
    epoch: u64,
): PoolTokenExchangeRate {
    staking_pool::pool_token_exchange_rate_at_epoch(pool, epoch)
}

#[spec(prove, target=staking_pool::split, ignore_abort, no_opaque)]
fun split_spec(
    self: &mut StakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
): StakedSui {
    staking_pool::split(self, split_amount, ctx)
}

#[spec(prove, target=staking_pool::split_fungible_staked_sui, ignore_abort)]
fun split_fungible_staked_sui_spec(
    fungible_staked_sui: &mut FungibleStakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
): FungibleStakedSui {
    staking_pool::split_fungible_staked_sui(fungible_staked_sui, split_amount, ctx)
}

// @VERIFY(⚙️/✅) cloud out-of-resources; verified locally via run_on
#[spec(prove, target=staking_pool::split_staked_sui, run_on = b"local")]
fun split_staked_sui_spec(
    stake: &mut StakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
) {
    ghost::declare_global_mut<SpecTransferAddressExists, bool>();
    ghost::declare_global_mut<SpecTransferAddress, address>();
    let original_amount = staking_pool::staked_sui_amount(stake);
    let old_pool_id = staking_pool::pool_id(stake);
    let old_epoch = staking_pool::stake_activation_epoch(stake);
    asserts(split_amount <= original_amount);
    asserts(original_amount.to_int().sub(split_amount.to_int()).gte(MIN_STAKING_THRESHOLD.to_int()));
    asserts(split_amount >= MIN_STAKING_THRESHOLD);
    staking_pool::split_staked_sui(stake, split_amount, ctx);
    ensures(staking_pool::staked_sui_amount(stake).to_int() == original_amount.to_int().sub(split_amount.to_int()));
    ensures(staking_pool::pool_id(stake) == old_pool_id);
    ensures(staking_pool::stake_activation_epoch(stake) == old_epoch);
}

#[spec(prove, target=staking_pool::stake_activation_epoch, ignore_abort, no_opaque)]
fun stake_activation_epoch_spec(
    staked_sui: &StakedSui,
): u64 {
    staking_pool::stake_activation_epoch(staked_sui)
}

#[spec(prove, target=staking_pool::staked_sui_amount, ignore_abort, no_opaque)]
fun staked_sui_amount_spec(
    staked_sui: &StakedSui,
): u64 {
    staking_pool::staked_sui_amount(staked_sui)
}

#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}

#[spec(prove, target=staking_pool::sui_balance, ignore_abort)]
fun sui_balance_spec(
    pool: &StakingPool,
): u64 {
    staking_pool::sui_balance(pool)
}
