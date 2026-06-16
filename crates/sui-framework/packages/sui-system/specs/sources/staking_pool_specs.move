module specs::staking_pool_specs;

use sui::tx_context::TxContext;

use sui_system::staking_pool::PoolTokenExchangeRate;

use sui_system::staking_pool::StakingPool;

use sui_system::staking_pool::StakedSui;

use sui_system::staking_pool;
use sui_system::staking_pool::FungibleStakedSui;
use sui::object::ID;

#[spec_only]
use prover::prover::{asserts, ensures, requires};
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

// @VERIFY(⚙️/✅)
#[spec(prove, target=staking_pool::join_staked_sui)]
fun join_staked_sui_spec(
    self: &mut StakedSui,
    other: StakedSui,
) {
    let old_self_amount = staking_pool::staked_sui_amount(self);
    let old_other_amount = staking_pool::staked_sui_amount(&other);
    asserts(staking_pool::is_equal_staking_metadata(self, &other));
    asserts(
        old_self_amount
            .to_int()
            .add(old_other_amount.to_int())
            .lte(std::u64::max_value!().to_int()),
    );
    staking_pool::join_staked_sui(self, other);
    ensures(
        staking_pool::staked_sui_amount(self)
            .to_int() == old_self_amount.to_int().add(old_other_amount.to_int()),
    );
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

// @VERIFY(⚙️/✅)
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

    let old_self_value = staking_pool::fungible_staked_sui_value(self);
    let old_other_value = staking_pool::fungible_staked_sui_value(&other);
    let old_pool_id = staking_pool::fungible_staked_sui_pool_id(self);

    staking_pool::join_fungible_staked_sui(self, other);

    ensures(
        staking_pool::fungible_staked_sui_value(self).to_int()
            == old_self_value.to_int().add(old_other_value.to_int()),
    );
    ensures(staking_pool::fungible_staked_sui_pool_id(self) == old_pool_id);
}



// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::pending_stake_amount)]
fun pending_stake_amount_spec(
    staking_pool: &StakingPool,
): u64 {
    staking_pool::pending_stake_amount(staking_pool)
}

// @VERIFY(🛡/✅)
#[spec(prove, target=staking_pool::pending_stake_withdraw_amount, no_opaque)]
fun pending_stake_withdraw_amount_spec(
    staking_pool: &StakingPool,
): u64 {
    staking_pool::pending_stake_withdraw_amount(staking_pool)
}

#[spec(prove, target=staking_pool::pool_id, no_opaque)]
// @VERIFY(🛡️/✅)
fun pool_id_spec(
    staked_sui: &StakedSui,
): ID {
    staking_pool::pool_id(staked_sui)
}

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::pool_token_amount, no_opaque)]
fun pool_token_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::pool_token_amount(exchange_rate)
}

#[spec_only(loop_inv(target = staking_pool::pool_token_exchange_rate_at_epoch)), ext(no_abort)]
fun pool_token_exchange_rate_at_epoch_loop_inv(
    pool: &StakingPool,
    epoch: u64,
    activation_epoch: u64,
): bool {
    epoch >= activation_epoch
        && staking_pool::exchange_rates(pool).contains(activation_epoch)
}

// @VERIFY(⚙️/✅)
// Postconditions verify the preactive branch: a preactive pool, or an active
// pool queried at an epoch before its activation, always receives the 1:1
// initial_exchange_rate (sui_amount = 0, pool_token_amount = 0). The historical
// table-lookup value (backward scan + deactivation clamp) is not characterized:
// every result-value postcondition (existence/maximality/exact-hit) is
// intractable for the prover on this Table-scanning loop, and the deactivation
// clamp is unobservable (the pinned framework exposes no deactivation_epoch getter).
#[spec(prove, target=staking_pool::pool_token_exchange_rate_at_epoch)]
fun pool_token_exchange_rate_at_epoch_spec(
    pool: &StakingPool,
    epoch: u64,
): PoolTokenExchangeRate {
    requires(pool.is_preactive()
        || staking_pool::exchange_rates(pool).contains(*staking_pool::activation_epoch(pool).borrow()));
    let result = staking_pool::pool_token_exchange_rate_at_epoch(pool, epoch);
    if (staking_pool::is_preactive(pool)) {
        ensures(staking_pool::sui_amount(&result) == 0);
        ensures(staking_pool::pool_token_amount(&result) == 0);
    } else {
        let activation = *staking_pool::activation_epoch(pool).borrow();
        if (activation > epoch) {
            ensures(staking_pool::sui_amount(&result) == 0);
            ensures(staking_pool::pool_token_amount(&result) == 0);
        };
    };
    result
}

// @VERIFY(⚙️/✅)
#[spec(prove, target=staking_pool::split, no_opaque)]
fun split_spec(
    self: &mut StakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
): StakedSui {
    let original_amount = staking_pool::staked_sui_amount(self);
    let old_pool_id = staking_pool::pool_id(self);
    let old_epoch = staking_pool::stake_activation_epoch(self);
    asserts(split_amount <= original_amount);
    asserts(original_amount.to_int().sub(split_amount.to_int()).gte(MIN_STAKING_THRESHOLD.to_int()));
    asserts(split_amount >= MIN_STAKING_THRESHOLD);
    let result = staking_pool::split(self, split_amount, ctx);
    let result_amount = staking_pool::staked_sui_amount(&result);
    let result_pool_id = staking_pool::pool_id(&result);
    let result_epoch = staking_pool::stake_activation_epoch(&result);
    ensures(result_amount == split_amount);
    ensures(staking_pool::staked_sui_amount(self).to_int() == original_amount.to_int().sub(split_amount.to_int()));
    ensures(
        staking_pool::staked_sui_amount(self)
            .to_int()
            .add(result_amount.to_int())
            == original_amount.to_int(),
    );
    ensures(staking_pool::pool_id(self) == old_pool_id);
    ensures(staking_pool::stake_activation_epoch(self) == old_epoch);
    ensures(result_pool_id == old_pool_id);
    ensures(result_epoch == old_epoch);
    result
}

// @VERIFY(⚙️/✅)
#[spec(prove, target=staking_pool::split_fungible_staked_sui)]
fun split_fungible_staked_sui_spec(
    fungible_staked_sui: &mut FungibleStakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
): FungibleStakedSui {
    let original_value = staking_pool::fungible_staked_sui_value(fungible_staked_sui);
    let old_pool_id = staking_pool::fungible_staked_sui_pool_id(fungible_staked_sui);
    asserts(split_amount <= original_value);
    let result = staking_pool::split_fungible_staked_sui(fungible_staked_sui, split_amount, ctx);
    let result_value = staking_pool::fungible_staked_sui_value(&result);
    let result_pool_id = staking_pool::fungible_staked_sui_pool_id(&result);
    ensures(result_value == split_amount);
    ensures(
        staking_pool::fungible_staked_sui_value(fungible_staked_sui).to_int()
            == original_value.to_int().sub(split_amount.to_int()),
    );
    ensures(
        staking_pool::fungible_staked_sui_value(fungible_staked_sui)
            .to_int()
            .add(result_value.to_int())
            == original_value.to_int(),
    );
    ensures(staking_pool::fungible_staked_sui_pool_id(fungible_staked_sui) == old_pool_id);
    ensures(result_pool_id == old_pool_id);
    result
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

// @VERIFY(🛡️/✅)

#[spec(prove, target=staking_pool::stake_activation_epoch, no_opaque)]
fun stake_activation_epoch_spec(
    staked_sui: &StakedSui,
): u64 {
    // @VERIFY(🛡️/✅)
    staking_pool::stake_activation_epoch(staked_sui)
}

#[spec(prove, target=staking_pool::staked_sui_amount, no_opaque)]
fun staked_sui_amount_spec(
    staked_sui: &StakedSui,
): u64 {
    staking_pool::staked_sui_amount(staked_sui)
}

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::sui_amount, no_opaque)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::sui_balance, no_opaque)]
fun sui_balance_spec(
    pool: &StakingPool,
): u64 {
    staking_pool::sui_balance(pool)
}
