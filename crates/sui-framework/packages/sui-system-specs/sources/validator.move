#[allow(unused_const)]
module sui_system_specs::validator;

#[spec_only]
use sui_system::staking_pool::{
    PoolTokenExchangeRate,
};

#[spec_only]
use sui_system::validator::{
    Validator,
    ValidatorMetadata,
    is_preactive,
    is_duplicate,
    validate_metadata,
    validate_metadata_bcs,
    staking_pool_id,
    pool_token_exchange_rate_at_epoch,
    pool_activation_epoch,
};
#[spec_only]
use sui_system::validator_cap::{Self, ValidatorOperationCap};

#[spec_only]
use prover::prover::requires;

#[spec_only]
public fun pool_activation_epoch_is_positive(validator: &Validator): bool {
    let epoch_opt = pool_activation_epoch(validator);
    epoch_opt.is_some() &&
    *epoch_opt.borrow() > 0
}

#[spec(prove, no_opaque, target = sui_system::validator::is_preactive)]
public fun is_preactive_spec(self: &Validator): bool {
    is_preactive(self)
}

#[spec(prove, target = sui_system::validator::pool_token_exchange_rate_at_epoch)]
public fun pool_token_exchange_rate_at_epoch_spec(self: &Validator, epoch: u64): PoolTokenExchangeRate {
    requires(!is_preactive(self));
    requires(pool_activation_epoch_is_positive(self));
    pool_token_exchange_rate_at_epoch(self, epoch)
}

#[spec(prove, no_opaque, target = sui_system::validator::staking_pool_id)]
public fun staking_pool_id_spec(self: &Validator): ID {
    staking_pool_id(self)
}

#[spec(prove, target = sui_system::validator::is_duplicate)]
public fun is_duplicate_spec(self: &Validator, other: &Validator): bool {
    is_duplicate(self, other)
}

#[spec(prove, target = sui_system::validator::validate_metadata)]
public fun validate_metadata_spec(metadata: &ValidatorMetadata) {
    validate_metadata(metadata);
}

// "aborts if metadata is not valid" -- how to capture?
#[spec(target = sui_system::validator::validate_metadata_bcs)]
public fun validate_metadata_bcs_spec(metadata: vector<u8>) {
    validate_metadata_bcs(metadata);
}
