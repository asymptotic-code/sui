module sui_system_specs::stake_subsidy;

#[spec_only]
use sui_system::stake_subsidy::{
    StakeSubsidy,
    current_epoch_subsidy_amount
};

// This invariant is needed to prove no abort in some of the non-public functions,
// so can be skipped for now.

// #[spec_only]
// fun StakeSubsidy_inv(x: &StakeSubsidy): bool {
//     x.stake_subsidy_decrease_rate <= BASIS_POINT_DENOMINATOR as u16 &&
//     x.stake_subsidy_period_length > 0
// }

#[spec(prove, no_opaque, target = sui_system::stake_subsidy::current_epoch_subsidy_amount)]
public fun current_epoch_subsidy_amount_spec(self: &StakeSubsidy): u64 {
    current_epoch_subsidy_amount(self)
}
