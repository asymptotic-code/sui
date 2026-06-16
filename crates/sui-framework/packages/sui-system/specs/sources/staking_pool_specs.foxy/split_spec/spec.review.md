# Spec Review: `staking_pool_specs::split_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has zero ensures. None of the postconditions are verified: that the returned StakedSui has principal exactly equal to split_amount (result.principal.value() == split_amount), that self.principal.value() == old(self.principal.value()) - split_amount, and the principal conservation invariant self.principal.value() + result.principal.value() == old(self.principal.value()). The summary explicitly names principal conservation as the key invariant to verify, yet the spec proves nothing about the function's actual effect. As written, an implementation that split off the wrong amount (or corrupted the remaining balance) would still pass.
- ○ **medium/high**: Metadata-preservation postconditions are unverified. The function copies pool_id and stake_activation_epoch into the new StakedSui and leaves self's pool_id / stake_activation_epoch unchanged. The spec should ensure result.pool_id == old(self.pool_id), result.stake_activation_epoch == old(self.stake_activation_epoch), self.pool_id == old(self.pool_id), and self.stake_activation_epoch == old(self.stake_activation_epoch). A bug copying the wrong field into the new object would go undetected.

## Analysis

`split_spec` targets `staking_pool::split` with `no_opaque` and covers the abort surface completely but verifies none of the function's effects.

Abort coverage (correct and complete):
- `asserts(split_amount <= original_amount)` matches the source `assert!(split_amount <= original_amount, EInsufficientSuiTokenBalance)`.
- `asserts(original_amount.to_int().sub(split_amount.to_int()).gte(MIN_STAKING_THRESHOLD.to_int()))` matches `assert!(remaining_amount >= MIN_STAKING_THRESHOLD, EStakedSuiBelowThreshold)`. Using `.to_int().sub(...)` here is the right choice — it sidesteps the u64 underflow that a naive `original_amount - split_amount` spec expression would hit, while the first assert already guarantees `split_amount <= original_amount` so the subtraction is semantically safe.
- `asserts(split_amount >= MIN_STAKING_THRESHOLD)` matches the third source assertion.

All three abort conditions in the writeup are present and correctly inverted, and there are no spurious asserts. The `_Assume` direction (these asserts are exactly the non-abort condition) is meaningful here since the spec is not `ignore_abort`.

Effect coverage (the gap): the spec ends with `staking_pool::split(self, split_amount, ctx)` and adds no `ensures`. This is a self-custody object split with no StakingPool accounting side-effects, so the entire correctness content lives in a handful of simple equalities — principal of the new object, decremented principal of self, the conservation identity, and copied/unchanged metadata. Because `self` is `&mut StakedSui` and a brand new StakedSui is returned, none of these are guaranteed by the type system; they must be stated explicitly. The field accessors exist (`staked_sui_amount` for principal, `pool_id`, `stake_activation_epoch` / `activation_epoch`), so the ensures are straightforward to write:
  let old_amount = staking_pool::staked_sui_amount(self);
  let old_pool_id = staking_pool::pool_id(self);
  let old_epoch = staking_pool::stake_activation_epoch(self);
  let result = staking_pool::split(self, split_amount, ctx);
  ensures(staking_pool::staked_sui_amount(&result) == split_amount);
  ensures(staking_pool::staked_sui_amount(self) == old_amount - split_amount);
  ensures(staking_pool::staked_sui_amount(self) + staking_pool::staked_sui_amount(&result) == old_amount);
  ensures(staking_pool::pool_id(&result) == old_pool_id);
  ensures(staking_pool::pool_id(self) == old_pool_id);
  ensures(staking_pool::stake_activation_epoch(&result) == old_epoch);
  ensures(staking_pool::stake_activation_epoch(self) == old_epoch);

Without these, the spec passes the prover (it is consistent with the implementation) but provides false confidence — it establishes only that the function aborts under the right conditions, not that it does the right thing when it doesn't abort.

## Strengths

- Abort conditions are complete and correctly inverted, including all three EStakedSuiBelowThreshold / EInsufficientSuiTokenBalance paths.
- Uses Integer arithmetic (`.to_int().sub(...).gte(...)`) for the remaining-balance threshold check, correctly avoiding a spurious u64 underflow in the spec expression.
- Captures `original_amount` once via the `staked_sui_amount` accessor and reuses it across asserts.
