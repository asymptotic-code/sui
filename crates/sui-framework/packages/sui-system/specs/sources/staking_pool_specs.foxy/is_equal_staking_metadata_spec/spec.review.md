# Spec Review: `staking_pool_specs::is_equal_staking_metadata_spec`

**Verdict:** ✔ Complete

No issues found — spec coverage is complete.

## Analysis

`is_equal_staking_metadata` is a pure read-only predicate over two `&StakedSui` references. Its entire behavior is the boolean conjunction `(self.pool_id == other.pool_id) && (self.stake_activation_epoch == other.stake_activation_epoch)`. The principal balance is deliberately excluded so that StakedSui receipts with the same pool and activation epoch but different principal can be merged in `join_staked_sui`.

Aborts: none. The function performs no table access, arithmetic, or assertion, so there are no abort paths and the spec correctly carries zero `asserts`. The `_Assume` and `_SpecNoAbortCheck` checks pass vacuously.

Requires: none expected, none present — correct.

Ensures coverage: the single `ensures` is an exact equality between `result` and the full boolean conjunction the implementation computes, expressed through the public accessors `pool_id(...)` and `stake_activation_epoch(...)`. This is the strongest possible postcondition (an iff over both fields), and it directly captures the writeup's primary requirement. Because the equality is exact, it also subsumes the two derived conditions in the writeup: if `pool_id(self) != pool_id(other)` the RHS is false so `result` must be false, and likewise for differing activation epochs. The accessors match the actual struct fields read by the implementation, so there is no risk of the spec verifying the wrong fields.

State preservation: both parameters are immutable references (`&StakedSui`), so the type system already guarantees no mutation — per review guidelines, asserting unchanged fields on immutable refs is not required and its absence is not a gap.

The spec is complete and correct for this trivial accessor. It uses public getters rather than raw field access, which keeps it robust to internal struct layout; the only subtle point is that for the equality to discharge, those accessors' `$pure` forms must remain inline (no conflicting accessor spec included), which is the case here since the getters are plain field reads.

## Strengths

- Postcondition is an exact iff over both compared fields, the strongest formulation possible — it subsumes the two derived 'differing field implies false' cases without needing separate clauses.
- Correctly carries zero asserts/requires, matching the function's abort-free, precondition-free nature.
- Expresses the comparison through public accessors (pool_id, stake_activation_epoch) that mirror exactly the fields the implementation reads, avoiding any wrong-field verification.
