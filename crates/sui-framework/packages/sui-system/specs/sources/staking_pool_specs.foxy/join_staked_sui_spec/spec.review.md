# Spec Review: `staking_pool_specs::join_staked_sui_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has no `ensures` verifying the core merge accounting: `self.principal.value()` after the call must equal `old(self.principal.value()) + old(other.principal.value())`. This is the central correctness property of `join_staked_sui` and is entirely unverified — an implementation that dropped, double-counted, or mis-summed the joined principal would still pass. The overflow assert already computes this exact sum in Integer space, so capturing the two `old` amounts and adding one equality `ensures` is straightforward.
- ○ **medium/high**: No frame `ensures` that `self.pool_id` and `self.stake_activation_epoch` are unchanged. `self` is a `&mut StakedSui`, so the prover does not guarantee these fields are preserved across the call; an implementation overwriting `self`'s metadata from `other` would go undetected. Both should be asserted equal to their pre-call (`clone!`) values.
- ▫ **low/medium**: The minimum-stake invariant `self.principal.value() >= MIN_STAKING_THRESHOLD` (1 SUI) is not asserted post-join. It follows trivially from the input invariant but is the protocol-level property this function must preserve; making it explicit documents and guards the invariant.

## Analysis

The target `join_staked_sui(self, other)` does three observable things:
1. Aborts if `is_equal_staking_metadata(self, &other)` is false (mismatched pool_id or stake_activation_epoch -> EIncompatibleStakedSui).
2. Destroys `other` (deletes its UID, extracts its principal Balance<SUI>).
3. Joins `other.principal` into `self.principal`.

ABORT COVERAGE
- The single abort path is `!is_equal_staking_metadata(self, &other)`. The spec correctly inverts it: `asserts(staking_pool::is_equal_staking_metadata(self, &other))`. This is exact and complete.
- The spec adds a second assert bounding the combined principal: `staked_sui_amount(self).to_int().add(staked_sui_amount(&other).to_int()).lte(u64::MAX)`. The source's `self.principal.join(principal)` is a `Balance::join` which sums u64 values; the prover models this as a potential overflow abort, so this assert is needed to discharge the SpecNoAbort/Check abort proof. It is a defensive precondition rather than a real-world reachable abort (combined principal exceeding ~1.8e19 MIST is beyond total SUI supply), but it is correctly placed BEFORE the call. No abort path is missing.

POSTCONDITION COVERAGE — THIS IS THE MAIN GAP
The spec has ZERO `ensures`. It is purely an abort/precondition spec. Every behavioral guarantee from the writeup is unverified:
- `self.principal.value() == old(self.principal.value()) + old(other.principal.value())` — the central correctness property of a merge. Not checked. A buggy implementation that, say, dropped `other`'s principal, double-counted it, or wrote the wrong sum would still pass this spec. Given the overflow assert already computes exactly this sum in Integer space, capturing `old` values and adding one `ensures(staked_sui_amount(self).to_int() == old_self_amt.add(old_other_amt))` would be cheap and is the single most valuable addition.
- `self.pool_id` unchanged and `self.stake_activation_epoch` unchanged — frame conditions on the mutated `&mut StakedSui`. Not checked. Since `self` is a mutable reference, the prover does not guarantee these fields are preserved across the opaque-free call; an implementation that overwrote `self.pool_id` from `other` would go undetected.
- `self.principal.value() >= MIN_STAKING_THRESHOLD` (min-stake invariant preservation) — not checked. This follows trivially from the inputs but is the protocol-level invariant this function must not break.

The spec is consistent with the implementation (it passes the prover) and correctly models the only abort, but it verifies essentially nothing about WHAT the function computes. For a merge function whose entire purpose is balance accounting, the absence of the sum `ensures` is a real correctness gap, not a stylistic one.

NOTES
- `@VERIFY(🛡️/✅)` marks this as having an abort-modeling concern resolved; the overflow assert is the documented handling.
- This is a client-side object merge with no StakingPool state touched, so no pool-side frame conditions are needed — that part of the writeup is correctly reflected by the spec's narrow scope.

## Strengths

- The sole abort condition (`!is_equal_staking_metadata`) is modeled exactly and correctly inverted as an `asserts`.
- The combined-principal overflow assert is placed correctly before the call and uses `.to_int()` arbitrary-precision arithmetic, properly discharging the `Balance::join` overflow proof obligation.
- Scope is correctly narrow — it recognizes this is a client-side object merge with no StakingPool state changes, so it avoids spurious pool-side frame conditions.
