# Spec Review: `staking_pool_specs::split_staked_sui_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The new StakedSui transferred to ctx.sender() is completely unverified. None of its three properties are ensured: principal.value() == split_amount, pool_id == original stake.pool_id, stake_activation_epoch == original stake.stake_activation_epoch. This is the function's entire output — a buggy split could give the new receipt the wrong principal, wrong pool, or wrong activation epoch and the spec would still pass. The conservation property (the two halves' principals sum to the original) is therefore also unverified.
- ● **high/high**: The transfer destination is not verified. The spec declares SpecTransferAddressExists and SpecTransferAddress ghosts but never reads them in any ensures. There is no check that a transfer actually occurred or that the recipient equals ctx.sender(). The newly split StakedSui could be sent to the wrong address without detection. Add ensures(*ghost::global<SpecTransferAddressExists, bool>()) and ensures(*ghost::global<SpecTransferAddress, address>() == ctx.sender()).

## Analysis

## Target

`split_staked_sui(stake, split_amount, ctx)` is a thin entry wrapper:
```move
transfer::transfer(stake.split(split_amount, ctx), ctx.sender());
```
It calls `split` (which mutates `stake.principal` down by `split_amount` and produces a new `StakedSui` carrying `split_amount`, same `pool_id` and `stake_activation_epoch`), then transfers that new object to `ctx.sender()`.

## Abort coverage

The spec's three `asserts` exactly mirror the three `assert!` checks inside `split` (inverted):
- `assert!(split_amount <= original_amount, EInsufficientSuiTokenBalance)` -> `asserts(split_amount <= original_amount)` ✓
- `assert!(remaining_amount >= MIN_STAKING_THRESHOLD, EStakedSuiBelowThreshold)` -> `asserts(original_amount - split_amount >= MIN_STAKING_THRESHOLD)` ✓ (uses `.to_int().sub(...).gte(...)`, the correct underflow-safe form)
- `assert!(split_amount >= MIN_STAKING_THRESHOLD, EStakedSuiBelowThreshold)` -> `asserts(split_amount >= MIN_STAKING_THRESHOLD)` ✓

`object::new(ctx)` and the `transfer::transfer` do not introduce additional reachable abort conditions, so abort modeling is complete. The `_Assume` direction (these three asserts are *exactly* the non-abort condition) is sound because there are no other abort sites.

## Ensures coverage — the original `stake` (mutated in place)

All three writeup postconditions on the surviving `stake` are present and use exact equality:
- `ensures(staked_sui_amount(stake) == original_amount - split_amount)` ✓ — principal correctly reduced
- `ensures(pool_id(stake) == old_pool_id)` ✓ — pool unchanged
- `ensures(stake_activation_epoch(stake) == old_epoch)` ✓ — activation epoch unchanged

These fully verify that the in-place mutation only touches the principal and preserves the staking metadata.

## Gap — the newly created / transferred StakedSui is entirely unverified

This is the core weakness. The writeup explicitly lists three postconditions about the *new* `StakedSui` that is transferred to `ctx.sender()`:
- new principal value == `split_amount`
- new `pool_id` == old `stake.pool_id`
- new `stake_activation_epoch` == old `stake.stake_activation_epoch`

The spec verifies **none** of these. The spec declares two transfer-tracking ghosts:
```move
ghost::declare_global_mut<SpecTransferAddressExists, bool>();
ghost::declare_global_mut<SpecTransferAddress, address>();
```
but then never reads them — there is no `ensures(*ghost::global<SpecTransferAddressExists, bool>())` and no `ensures(*ghost::global<SpecTransferAddress, address>() == ctx.sender())`. So even the destination of the transfer is not asserted.

Because the new object is the entire *output* of this function (the whole point of the split-to-sender entry), leaving it unconstrained means a faulty implementation could:
- transfer a new `StakedSui` with the wrong principal (e.g. `original_amount` instead of `split_amount`),
- attach the wrong `pool_id` or `stake_activation_epoch`,
- transfer it to the wrong address,

and this spec would still pass. The conservation property (the two halves' principals sum to the original) is not captured at all — only the surviving half is checked.

Verifying the new object requires capturing it before the `transfer::transfer` consumes it. Since `split_staked_sui` itself does the transfer, the standard approach is to verify the inner `split` (which returns the new object directly — and indeed `split_spec` exists separately) and use the transfer-address ghost machinery here. Given the ghosts are already declared, the minimal fix is to add the `ensures` reading them (existence + recipient == `ctx.sender()`); the new object's field values are most naturally pinned via the `split` callee spec being `include`d, or via ghost tags on the construction.

## Note on run_on/local

The `run_on = b"local"` attribute and the `@VERIFY` comment ("cloud out-of-resources") are configuration, not correctness concerns.

## Strengths

- All three abort conditions of the inner split (EInsufficientSuiTokenBalance and both EStakedSuiBelowThreshold checks) are precisely mirrored as asserts, giving complete and sound abort modeling.
- The underflow-prone remaining-principal check uses original_amount.to_int().sub(split_amount.to_int()).gte(...) — correct arbitrary-precision form rather than a subtraction that could underflow.
- The surviving stake is verified with exact equality on all three relevant fields (reduced principal, unchanged pool_id, unchanged activation epoch), correctly bounding the in-place mutation.
