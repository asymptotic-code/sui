# Spec Review: `staking_pool_specs::pending_stake_withdraw_amount_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has no `ensures` clause, so it verifies nothing about the return value. The writeup's sole postcondition `result == staking_pool.pending_total_sui_withdraw` is not encoded. Because the function reads a private field, add a `#[test_only, ext(pure)]` accessor (or spec-only getter) and then `ensures(result == staking_pool.pending_total_sui_withdraw())`. As written the spec would still pass even if the implementation returned a different field (e.g. `pending_stake`), creating false confidence.

## Analysis

The target `pending_stake_withdraw_amount` is a one-line pure field projection: it returns `staking_pool.pending_total_sui_withdraw`. The writeup confirms there are no abort conditions and no preconditions; the only stated postcondition is `result == staking_pool.pending_total_sui_withdraw`.

**Aborts:** None. A bare field read on an immutable `&StakingPool` reference cannot abort, so the absence of `asserts` is correct and complete. There is nothing to cover here.

**Postcondition coverage:** This is the one substantive gap. The spec body merely calls the target and returns its result without an `ensures` clause. Because the spec carries `no_opaque`, the prover inlines the function body, so the `_Check` proof trivially passes — but the spec verifies *nothing* about the return value. It does not assert `result == staking_pool.pending_total_sui_withdraw`. As written, this spec only establishes that the call type-checks and is abort-free; it provides no behavioral guarantee a caller could rely on, and it would still pass if the function were changed to return a different field (e.g. `pending_stake` or `pending_pool_token_withdraw`).

Since `pending_total_sui_withdraw` is a private field, exposing it in the `ensures` requires a `#[test_only, ext(pure)]` accessor in the implementation module (or a spec-only getter), after which the postcondition `ensures(result == staking_pool.pending_total_sui_withdraw())` should be added. With `no_opaque` the projection unfolds inline, so this is straightforward to discharge.

**Unchanged state:** The parameter is an immutable `&StakingPool`, so no frame conditions are needed — the type system guarantees the pool is not mutated.

## Strengths

- Correctly identifies the function as abort-free and omits unnecessary `asserts`.
- Uses `no_opaque`, the right choice for a trivial getter so the field projection unfolds inline.
- Targets the correct function with the proper `_spec` naming convention.
