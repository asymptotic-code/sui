# Spec Review: `staking_pool_specs::sui_balance_spec`

**Verdict:** △ Has Gaps

## Issues

- ○ **medium/high**: The spec has no `ensures` clause, so it verifies only that staking_pool::sui_balance is abort-free, not that the returned u64 equals pool.sui_balance. For a pure field getter the value-equality postcondition is the only substantive property, and it is missing. Add `ensures(result == pool.sui_balance);` (exposing the field via a #[test_only, ext(pure)] accessor if needed). This is the property the writeup specifies.

## Analysis

The target `staking_pool::sui_balance(pool: &StakingPool): u64` is a pure direct field projection — it reads and returns `pool.sui_balance` with no arithmetic, no mutation, and no abort paths. The writeup correctly lists zero aborts and zero requires, with a single ensures: `result == pool.sui_balance`.

**Coverage gap.** The spec is `#[spec(prove, target=staking_pool::sui_balance, no_opaque)]` with an empty body that just forwards to the target (0 asserts, 0 requires, 0 ensures). Because there is no `ensures(result == pool.sui_balance)`, the spec proves only that the function does not abort — it does NOT pin the return value to the field. For a getter the single meaningful property is exactly the value-equality postcondition, and it is absent.

**Why this still passes the prover.** With `no_opaque`, the target body is inlined, so the `_SpecNoAbortCheck` trivially holds (the projection cannot abort) and there is nothing else to discharge. Passing here therefore reflects only abort-freedom, not behavioral correctness. Callers that rely on this spec opaquely would get no guarantee linking the result to the field — though in practice `no_opaque` means callers inline the body anyway, which softens the impact.

**Field accessor caveat.** Note the system-prompt guidance that giving a getter a spec makes its `$pure` Boogie function uninterpreted; callers that need `sui_balance(pool)` to connect to `pool.sui_balance` may now require the explicit `ensures` to re-establish that link. Adding `ensures(result == pool.sui_balance)` both states the intended property and restores that connection. To access the private field from the spec, a `#[test_only, ext(pure)]` accessor (or the existing `sui_balance` getter, read with care to avoid the uninterpreted-$pure issue) is needed.

No frame conditions are required: the parameter is an immutable `&StakingPool`, so the type system already guarantees no field mutation.

## Strengths

- Correctly targets the getter and marks it no_opaque, which is the appropriate flag for a trivial field projection.
- Correctly omits asserts/requires — the writeup lists no abort or precondition obligations, matching the function's branch-free body.
