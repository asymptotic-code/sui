# Spec Review: `staking_pool_specs::pending_stake_amount_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec states no postcondition. The expected ensures `result == staking_pool.pending_stake` (the function's only observable behavior — returning the raw pending_stake field) is missing, so the spec verifies nothing about the return value. The _Check VC is effectively vacuous: it confirms the body does not abort but never ties `result` to the field, leaving the getter's core contract unverified. Add an ensures via a pure accessor on the private `pending_stake` field.

## Analysis

`pending_stake_amount` is a pure field projection: it returns `staking_pool.pending_stake` with no transformation, no branching, and no mutation. The writeup correctly classifies it as low complexity with no aborts and no requires, and lists a single postcondition: `result == staking_pool.pending_stake`.

**Aborts / requires:** None expected. A bare `u64` field read on an immutable reference cannot abort, so the absence of `asserts`/`requires` in the spec is correct and complete.

**Ensures coverage:** This is the one gap. The writeup's sole ensures (`result == staking_pool.pending_stake`) is NOT present in the spec body. The spec calls the target and returns its value but states no postcondition at all (0 ensures). Without the ensures, the _Check VC is vacuous — the prover confirms only that the body type-checks and does not abort, not that the returned value actually equals the field. A caller relying on this spec opaquely would learn nothing about the return value.

Because `pending_stake` is a private field, the ensures needs a `#[test_only, ext(pure)]` accessor (or the existing public `pending_stake_amount` getter, used carefully to avoid the `$pure`-opaque trap described in the accessor-spec guidance). The fix is a one-liner: `ensures(result == staking_pool::pending_stake_amount(staking_pool));` — though since that getter is the target itself, a private-field accessor is cleaner.

**Immutable reference:** Frame conditions are unnecessary here. The parameter is `&StakingPool` (immutable), so the type system already guarantees no field is mutated; no unchanged-field ensures are warranted.

The `@VERIFY(🛡️/✅)` marker and `no_opaque` are not present; this is a plain `#[spec(prove, target=...)]`. The spec passes the prover precisely because, with no ensures, there is nothing substantive to discharge.

## Strengths

- Correctly omits asserts/requires — a u64 field read on an immutable reference has no abort paths.
- Spec structure (target, signature, return-through) matches the trivial getter and passes all three checks.
