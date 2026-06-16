# Spec Review: `staking_pool_specs::is_inactive_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has no `ensures` clause. It targets `is_inactive` and calls it through, but never asserts `result == pool.deactivation_epoch.is_some()`. Without an `ensures`, the prover verifies nothing about the return value — the only thing checked is the (trivially true) abort-freedom of a pure getter. A buggy implementation that returned `false` unconditionally, or that tested `activation_epoch` instead of `deactivation_epoch`, would pass this spec unchanged. The whole point of specifying a predicate that callers branch on (gating stake operations, double-deactivation guards, fungible-conversion guards) is to give the prover the equation it can use opaquely; that equation is missing.
- ○ **medium/high**: The spec relies on the default opaque treatment but provides no postcondition, so any caller that `include`s this spec to reason about `pool.is_inactive()` gains nothing — the return value is fully havoced. For the predicate to be useful downstream (e.g. proving `request_add_stake` aborts with EDelegationToInactivePool, or that `request_withdraw_stake` takes the immediate-processing branch), it must either carry `ensures(result == pool.deactivation_epoch.is_some())` or be marked `no_opaque` so its one-line body is inlined.

## Analysis

`is_inactive` is a trivial read-only getter: `pool.deactivation_epoch.is_some()`. It takes an immutable `&StakingPool`, performs no mutation, and cannot abort, so there is nothing to model on the aborts or requires side — the empty `aborts: []` / `requires: []` in the plan is correct and the spec rightly omits asserts.

The gap is entirely on the ensures side. The plan lists four postconditions, all of which reduce to one equation: `result == pool.deactivation_epoch.is_some()` (the other three — the true/false witnesses and the no-mutation clause — are corollaries; no-mutation is also guaranteed by the `&` type). The actual spec contains zero `ensures`, so it proves none of them. The body `staking_pool::is_inactive(pool)` simply forwards the call; under the default opaque contract the prover havocs the boolean result and learns nothing, so the spec passes vacuously. This is the classic 'passes the prover but verifies nothing' situation: consistency with the implementation is satisfied trivially, while correctness (the return value actually tracks `deactivation_epoch`) is unchecked.

The fix is one line: add `ensures(result == pool.deactivation_epoch.is_some())` before returning (exposing the private field via a `#[test_only, ext(pure)]` accessor or relying on the inlined `$pure` body if `no_opaque` is used). Because the predicate is the linchpin that many callers branch on, getting this equation into the spec — transparently — is what makes the surrounding stake/withdraw/deactivate proofs possible.

## Strengths

- Correctly identifies that the function is abort-free and mutation-free — no spurious asserts or unchanged-field clauses on the immutable reference.
- Targets the function directly and forwards the real call, so whatever ensures are added will be checked against the true implementation.
