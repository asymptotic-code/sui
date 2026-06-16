# Spec Review: `staking_pool_specs::is_preactive_spec`

**Verdict:** ✔ Complete

No issues found — spec coverage is complete.

## Analysis

The target `is_preactive(pool: &StakingPool): bool` is a pure boolean projection of a single `Option` field: it returns `pool.activation_epoch.is_none()`. The writeup lists no aborts and no requires, which matches the implementation exactly — there is no arithmetic, no table access, no borrow, and no branching that could abort. The spec correctly omits any `asserts`/`requires`.

Ensures coverage: the writeup's primary postcondition is `result == pool.activation_epoch.is_none()`, and the spec encodes precisely this:
    ensures(result == staking_pool::activation_epoch(pool).is_none());
Here `staking_pool::activation_epoch(pool)` is the public(package) getter returning `pool.activation_epoch`, so `activation_epoch(pool).is_none()` is semantically identical to the implementation's `pool.activation_epoch.is_none()`. This is an exact equality on the boolean result — the strongest form available — not a loose inequality.

The writeup's other two ensures (`result == true implies activation_epoch == none()` and `result == false implies activation_epoch.is_some()`) are logically redundant given the biconditional already stated: `result == activation_epoch.is_none()` entails both directions (`is_none()` and its negation `is_some()`). The single biconditional ensures is therefore complete — adding the two implications would be strictly weaker restatements of the same fact, so their absence is not a gap.

State changes: `pool` is an immutable reference (`&StakingPool`), so there are no writes to verify and no frame conditions needed — the type system guarantees field preservation. No `ignore_abort` or `no_opaque` attributes are present; the spec is proved in full `prove` mode across all three checks (_Check, _Assume vacuous since no asserts, _SpecNoAbortCheck trivial since the spec body only calls the pure getter).

The spec is both complete and correct for this getter — it captures exactly the intended behavior with the tightest possible postcondition and no spurious checks.

## Strengths

- Encodes the exact biconditional `result == activation_epoch.is_none()`, which subsumes both directional implications from the writeup in a single tight postcondition.
- Correctly omits asserts/requires: the getter cannot abort, so abort modeling would be spurious.
- Uses the public `activation_epoch(pool)` getter rather than a hand-rolled field access, keeping the spec aligned with the module's accessor surface.
