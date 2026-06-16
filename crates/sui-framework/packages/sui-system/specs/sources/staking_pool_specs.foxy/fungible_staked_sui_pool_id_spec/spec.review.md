# Spec Review: `staking_pool_specs::fungible_staked_sui_pool_id_spec`

**Verdict:** △ Has Gaps

## Issues

- ○ **medium/high**: The spec invokes staking_pool::fungible_staked_sui_pool_id and returns its result, but adds no ensures(result == fungible_staked_sui.pool_id) postcondition. For a getter, the value contract is the only thing worth verifying; without it the spec proves nothing beyond abort-freedom (which is trivially true here). Because no_opaque is set the prover inlines the body, so callers still see result == field via inlining, but the spec itself does not assert the getter's defining property. Adding the explicit ensures (using a test_only/spec_only accessor for the private pool_id field) would make the spec meaningful rather than a tautology.

## Analysis

**Target.** `fungible_staked_sui_pool_id(&FungibleStakedSui): ID` is a one-line field accessor: `fungible_staked_sui.pool_id`. It takes an immutable reference, performs no arithmetic, mutates no state, and has no abort paths. The writeup correctly classifies it as low-complexity with empty aborts/requires and a single ensures: `result == fungible_staked_sui.pool_id`.

**Aborts.** None. The function cannot abort (pure copy of a `copy`-able `ID` out of an immutable reference), so the absence of `asserts` is correct and complete. There is nothing to cover here.

**Requires.** None needed — any well-typed `FungibleStakedSui` is a valid input. Correct.

**Ensures.** This is the one substantive gap. The writeup's only ensures — that the result equals the stored `pool_id` — is *not* present in the spec body. The spec calls the function and returns its value but states no postcondition. The spec is consistent with the implementation and passes the prover, but it verifies nothing about the return value: it is effectively a tautology (call returns what call returns). Because `no_opaque` is set, the prover inlines the body when verifying *callers*, so downstream specs that rely on `fungible_staked_sui.pool_id()` still get `result == field` by inlining — which is presumably why this getter spec was left value-free. That makes the omission low-impact in practice, but the spec itself does not assert the getter's defining property, so I flag it as a medium/high gap rather than dismissing it. Adding `ensures(result == module::get_pool_id(fungible_staked_sui))` via a `#[test_only, ext(pure)]` accessor (since `pool_id` is a private field) would turn the spec from a no-op into a real contract.

**Collections / partial mutations.** N/A — no collection, no mutation, no blast radius to bound.

**Note on `no_opaque`.** Appropriate for a trivial getter: it keeps the body inline for callers so the field-projection fact propagates without needing the getter's `$pure` to be opaque. Not an issue.

**Observation.** The sibling `pool_id(&StakedSui): ID` has identical structure; the same observation (missing value ensures, mitigated by inlining) and the same fix apply to its spec.

## Strengths

- Correctly models a no-abort getter: no spurious asserts or requires are added where none are needed.
- Uses no_opaque appropriately so the field-projection fact stays inline for callers of the getter.
- Faithfully mirrors the function signature and target; the spec is consistent with the implementation and passes all three checks.
