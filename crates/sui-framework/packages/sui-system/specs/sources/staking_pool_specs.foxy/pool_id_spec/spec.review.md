# Spec Review: `staking_pool_specs::pool_id_spec`

**Verdict:** △ Has Gaps

## Issues

- ○ **medium/high**: The spec has no `ensures` at all, so it does not verify the function's only correctness property: that the returned `ID` equals the `StakedSui.pool_id` field. As written, the spec proves nothing about the return value (it passes only because `no_opaque` inlines the trivial body at call sites). Add `ensures(result == staked_sui.pool_id)` (via a spec-visible field accessor) so the projection is actually checked — this is the foundation for the EWrongPool pool-membership invariants used by request_withdraw_stake, withdraw_from_principal, and redeem_fungible_staked_sui.

## Analysis

`pool_id` is a pure field projection: `public fun pool_id(staked_sui: &StakedSui): ID { staked_sui.pool_id }`. It reads a single immutable `&StakedSui` reference and returns the `pool_id` field with no computation, no mutation, and no abort paths.

The writeup lists `aborts: []`, `requires: []`, and a single `ensures: result == staked_sui.pool_id`. That postcondition is the entire correctness contract for this accessor.

The actual spec calls `staking_pool::pool_id(staked_sui)` under `#[spec(prove, target=..., no_opaque)]` but contains **zero ensures, zero asserts, zero requires**. With `no_opaque`, the function body is inlined for callers, so the prover can still connect `pool_id(s)` to `s.pool_id` at call sites — which is why this passes. But the spec itself asserts nothing: it neither states nor checks that the returned ID equals the stored `pool_id` field. A spec that proves no property provides no verification value beyond confirming the function compiles and is abort-free.

There are correctly no abort checks to add (the function cannot abort — immutable field read), and no unchanged-field checks are warranted because the parameter is an immutable reference. The one missing piece is the single postcondition from the writeup.

Adding the field projection postcondition is straightforward. Because `pool_id` is itself the canonical accessor for the field, the ensures can be expressed against the captured input, e.g. `let r = staking_pool::pool_id(staked_sui); ensures(r == staking_pool::pool_id(staked_sui)); r` is tautological — instead bind a snapshot of the field via a `#[test_only]`/`#[spec_only]` accessor or compare against the input's field directly so the ensures is meaningful (e.g. `ensures(result == staked_sui.pool_id)` using a spec-visible field accessor). This is foundational: multiple callers (`request_withdraw_stake`, `withdraw_from_principal`, `redeem_fungible_staked_sui`) rely on pool-id equality for their `EWrongPool` membership checks, so a meaningful ensures here anchors those downstream proofs.

## Strengths

- Correctly targets the right function with a faithful signature and uses `no_opaque`, which is the appropriate flag for a trivial getter so callers see the inlined body.
- Correctly adds no spurious abort checks — the function reads an immutable reference and cannot abort.
