# Spec Review: `staking_pool_specs::split_fungible_staked_sui_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: No ensures verifying value conservation: the spec never checks that old(fungible_staked_sui.value) == fungible_staked_sui.value + result.value. This is the headline correctness property of a split (no pool tokens created or destroyed). Capture the original value with clone! before the call and ensure the conservation identity.
- ● **high/high**: No ensures on the new object's value: result.value == split_amount is unverified. An implementation returning the wrong amount (e.g. 0) would pass the spec.
- ● **high/high**: No ensures on the original object's mutation: fungible_staked_sui.value == old(value) - split_amount is unverified. The decrement is the primary state change of the function and is left entirely unchecked.
- ○ **medium/high**: No ensures on pool membership preservation: result.pool_id == fungible_staked_sui.pool_id is unverified. The split is meant to keep both halves bound to the same pool; nothing confirms the new object copies the correct pool_id.

## Analysis

The target `split_fungible_staked_sui` is a simple, self-contained utility: it asserts `split_amount <= fungible_staked_sui.value`, deducts `split_amount` from the original's `value`, and packs a new `FungibleStakedSui { id: object::new(ctx), pool_id, value: split_amount }`. It has no callees, touches no `StakingPool` state, and the only abort source is the explicit `assert!(split_amount <= value, EInsufficientPoolTokenBalance)`.

ABORTS: The writeup lists one abort — `split_amount > fungible_staked_sui.value`. The spec covers this exactly via `asserts(split_amount <= staking_pool::fungible_staked_sui_value(fungible_staked_sui))` (the semantic inversion `>` → `<=`). No other abort paths exist (`object::new` does not abort under the prover's model), so abort coverage is complete and correct.

ENSURES: This is where the spec is materially incomplete — it has ZERO `ensures`. The writeup specifies five postconditions, none of which are verified:
1. `fungible_staked_sui.value == old(value) - split_amount` — the deduction on the original object is unverified. This is the core mutation of the function (`writes` on `FungibleStakedSui.value`) and is the single most important property to check.
2. `result.value == split_amount` — the new object's value is unverified.
3. `result.pool_id == fungible_staked_sui.pool_id` — pool membership preservation across the split is unverified. The observations note `pool_id` is copied by value so both objects refer to the same pool; nothing confirms the implementation actually copies it rather than, say, minting a fresh/wrong id.
4. Conservation: `old(value) == fungible_staked_sui.value + result.value` — the value-conservation invariant (no pool tokens created or destroyed) is unverified. This is the headline correctness property for a split.
5. `result` has a fresh UID — not a property the prover meaningfully tracks here, so its absence is acceptable.

Because the spec is only an `asserts`-gate with no `ensures`, the prover proves the function aborts exactly when `split_amount > value` and otherwise terminates — but says nothing about WHAT it computes. A buggy implementation that returned `result.value = 0`, copied the wrong `pool_id`, or failed to decrement the original would still pass this spec. For a split/conservation primitive, the value arithmetic and pool_id frame are the whole point, so this is a high-severity gap.

The original is passed by `&mut`, so a `clone!` snapshot before the call is needed to express the `old(value)` postconditions. Private field `value` is already exposed via the `fungible_staked_sui_value` getter and `pool_id` via `fungible_staked_sui_pool_id`, so all five ensures are straightforwardly expressible. The `@VERIFY(🛡️/✅)` marker suggests abort modeling is considered done, but the semantic postcondition stage is effectively empty.

## Strengths

- The single abort condition (split_amount > value, EInsufficientPoolTokenBalance) is covered exactly and correctly via the inverted asserts(split_amount <= value).
- Uses the public getter fungible_staked_sui_value in the asserts rather than reaching into private fields, keeping the precondition robust.
