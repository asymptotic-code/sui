# Spec Review: `staking_pool_specs::join_fungible_staked_sui_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has no `ensures` verifying the core accounting effect `self.value == old(self.value) + old(other.value)`. This is the only meaningful state change the function makes, and with both abort preconditions (same-pool, no-overflow) already asserted it would prove exactly. Without it, the spec proves only abort-freedom — an implementation that dropped, no-oped, or miscomputed the merged value (e.g. `self.value = other.value`) would pass unchanged. Capture `self.value` before the call via `clone!` and assert the sum after.
- ○ **medium/high**: No frame `ensures` that the receiver's pool identity is preserved (`self.pool_id == old(self.pool_id)`). `self` is a `&mut` reference, so the type system does not guarantee `pool_id` is untouched; a complete spec should assert it stays equal to its pre-call value.

## Analysis

The target `join_fungible_staked_sui` is a low-complexity client-side merge: it destructures `other` (a FungibleStakedSui), checks both tokens belong to the same pool (`self.pool_id == pool_id`, else aborts EWrongPool), deletes `other`'s UID, and sets `self.value = self.value + value`. No StakingPool state is touched.

ABORTS COVERAGE — complete. The single source abort is `assert!(self.pool_id == pool_id, EWrongPool)`. The spec covers it with `asserts(fungible_staked_sui_pool_id(self) == fungible_staked_sui_pool_id(&other))`, the correct inverted-logic precondition. The spec additionally adds `asserts(self.value.to_int().add(other.value.to_int()).lte(u64::max))` to model the implicit overflow abort on `self.value + value`. This is a legitimate, even superior, treatment of an abort the writeup only mentions as a theoretical concern in its observations — the `_Assume` check would force this precondition anyway, so making it explicit is correct.

ENSURES COVERAGE — entirely absent, and this is the substantive gap. The function has exactly one meaningful state effect — `self.value` grows by `other.value` — and the writeup spells out four postconditions, none of which the spec verifies:
  1. `self.value == old(self.value) + old(other.value)` — the core accounting property. With both abort preconditions asserted, this would prove cleanly (no overflow, exact sum). Its omission is the most important miss: the spec proves the function does not abort under the stated conditions but says nothing about *what it computes*. A buggy implementation that set `self.value = other.value` (drop instead of add), or `self.value = self.value` (no-op), or applied a wrong arithmetic, would pass this spec unchanged. The spec is consistent with the implementation but does not pin the implementation's correctness.
  2. `self.pool_id == old(self.pool_id)` — frame condition on the receiver's pool identity. `self` is `&mut`, so the type system does NOT guarantee this field is unchanged; it should be asserted. Cheap to add via `clone!`.

The `tag @VERIFY(🛡️/✅)` marks this as abort-modeling-focused and passing, but a complete spec for a merge/accounting function must verify the value arithmetic. The mirror function `split_fungible_staked_sui` has the analogous `self.value - split_amount` effect; whatever spec exists there should have a symmetric value ensures, and this one should match.

The remaining writeup postconditions — `other is destroyed` (UID deleted) and `no new objects created / total value conserved` — are object-lifecycle properties the prover does not track as ensures, so their absence is not a reportable gap. Value conservation is, in effect, the same statement as postcondition 1 (self grows by exactly other's full value), so asserting the arithmetic equality captures the economically meaningful half of conservation.

## Strengths

- Correctly inverts the sole source-level abort (`EWrongPool`) into a same-pool precondition using the getter form `fungible_staked_sui_pool_id`.
- Proactively models the implicit `self.value + value` overflow abort with an exact `Integer`-space `lte(u64::max)` precondition — stronger than the writeup, which only flagged overflow as a theoretical aside, and exactly what `_Assume` requires.
- Minimal and well-scoped for a low-complexity client-side merge; no unnecessary StakingPool-state assertions (the function touches no pool state).
