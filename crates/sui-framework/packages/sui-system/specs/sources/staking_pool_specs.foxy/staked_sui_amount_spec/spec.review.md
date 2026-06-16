# Spec Review: `staking_pool_specs::staked_sui_amount_spec`

**Verdict:** △ Has Gaps

## Issues

- ○ **medium/high**: The spec has no `ensures`. It calls `staking_pool::staked_sui_amount(staked_sui)` and returns the result without asserting `result == staked_sui.principal.value()`. Add an `ensures` binding the returned u64 to the underlying principal balance value (via a `#[test_only, ext(pure)]` accessor on the private `principal` field, then `balance::value`) so the accessor's contract is actually verified rather than passing as a tautology.

## Analysis

`staked_sui_amount` is a transparent one-line accessor returning `staked_sui.principal.value()`. It has zero branches, zero aborts, and no side effects, so abort coverage is correctly empty (writeup lists no aborts) and the `no_opaque` flag is appropriate to keep the body inlined for callers.

The gap is the postcondition. The spec body simply calls the target and returns its result with NO `ensures`. The writeup's primary postcondition — `result == staked_sui.principal.value()` — is not verified. While this property is definitionally true (the function IS that expression), an explicit `ensures` is what lets opaque callers reason about the return value, and is the canonical contract for a getter. With `no_opaque` present the body is inlined so callers can currently see through it, but the spec still verifies nothing on its own: it is a tautological pass-through. The principal value getter is consumed directly by `calculate_rewards` (via `.amount()`), so pinning `result == staked_sui.principal.value()` documents the exact quantity reward logic depends on.

The writeup's secondary postcondition — `result >= MIN_STAKING_THRESHOLD` — is, as the writeup itself notes, a system-level invariant established by `request_add_stake` / `split`, not something this accessor enforces. It would only be sound here as a `requires` assumption, and is not load-bearing for this function; its absence is not a real gap. The exposed principal value field is read-only via `&StakedSui`, so no unchanged-field framing is needed.

## Strengths

- Correctly models the function as abort-free — no `asserts`/`requires`, matching a pure projection with zero branches.
- Uses `no_opaque`, the right choice for a trivial getter so callers see the inlined body.
- Does not overreach by encoding the MIN_STAKING_THRESHOLD system invariant as an ensures, which would be unsound for an isolated accessor.
