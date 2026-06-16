# Spec Review: `staking_pool_specs::pool_token_amount_spec`

**Verdict:** △ Has Gaps

## Issues

- ○ **medium/high**: The spec has no `ensures`, so it does not verify the function's only behavioral property: that the returned u64 equals the struct's `pool_token_amount` field. As written the spec passes vacuously and would continue to pass even if the accessor returned the wrong field (e.g. `sui_amount`) or an unrelated value. Add an ensures tying the result to the field -- e.g. expose `pool_token_amount` via a `#[test_only, ext(pure)]` accessor and assert `result == that_accessor(exchange_rate)`.

## Analysis

The target `staking_pool::pool_token_amount` is a pure field projection: `exchange_rate.pool_token_amount`. It takes an immutable `&PoolTokenExchangeRate`, performs no arithmetic, touches no table/dynamic field, never aborts, and has no preconditions. The writeup confirms this: empty `aborts`, empty `requires`, and a single `ensures` -- `result == exchange_rate.pool_token_amount`.

The spec is marked `#[spec(prove, no_opaque)]` and simply forwards to the target, returning its result. There are 0 asserts, 0 requires, and 0 ensures.

Abort coverage: nothing to cover -- the function cannot abort, so the absence of `asserts` is correct. The `_Assume` check passes vacuously (no asserts), and `_SpecNoAbortCheck` is trivial.

Postcondition coverage: this is the one real gap. The writeup's sole `ensures` (`result == exchange_rate.pool_token_amount`) is not present in the spec. Because the spec carries `no_opaque`, the prover inlines the body, so the call site is connected to the field read and the spec still passes -- but the spec does not actually *assert* the projection. As written it verifies nothing beyond well-formedness: it would still pass if the accessor returned the wrong field (e.g. `sui_amount`) or any other u64, since there is no `ensures` to contradict. Adding `ensures(result == staking_pool::pool_token_amount(exchange_rate))` is circular and useless; the meaningful check requires exposing the field to the spec (e.g. a `#[test_only, ext(pure)]` accessor) and asserting equality against it, matching the writeup's intent.

Since the input is an immutable reference, there are no mutations or unchanged-field obligations to verify -- the type system already guarantees `PoolTokenExchangeRate` is untouched. The `no_opaque` flag is appropriate for a getter and is not itself an issue.

## Strengths

- Correctly omits abort modeling -- the target is a pure, non-aborting field read, so the empty asserts/requires set is right.
- Uses `no_opaque`, the appropriate flag for a trivial getter, keeping the field read transparent to callers.
- Signature and forwarding match the target exactly, so the spec is well-formed and consistent with the implementation.
