# Spec Review: `staking_pool_specs::sui_amount_spec`

**Verdict:** △ Has Gaps

## Issues

- ○ **medium/high**: The spec has no `ensures` for the return value. The only verifiable property of this getter is `result == exchange_rate.sui_amount`, and it is absent — so the spec proves nothing about correctness (it does not even constrain that the returned u64 is the `sui_amount` field rather than, say, `pool_token_amount`). Add `ensures(result == exchange_rate.sui_amount);` (exposing the private field via a `#[test_only, ext(pure)]` accessor or `no_opaque`).

## Analysis

`sui_amount` is a pure field getter: it takes `&PoolTokenExchangeRate` and returns the `sui_amount: u64` field with no transformation, no arithmetic, no state mutation, and no abort paths. `PoolTokenExchangeRate` has `copy + drop + store` (no `key`), so it is a pure value type and the getter cannot touch protocol state.

The writeup lists no aborts and no requires (correct — there are none), and a single ensures: `result == exchange_rate.sui_amount`. The actual spec targets `staking_pool::sui_amount`, calls it, and returns the result, but contains **0 asserts, 0 requires, and 0 ensures**. It therefore verifies nothing beyond the fact that the call type-checks and does not abort (the implicit `_SpecNoAbortCheck`/abort-freedom obligation).

The one verifiable property — the return-value identity `result == exchange_rate.sui_amount` — is missing. Adding it is cheap and would pin the getter to the field it claims to read. Because `sui_amount` is a private field, the ensures needs the field exposed to the spec, e.g. via a `#[test_only, ext(pure)]` accessor or by reading `exchange_rate.sui_amount` directly through a spec helper; alternatively `no_opaque` keeps the body inline so callers see the field read. Note the `@VERIFY(🛡️/✅)` marker claims a passing/abort-modeled state, but with no ensures the spec is effectively an empty stub for a function that has a clear, trivially provable postcondition.

Since the input is an immutable reference (`&PoolTokenExchangeRate`), there are no unchanged-field obligations to add — the type system guarantees the value is not mutated.

## Strengths

- Correctly identifies that the getter has no abort paths and no preconditions — no spurious asserts/requires were added.
- Spec compiles and passes the prover, establishing abort-freedom for the call.
