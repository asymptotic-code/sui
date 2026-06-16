# Spec Review: `staking_pool_specs::fungible_staked_sui_value_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec contains no `ensures`, so it does not verify the function's only behavioral property: that the returned u64 equals the `value` field of the input. The body just calls the target and returns the result. With `no_opaque` the prover inlines the body and the spec passes vacuously, but it would also pass for an implementation returning any other value. Add `ensures(result == <value-field accessor>(fungible_staked_sui))`, exposing the private `value` field via a `#[test_only, ext(pure)]` accessor so the prover can connect the result to the field.

## Analysis

`fungible_staked_sui_value` is a pure read-only getter: it returns `fungible_staked_sui.value` directly off an immutable reference, with no callees, no mutation, and no abort paths. The writeup correctly characterizes it as a low-complexity accessor whose sole obligation is the postcondition `result == fungible_staked_sui.value`.

**Aborts / requires:** The writeup lists no abort conditions and no requires, which matches the implementation (a single field read cannot abort). The spec carries no `asserts`/`requires`, which is correct.

**Ensures:** This is the one real gap. The writeup's single ensures (`result == fungible_staked_sui.value`) is the entire correctness content of this function, yet the spec body contains zero `ensures`. As written, the spec merely calls the target and returns the result; it does not constrain the return value at all. Because the spec is marked `no_opaque`, the prover inlines the body during verification of this target, so the suite passes trivially — but it passes vacuously: any implementation that returned a different `u64` would still satisfy this spec, since nothing is asserted about the result. More importantly, callers that `include=` this spec to treat the getter opaquely would get no postcondition to rely on. To connect `fungible_staked_sui_value(x)` to `x.value` for opaque callers, the `value` field must be exposed to specs (e.g. a `#[test_only, ext(pure)]` accessor) and the postcondition stated explicitly: `ensures(result == module::value_field(fungible_staked_sui))`.

Since the input is an immutable reference (`&FungibleStakedSui`), no unchanged-field ensures are needed — the type system already guarantees the struct is not mutated.

The `@VERIFY(🛡️/✅)` marker on the spec flags an abort-modeling concern, which is puzzling for a function with no abort paths; it likely just reflects the verification harness state rather than a genuine issue.

## Strengths

- Correctly models the absence of abort paths — no spurious asserts/requires on a pure getter.
- Targets the right function with no_opaque, so the getter's field read can be inlined where needed.
