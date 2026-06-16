# Spec Review: `staking_pool_specs::stake_activation_epoch_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has no `ensures` clause. The verification plan requires `ensures(result == staked_sui.stake_activation_epoch)`, which is the entire point of specifying a pure field projection. As written, the spec proves only that `stake_activation_epoch` does not abort (trivially true for an immutable-reference field read) and asserts nothing about the returned value. A buggy implementation that returned a different field (e.g. `pool_id` cast, or a constant) would pass this spec unchanged. The single load-bearing postcondition is missing.

## Analysis

`stake_activation_epoch` is a pure accessor that returns `staked_sui.stake_activation_epoch` directly. It takes an immutable `&StakedSui` reference, mutates nothing, and cannot abort — so the writeup correctly lists no aborts and no requires. The only obligation is the single postcondition tying the return value to the stored field.

**Aborts / requires:** Nothing to cover. The function reads a `u64` field through an immutable reference; there is no arithmetic, no table access, no early return. The empty `asserts`/`requires` sets are correct and complete.

**Ensures (the gap):** The spec is empty where it most matters. The writeup's sole `ensures` — `result == staked_sui.stake_activation_epoch` — is absent. Without it the proof degenerates to a `_SpecNoAbortCheck` tautology: it confirms the body is abort-free but says nothing about the value returned. For a getter, the value identity *is* the specification; omitting it means the spec verifies essentially nothing about correctness. This is the classic 'passes the prover but checks the wrong (empty) thing' situation.

**no_opaque:** The spec carries `no_opaque`, which is appropriate for a getter — callers that rely on `stake_activation_epoch(s)` resolving to `s.stake_activation_epoch` need the body inlined rather than abstracted behind a (here, vacuous) contract. Note, however, that `no_opaque` makes the missing `ensures` even more consequential: with no postcondition AND no opaque contract, this spec provides no reusable guarantee to the many downstream callers (`request_withdraw_stake`, `convert_to_fungible_staked_sui`, `withdraw_from_principal`, `calculate_rewards`) that index the exchange-rates table by this epoch.

**Fix:** add the field-projection postcondition. Exposing the private field to the spec may require a `#[test_only, ext(pure)]` accessor on `StakedSui` (or reading it directly if the spec module has visibility):

```move
ensures(result == staking_pool::stake_activation_epoch_field(staked_sui));
```

where the helper returns `staked_sui.stake_activation_epoch`. With `no_opaque` the prover inlines the body, so the equality discharges immediately.

## Strengths

- Correctly identifies the function as abort-free and includes no spurious asserts/requires.
- Uses `no_opaque`, the right opacity choice for a field-projecting getter so callers see the inlined body.
