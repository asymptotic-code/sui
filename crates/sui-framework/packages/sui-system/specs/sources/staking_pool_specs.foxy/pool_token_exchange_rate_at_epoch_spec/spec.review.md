# Spec Review: `staking_pool_specs::pool_token_exchange_rate_at_epoch_spec`

**Verdict:** △ Has Gaps

## Issues

- ● **high/high**: The spec has no `ensures` at all, so the function's defining property — that it returns the historically-correct PoolTokenExchangeRate — is completely unverified. The writeup requires (a) result == initial_exchange_rate() (sui_amount=0, pool_token_amount=0) when the pool is preactive at the requested epoch, and (b) result == exchange_rates[e] for the maximal recorded epoch e in [activation_epoch, min(deactivation_or_epoch, epoch)] otherwise. As written, an implementation returning the 1:1 rate unconditionally, or the rate at the wrong epoch (missing the deactivation clamp, scanning forward instead of backward), would still pass. Since this rate feeds withdraw/redeem/reward accounting across the whole staking lifecycle, returning the wrong rate is the core correctness risk and it is unchecked.
- ○ **medium/high**: The preactive branch postcondition is missing even though it is the simplest to express and verify: when pool.is_preactive_at_epoch(epoch) holds, the result must be the zero exchange rate (sui_amount == 0 and pool_token_amount == 0). Add `if (pool.is_preactive_at_epoch(epoch)) { ensures(staking_pool::sui_amount(&result) == 0); ensures(staking_pool::pool_token_amount(&result) == 0); };`.
- ○ **medium/medium**: The deactivation-epoch clamping behavior is unverified: for a deactivated pool, any query with epoch > deactivation_epoch should return the rate recorded at deactivation_epoch (per the writeup observation). This clamp is exactly the kind of off-by-one boundary logic a value-level `ensures` would catch; without it the spec gives no confidence the clamp is implemented correctly.

## Analysis

The target `pool_token_exchange_rate_at_epoch` is a read-only lookup: given an epoch, it returns the recorded `PoolTokenExchangeRate`. It is preactive-guarded (returns the 1:1 `initial_exchange_rate` when the pool is preactive at the requested epoch), otherwise clamps the epoch to the deactivation epoch and scans backwards through `pool.exchange_rates` to find the latest recorded rate at or before the requested epoch. The function performs no arithmetic on the returned value, so there are no overflow/underflow abort paths — consistent with the writeup's empty `aborts` list.

ABORTS. The only realistic abort source inside the body is `*pool.activation_epoch.borrow()` (and the same `.borrow()` inside `is_preactive_at_epoch`), which aborts if `activation_epoch` is `None`. But `is_preactive_at_epoch` short-circuits via `pool.is_preactive()` (i.e. `activation_epoch.is_none()`), so on the path that reaches `borrow()` the option is always `Some`. The spec carries no `asserts` and no `ignore_abort`; because the prover passes all three checks, the `.borrow()` non-abort must be discharged from the structure itself (the preactive early return means `borrow()` is only reached when `is_some()`). So abort modeling is implicitly handled — no missing abort `asserts`. The `requires` clause present is not needed for abort-freedom of `borrow()`; see below.

REQUIRES. The spec has exactly one precondition:
`requires(pool.is_preactive() || exchange_rates(pool).contains(*activation_epoch(pool).borrow()))`.
This encodes the protocol invariant from `activate_staking_pool` (the initial rate is always inserted at `activation_epoch`). Note the `requires` expression itself dereferences `activation_epoch(pool).borrow()` — but it is guarded by the `is_preactive()` short-circuit (`||`), so when the pool is preactive the second disjunct (with the borrow) is never evaluated. This precondition is what makes the backwards scan provably terminate at a found entry rather than the 'unreachable' fallback: it guarantees there exists a recorded rate at `activation_epoch`, which is the lower bound of the scan range. Reasonable and well-targeted.

ENSURES — THE CENTRAL GAP. The spec has **zero** `ensures`. The function's entire purpose is the value it returns, and the writeup spells out exactly what that value must be:
  1. `result == initial_exchange_rate()` (i.e. `{sui_amount: 0, pool_token_amount: 0}`) when `pool.is_preactive_at_epoch(epoch)`;
  2. `result == pool.exchange_rates[e]` where `e` is the maximal epoch in `[activation_epoch, min(deactivation_or_epoch, epoch)]` with a recorded entry, otherwise.

None of this is verified. As written, the spec proves only that the function does not abort under the stated precondition — it says nothing about WHICH exchange rate is returned. A buggy implementation that returned `initial_exchange_rate()` unconditionally, or that returned the rate at the wrong epoch (e.g. forgot the deactivation clamp, or scanned forward instead of backward), would still satisfy this spec. Given the writeup's own framing — 'Accuracy is essential for correct reward accounting across the entire staking lifecycle' and that this helper feeds `withdraw_from_principal`, `withdraw_rewards`, `calculate_rewards`, `redeem_fungible_staked_sui`, `convert_to_fungible_staked_sui`, and `check_balance_invariants` — returning the historically-correct rate is the single property that matters, and it is entirely unverified.

The preactive branch is the most tractable to add and should be present at minimum:
`if (pool.is_preactive_at_epoch(epoch)) { ensures(staking_pool::sui_amount(&result) == 0); ensures(staking_pool::pool_token_amount(&result) == 0); };`
The deactivation-clamp behavior (the writeup's observation that a query past the deactivation epoch returns the deactivation-epoch rate) is also unverified and is exactly the kind of off-by-one / clamp logic that a value `ensures` would catch.

This is a getter-shaped function only superficially: unlike a field accessor, its return value is the result of nontrivial clamping + backward-scan logic, so the 'don't over-spec simple getters' caveat does not apply. The missing return-value ensures is a genuine high-severity gap, not a false positive.

COLLECTIONS. The function is read-only over `exchange_rates` (no mutation), so there is no partial-mutation / blast-radius concern — nothing to preserve.

## Strengths

- The single `requires` correctly captures the protocol invariant from activate_staking_pool (the initial rate is always inserted at activation_epoch), which is precisely what guarantees the backward scan terminates at a real entry rather than the source-annotated 'unreachable' fallback.
- The `requires` disjunct is correctly short-circuited via pool.is_preactive() so the `*activation_epoch(pool).borrow()` inside it is only evaluated when activation_epoch is Some, mirroring the implementation's own preactive early-return guard.
- Abort behavior is handled implicitly and correctly: the only abort source (borrowing activation_epoch) is unreachable on the live path due to the preactive guard, so no spurious asserts or ignore_abort are needed.
