---
schema_version: 1
agent: function_local_accesses
model: claude-sonnet-4-6
max_tokens: 4096
interactive: false
effort: null
timeout: 60
token_budget: null
harness: foxy
parent_session: /Users/cos/asymptotic/agent/clients/mysten/sui/.foxy/sessions/spec_writeup_20260615_234357_2491
name: staking-pool-specs-qualify
description: Qualify structs in staking_pool_specs module using last segment as prefix.
cwd: /Users/cos/asymptotic/agent/clients/mysten/sui
started_at: 2026-06-15T23:44:00.544175
foxy_commit: 38346c7c25594d3c381dff95b53fe33dba150411
prompt_part_hashes: {"function_local_accesses": "6befc61053169412", "_global": "550ca333122fa91857467a30ba42324dcecf631304e8ab83944a55ebbb4a5ebb"}
---

## System Prompt

````
# Work Directly On the Current Branch (read first)

Unless the user explicitly asks otherwise, do ALL work in place — in the current checkout, on whatever branch is already checked out (normally `main`). Do NOT create feature branches, git worktrees, separate working folders, or clones of the project. This overrides any default tooling guidance that says to branch, clone, or spin up an isolated copy before committing. If you catch yourself reaching for `git checkout -b`, `git switch -c`, `git branch`, `git worktree add`, `git clone`, or a new project folder without being asked, stop and work in the current checkout instead.

The one exception is an explicit request: if the user specifically asks for a branch, worktree, folder, or clone, do exactly that — this section never blocks an explicit instruction.

(Commit and push only when the user explicitly asks — and when you do, commit to the current branch.)


Analyze the Sui Move function source below. Report ONLY accesses that appear **directly in this function's body**. Do NOT infer or guess what called functions do internally -- transitive accesses are captured separately.

## Using the Context section

If a `## Context` section is present in the user message, it gives you:

- **Module** -- the fully-qualified module of the function. Structs defined in this file use the last segment as their prefix.
- **Imports** -- `use` statements. For any struct name that appears in the function body, the `use` list tells you which module to prefix it with (e.g. `use pyth::price_info::PriceInfoObject` means `PriceInfoObject` is `price_info::PriceInfoObject`).
- **Structs defined in this file** -- full struct definitions with field names and types. Cross-reference field names: if you write `pool.foo`, `foo` must appear as a field of `Pool` in that list.
- **Referenced structs (from other modules)** -- struct definitions pulled in because the function's parameters, return type, or body mention them. Use these the same way as local struct defs: to cross-check field names and, crucially, to determine each field's declared type for the instantiation list below.

**Always prefer the Context section over guessing.** If a struct or field is not in the context and you cannot qualify it from the imports, use `module::Struct` only for the module that actually declares the type -- never invent one from the project or file name.

## Output shape (unified 3-tuple)

Every recorded access -- static, dynamic, transfer, or emit -- is a 3-element tuple:

```
[struct_or_parent, field_key_or_kind, [instantiated_types]]
```

- `struct_or_parent`: `module::StructName` of the value being accessed / transferred / emitted.
- `field_key_or_kind`: depends on the bucket.
  - Static (`reads` / `writes`): the **field name** as declared on the struct.
  - Dynamic (`dynamic_reads` / `dynamic_writes`): the **key value** passed at runtime (see dynamic key rules below). The schema also permits a list of keys for chained dynamic accesses, but emit one entry per intrinsic call -- do not chain.
  - Transfers: the **flavour** -- one of `transfer`, `shared_object`, `freeze_object`.
  - Emits: the literal string `"emit"`.
- `instantiated_types`: **a list of strings**, sorted lexicographically, deduplicated. One entry per distinct type seen at use sites within this function's body. For static / dynamic the list must be **non-empty**. For `transfers` and `emits` an **empty list** is acceptable when the struct is non-generic (there is no instantiation to record).

## Qualification rules (READ CAREFULLY)

These rules apply to **every struct name** you write out, whether it appears as the parent in slot 0, or nested inside a type expression in the types list (slot 2).

**Always fully qualify** (`module::StructName`) -- never bare. This applies uniformly to project structs, vendor structs, and Sui/std framework structs alike. Examples:

| What you might write bare | What you must write |
|---------------------------|---------------------|
| `Account`                 | `vault::Account`    |
| `PriceInfoObject`         | `price_info::PriceInfoObject` |
| `Coin`                    | `coin::Coin`        |
| `Balance`                 | `balance::Balance`  |
| `UID`                     | `object::UID`       |
| `ID`                      | `object::ID`        |
| `TxContext`               | `tx_context::TxContext` |
| `Clock`                   | `clock::Clock`      |
| `Table`                   | `table::Table`      |
| `Bag`                     | `bag::Bag`          |
| `TypeName`                | `type_name::TypeName` |
| `Option`                  | `option::Option`    |

**Primitives stay bare**: `u8`, `u16`, `u32`, `u64`, `u128`, `u256`, `bool`, `address`, `vector<...>`. They are not struct types and have no module.

**`String` is ambiguous** -- always disambiguate explicitly. Sui has both `0x1::ascii::String` and `std::string::String`. Write `ascii::String` or `string::String` based on which the source actually uses; never bare `String`.

If you are unsure whether a struct is a project struct or a framework type, check the Context section. The validator cross-checks every struct reference against `struct_knowledge` and the framework registry; bare framework names get auto-rewritten to canonical form, but bare project structs are dropped.

## Static field accesses

Classify each as a **read** or **write**:

**Reads** -- the field value is observed but not modified:
- Dot access on a value or `&` ref: `obj.field`, `&obj.field`. **This applies to local variables too**, including ones holding a value the function just packed: after `let req = Request { owner, shares, ... };`, every subsequent `req.owner` / `req.shares` / `&req.owner` use is a read on `Request.owner` / `Request.shares` -- record it. Reads here are independent of the pack writes; both go to the output (the pack populates `writes`, each later dot access populates `reads`).
- Destructuring / unpack: `let Struct { field1, field2, .. } = obj;` binds each named field **out** of the struct, so every listed field is a **read**. This includes the shorthand form: `let I64 { bits } = v;` is a read on `I64.bits`. Do **not** mistake destructuring for packing -- the `{ field }` shorthand looks identical, so disambiguate by **position**: a `{ ... }` pattern on the **left of `=`** in a `let` (or any binding/match pattern) is destructuring (a **read**); a `Struct { ... }` expression being **constructed** as a value (right of `=`, returned, or passed as an argument) is packing (a **write**).

**Writes** -- the field value is created or modified:
- Mutation via `&mut`: `obj.field = expr;`
- Taking a mutable reference to the field: `&mut obj.field` (e.g. passed as a `&mut` argument). Also counts as a read, since the caller can both inspect and modify the current value.
- Struct packing (field is set during construction): `Struct { field: expr, .. }`. **Packing always counts as a write**, even when the packed value is consumed immediately in the same expression -- passed to `event::emit(...)`, returned, destructured, etc. Shorthand `{ field }` in a **constructor expression** is equivalent to `{ field: field }` and still writes every listed field. (This applies only to construction -- the same `{ field }` shorthand on the left of `let ... =` is destructuring, a read; see Reads above.)

**NOT an access** (default) -- do not record:
- Method calls on a struct: `obj.method()`, `obj.get_field()`, `vault.get_balance()`. Syntactically similar to field access but these are function calls; whatever fields the method reads/writes internally is captured transitively elsewhere.
- Passing the whole struct by `&` or `&mut` to another function: `validate(&instruction)`, `burn(&mut cap, coin)`. That's an argument pass, not a field access.
- Name collisions: if `foo(x)` exists as a free function AND `Struct` has a field named `foo`, `obj.foo` is still a field read. It's only a method call when the source uses parentheses: `obj.foo(...)`.
- **Object-lifecycle calls** (`transfer::*`, `event::emit`): these go to dedicated `transfers` / `emits` buckets, **not** `reads`/`writes`. See the "Transfers" and "Emits" sections below.

**Exception -- whitelisted callees:** if a called function appears in the `## Notes on called external functions` section below, treat it per the note. The note tells you how the call affects struct state (e.g. `balance::join` → write on the Balance struct, `table::add` → dynamic write on the enclosing struct). That's a record-able access even though syntactically it's a method/function call. **Outside of that whitelist, do not infer fields from method names** -- a call to `get_foo()` or `set_bar()` on a project struct does not record an access unless the Notes section says otherwise.

### Type list rules for static accesses

- Record the **field's declared type** after substituting the struct's generic parameters with whatever they are bound to at the use site. **Always qualify framework and project struct names**: `balance::Balance<SUI>`, not `Balance<SUI>`; `vault::Account`, not `Account`. Bare type parameters (`T`, `K`, `V`) and primitives (`u64`, `bool`, `address`, `vector<u8>`) are the only things that stay bare.
  - Example: if the struct is `Pool<phantom T>` with field `balance: Balance<T>`, and the function has a parameter `pool: &mut Pool<SUI>`, accessing `pool.balance` records `["balance::Balance<SUI>"]`.
  - Multiple distinct instantiations in the same function produce multiple entries: `["balance::Balance<SUI>", "balance::Balance<USDC>"]`.
- If the struct stays **generic** at the use site (the function is itself generic over `T` and the parameter is `pool: &Pool<T>`), record the declared type **with the type parameter as written**: `["balance::Balance<T>"]`. A bare type parameter (`T`, `K`, `V`) is correct and expected when it is not pinned at this level -- do **not** invent or substitute.
- For non-generic fields, the list is trivial: `["u64"]`, `["address"]`, `["bool"]`.
- **Fallback.** If you genuinely cannot determine the field's declared type (struct not in context), record what you can see, unchanged -- do not guess.

## Dynamic field accesses

Calls into `dynamic_field::*` / `dynamic_object_field::*`, or **mutating** table/bag/object_table/object_bag/linked_table/AcTable APIs noted in external callee hints (e.g. `table::add`, `table::borrow_mut`, `table::remove`, `bag::add`, `bag::borrow_mut`, `bag::remove`).

**Read-only library calls are opaque -- do NOT record them as dynamic accesses.** That includes `table::borrow`, `table::contains`, `bag::borrow`, `bag::contains`, `linked_table::borrow`, `object_bag::borrow`, etc. The static read of the field that holds the table/bag (e.g. `&q.table`) is the only signal -- whatever element gets fetched from inside is just typed data and not the agent's concern. Only operations that *insert*, *remove*, or take a *mutable* reference into the heterogeneous storage count as dynamic.

If a **Notes on called external functions** section is present in the input, use those notes to classify calls to those external functions. Calls without an explicit note do not produce dynamic accesses.

### Parent struct rules

The **parent struct** (slot 0) is the struct whose `UID` or field is being used as the storage root -- i.e. the thing the dynamic write/read is *rooted at*. Identify it by following the first `&`/`&mut` argument back to its value.

- `dynamic_field::remove(&mut pool.id, K)` -- rooted at `pool` whose type is `module::Pool`. Parent: `module::Pool`.
- `table::add(&mut vault.accounts, addr, v)` -- rooted at `vault.accounts`. Parent: `module::Vault`. The **parent is still the vault**, *not* the Table -- the Table is the field type, not the owning struct.
- `bag::borrow(supported_assets, name)` where `supported_assets: &Bag` came from `dynamic_field::borrow<_, Bag>(&vault.id, ...)` -- parent: the **Bag's enclosing struct chain**, which here is `module::Vault`. Record this dynamic access separately from the outer `dynamic_field::borrow`.

### Key rules (slot 1) -- COMMON MISTAKES HERE

The key slot is the **runtime key value passed to the call**, not the key's type and not the name of the field being traversed.

- If the key is a **literal or named constant** visible in the source, use it verbatim: `b"balance"`, `0`, `POSITION_KEY`, `SUPPORTED_ASSETS_DYNAMIC_FIELD`.
- If the key is a **`TypeName`** obtained via `type_name::get<T>()` or `type_name::with_defining_ids<T>()` (or assigned from one), use `"TypeName<T>"` with the concrete type parameter if visible, or `"TypeName"` if generic/unknown.
- If **computed** at runtime (e.g. `q.tail`, `addr`, a locally-computed expression), use `"*"`.

**Do not** put in the key slot:

- The **type** of the key (e.g. `vector<u8>` when the key is actually a constant of that type). Write the constant name instead.
- The **field name** of the field being traversed (e.g. `"accounts"` for `table::add(&mut vault.accounts, ...)`). That's not a key -- that's the field path. The key is the argument passed to `table::add`.
- The **type** of the value stored. That goes in slot 2.

### Type list rules for dynamic accesses

- Record the **value type** being read or written, with generic parameters substituted at the use site (same rules as static accesses).
- For `dynamic_field::borrow<K, V>(&obj.id, key)`, the value type is `V` with any substitutions.
- For `table::add(&mut t, k, v)` where `t: Table<K, V>`, the value type is `V` with any substitutions visible in `t`'s declared type.
- Bare type parameters from the enclosing function (e.g. `T` in a generic function) are fine -- record them verbatim.

## Transfers

Calls that hand an owned object to the Sui runtime go in the `transfers` bucket, **not** `reads`/`writes`. There are three flavours:

| Move call | Flavour |
|-----------|---------|
| `transfer::transfer(obj, addr)` / `transfer::public_transfer(obj, addr)` | `transfer` |
| `transfer::share_object(obj)` / `transfer::public_share_object(obj)` | `shared_object` |
| `transfer::freeze_object(obj)` / `transfer::public_freeze_object(obj)` | `freeze_object` |

Entry shape: `[module::StructName, "<flavour>", [types]]`.

- `module::StructName` is the type of the object being transferred / shared / frozen. Framework types like `Coin` may be bare; project structs must be qualified.
- The flavour string is one of `transfer`, `shared_object`, `freeze_object`. **Do not** distinguish `transfer` vs `public_transfer` -- both go in as `transfer`.
- The types list captures the struct's generic instantiation at the use site (`["SUI"]` for `Coin<SUI>`, `["T"]` when generic and not pinned, `[]` for non-generic structs like `AdminCap`).

Example:

```move
public fun launch(cap: AdminCap, cfg: Config<SUI>, ctx: &mut TxContext) {
    transfer::transfer(cap, ctx.sender());
    transfer::share_object(cfg);
}
```

```
"transfers": [
  ["admin::AdminCap", "transfer", []],
  ["admin::Config", "shared_object", ["SUI"]]
]
```

## Emits

Calls into `event::emit` go in the `emits` bucket. Entry shape: `[module::EventName, "emit", [types]]`.

- `module::EventName` is the event struct's qualified name. The event is typically defined in the project (`events::DepositEvent`) -- always fully qualify it.
- Slot 1 is literally the string `"emit"`.
- The types list captures the event's generic instantiation at the use site, or is empty for non-generic events.

The packing inside `event::emit(EventStruct { ... })` ALSO produces `writes` for each packed field on the event struct -- record both. (This rule is illustrated in the existing event-emission example below.)

Example:

```move
event::emit(DepositEvent { vault_id, amount });           // non-generic
event::emit(FeeCollectedEvent<SUI> { vault_id, fee });    // generic
```

```
"emits": [
  ["events::DepositEvent", "emit", []],
  ["events::FeeCollectedEvent", "emit", ["SUI"]]
]
```

## Output format

Reply with ONLY a JSON object (no explanation, no markdown fences). Six buckets, every list element is a 3-tuple:

```
{"reads":           [["module::StructName", "field",   ["Type1", "Type2"]], ...],
 "writes":          [["module::StructName", "field",   ["Type"]],           ...],
 "dynamic_reads":   [["module::StructName", "key",     ["ValueType"]],      ...],
 "dynamic_writes":  [["module::StructName", "key",     ["ValueType"]],      ...],
 "transfers":       [["module::StructName", "flavour", ["Type"]],           ...],
 "emits":           [["module::EventName",  "emit",    ["Type"]],           ...]}
```

Rules:
- All project struct names are `module::StructName`. Framework types (UID, Coin, Balance, Table, etc.) may be bare.
- Use empty list `[]` for categories with no accesses.
- Deduplicate entries within each category by `(struct, field_or_key)`; merge their type lists.
- A field that is both read and written appears in both `reads` and `writes`.
- `instantiated_types` is non-empty for `reads`/`writes`/`dynamic_reads`/`dynamic_writes`. For `transfers` and `emits` it MAY be empty when the struct is non-generic.

## Example (non-generic field, dynamic field with constant key)

Given:
```move
const RESERVE_KEY: vector<u8> = b"reserve";

public fun withdraw(pool: &mut Pool, amount: u64): Balance<SUI> {
    assert!(pool.balance_value >= amount, EInsufficientBalance);
    pool.balance_value = pool.balance_value - amount;
    let coin: Coin<SUI> = dynamic_field::remove<vector<u8>, Coin<SUI>>(&mut pool.id, RESERVE_KEY);
    coin::into_balance(coin)
}
```

Output:
{"reads": [["lending::Pool", "balance_value", ["u64"]]], "writes": [["lending::Pool", "balance_value", ["u64"]]], "dynamic_reads": [], "dynamic_writes": [["lending::Pool", "RESERVE_KEY", ["coin::Coin<SUI>"]]], "transfers": [], "emits": []}

Notes:
- Key slot is `"RESERVE_KEY"` (the constant name), **not** `"vector<u8>"` (the key's type).
- All 6 buckets are present, even when empty.
- `coin::Coin<SUI>` is fully qualified -- never bare `Coin<SUI>`. The qualification rule applies inside type expressions too.

## Example (transfer + share)

Given:
```move
public fun init_protocol(ctx: &mut TxContext) {
    let cap = AdminCap { id: object::new(ctx) };
    let cfg = ProtocolConfig { id: object::new(ctx), version: 1 };
    transfer::transfer(cap, ctx.sender());
    transfer::share_object(cfg);
}
```

Output:
{"reads": [], "writes": [["admin::AdminCap", "id", ["object::UID"]], ["admin::ProtocolConfig", "id", ["object::UID"]], ["admin::ProtocolConfig", "version", ["u64"]]], "dynamic_reads": [], "dynamic_writes": [], "transfers": [["admin::AdminCap", "transfer", []], ["admin::ProtocolConfig", "shared_object", []]], "emits": []}

Notes:
- Each `transfer::*` call adds an entry to the `transfers` bucket. Flavour is `transfer` for `transfer::transfer`/`public_transfer` and `shared_object` for `share_object`/`public_share_object`.
- The struct packing inside `AdminCap { ... }` and `ProtocolConfig { ... }` still produces `writes` for each packed field.
- Type slot is `[]` because `AdminCap` and `ProtocolConfig` are non-generic.

## Example (generic struct, concrete instantiation)

Given:
```move
public fun deposit_sui(pool: &mut Pool<SUI>, coin: Coin<SUI>) {
    let b = coin::into_balance(coin);
    balance::join(&mut pool.balance, b);
}
```

Where `Pool<phantom T>` has field `balance: Balance<T>`.

Output:
{"reads": [["pool::Pool", "balance", ["balance::Balance<SUI>"]]], "writes": [["pool::Pool", "balance", ["balance::Balance<SUI>"]]], "dynamic_reads": [], "dynamic_writes": [], "transfers": [], "emits": []}

## Example (pack + post-pack reads)

Given:
```move
public fun redeem(vault: &mut Vault, shares: u64, ctx: &mut TxContext): WithdrawalRequest {
    let request = WithdrawalRequest {
        owner: ctx.sender(),
        shares,
        timestamp: clock::timestamp_ms(clock),
    };
    events::emit_redeem_event(request.owner, request.shares, request.timestamp);
    request
}
```

Output:
{"reads": [["vault::WithdrawalRequest", "owner", ["address"]], ["vault::WithdrawalRequest", "shares", ["u64"]], ["vault::WithdrawalRequest", "timestamp", ["u64"]]], "writes": [["vault::WithdrawalRequest", "owner", ["address"]], ["vault::WithdrawalRequest", "shares", ["u64"]], ["vault::WithdrawalRequest", "timestamp", ["u64"]]], "dynamic_reads": [], "dynamic_writes": [], "transfers": [], "emits": []}

Notes:
- Each packed field is a **write** (pack populates the field).
- Each post-pack `request.owner` / `request.shares` / `request.timestamp` is **also a read** on the same struct field. The pack and the read are independent recorded accesses -- record both, even though they happen on the same local variable in the same function.
- Reading the local back after the pack is the easiest case to forget. Always scan the function body after a `let x = Struct { ... };` for subsequent `x.field` uses, including ones nested inside other call arguments (`f(req.owner, req.shares)`).

## Example (event emission with struct packing)

Given:
```move
public(package) fun emit_fee_event<T>(vault_id: ID, fee: u64) {
    event::emit(FeeCollectedEvent<T> { vault_id, fee });
}
```

Where `FeeCollectedEvent<phantom T>` is declared in the current `events` module with fields `vault_id: ID` and `fee: u64`.

Output:
{"reads": [], "writes": [["events::FeeCollectedEvent", "fee", ["u64"]], ["events::FeeCollectedEvent", "vault_id", ["object::ID"]]], "dynamic_reads": [], "dynamic_writes": [], "transfers": [], "emits": [["events::FeeCollectedEvent", "emit", ["T"]]]}

Notes:
- The pack inside `emit(...)` still counts as a write for every packed field. Event emission does not suppress packing writes.
- The `emit` itself is also recorded under the `emits` bucket: `["events::FeeCollectedEvent", "emit", ["T"]]`. The type slot reflects the use-site instantiation -- here `T` because the function is generic over `T` and it is not pinned at this level.
- Shorthand `{ vault_id, fee }` is equivalent to `{ vault_id: vault_id, fee: fee }` -- each field is being set.
- Field ordering in the output is lexicographic, not source order.

## Example (same field under multiple instantiations)

Given:
```move
public fun rebalance(p_sui: &mut Pool<SUI>, p_usdc: &mut Pool<USDC>) {
    p_sui.balance;
    p_usdc.balance;
    p_sui.balance = p_sui.balance;
}
```

Output:
{"reads": [["pool::Pool", "balance", ["balance::Balance<SUI>", "balance::Balance<USDC>"]]], "writes": [["pool::Pool", "balance", ["balance::Balance<SUI>"]]], "dynamic_reads": [], "dynamic_writes": [], "transfers": [], "emits": []}

## Example (generic function, T not pinned)

Given:
```move
public fun enqueue<T: copy + drop + store>(q: &mut Queue<T>, val: T) {
    table::add(&mut q.table, q.tail, val);
    q.tail = q.tail + 1;
}
```

Where `Queue<phantom T>` has fields `table: Table<u64, T>` and `tail: u64`.

Output:
{"reads": [["queue::Queue", "table", ["table::Table<u64, T>"]], ["queue::Queue", "tail", ["u64"]]], "writes": [["queue::Queue", "table", ["table::Table<u64, T>"]], ["queue::Queue", "tail", ["u64"]]], "dynamic_reads": [], "dynamic_writes": [["queue::Queue", "*", ["T"]]], "transfers": [], "emits": []}

Notes:
- `q.table` is read AND written (taking `&mut` of it counts as both).
- Dynamic write parent is `queue::Queue` (the owning struct), key is `"*"` (q.tail is a runtime-computed value), value type is `T` (the function's type parameter -- bare T is correct here since no substitution is possible at this level).

## Example (table::add on a struct field)

Given:
```move
public fun record_account<T,R>(vault: &mut Vault<T,R>, owner: address, acc: Account) {
    table::add(&mut vault.accounts, owner, acc);
}
```

Where `Vault<phantom T, phantom R>` has field `accounts: Table<address, Account>`.

Output:
{"reads": [["vault::Vault", "accounts", ["table::Table<address, vault::Account>"]]], "writes": [["vault::Vault", "accounts", ["table::Table<address, vault::Account>"]]], "dynamic_reads": [], "dynamic_writes": [["vault::Vault", "*", ["vault::Account"]]], "transfers": [], "emits": []}

Notes:
- `vault.accounts` is read AND written (`&mut` of it).
- Dynamic write parent is `vault::Vault` (the struct that owns the table), key is `"*"` (owner is a runtime address), value type is `vault::Account` (fully qualified -- Account is a project struct).
- The key slot is **not** `"accounts"`. "accounts" is a field name, not a key.

## Example (dynamic_field with constant key, nested bag::borrow)

Given:
```move
fun get_supported<T>(vault: &Vault<T,R>): &SupportedAsset<T> {
    let bag = dynamic_field::borrow<vector<u8>, Bag>(&vault.id, SUPPORTED_ASSETS_KEY);
    let name = type_name::get<T>();
    bag::borrow<TypeName, SupportedAsset<T>>(bag, name)
}
```

Output:
{"reads": [["vault::Vault", "id", ["object::UID"]]], "writes": [], "dynamic_reads": [["vault::Vault", "SUPPORTED_ASSETS_KEY", ["bag::Bag"]]], "dynamic_writes": [], "transfers": [], "emits": []}

Notes:
- Only the outer `dynamic_field::borrow` call surfaces as a dynamic read -- it's a real `dynamic_field::*` op against the vault's UID, with constant key `SUPPORTED_ASSETS_KEY` returning a `Bag`.
- The inner `bag::borrow` is a read-only library call and does NOT produce a dynamic_read. The Bag's typed API is treated as opaque -- whatever element it returns is just typed data.
- `vault.id` is a static read because the function takes `&vault.id` to pass to `dynamic_field::borrow`.

## Example (TypeName dynamic key, AcTable)

Given:
```move
public(friend) fun add_risk_model<T>(
    self: &mut AcTable<RiskModels, TypeName, RiskModel>,
    cap: &AcTableCap<RiskModels>,
    risk_model: RiskModel,
) {
    let type_name = type_name::get<T>();
    ac_table::remove(self, cap, type_name);
    ac_table::add(self, cap, type_name, risk_model);
}
```

Output:
{"reads": [], "writes": [], "dynamic_reads": [], "dynamic_writes": [["ac_table::AcTable", "type_name::TypeName<T>", ["ac_table::RiskModel"]]], "transfers": [], "emits": []}

# Execution Mode

You are running non-interactively as a programmatic sub-task. Complete the task decisively without asking questions. Make reasonable default choices. If you truly cannot proceed, use final_result({"error": "Cannot proceed: <reason>"}) to report the failure.

# Tool Reminder (READ BEFORE ANY FILE OR SHELL WORK)

Before reading, editing, searching, writing, or shelling out: use `mcp__plugin_foxy_foxy__python` with the pre-loaded Python functions. Never call Claude Code's native `Read`, `Edit`, `Write`, `Grep`, `Glob`, `Bash`, or `Agent` — they are blocked by the PreToolUse hook.

- Read → `read("path")` or `read("path", start=10, end=20)`
- Edit → `edit("path", "old", "new")`
- Write (new files only) → `write("path", content)`
- Glob → `glob("**/*.py")`
- Grep → `grep("pattern", include="*.py")`
- Bash → `subprocess.run([...], capture_output=True, text=True)`
- Agent → `run_agent("agent_name", "task")`

These functions are already in the namespace — no imports needed. The namespace persists across `mcp__plugin_foxy_foxy__python` calls, so variables defined in one snippet are available in the next.

**`read`, `grep`, `glob`, and `subprocess.run` return data silently** — you see nothing unless you `foxy_inspect(result)` or assign and inspect a summary. `edit` and `write` mutate the file directly, no inspect needed. `print()` goes to a temp file, not your context.
````
---

## User

## Context

**Module:** `specs::staking_pool_specs` -- qualify structs defined in this file using the last segment as the prefix (e.g. if module is `a::b`, use `b::StructName`).

**Imports:**
```move
use sui::tx_context::TxContext;
use sui_system::staking_pool::PoolTokenExchangeRate;
use sui_system::staking_pool::StakingPool;
use sui_system::staking_pool::StakedSui;
use sui_system::staking_pool;
use sui_system::staking_pool::FungibleStakedSui;
use sui::object::ID;
use prover::prover::{asserts, ensures};
use prover::ghost;
use specs::transfer_spec::{SpecTransferAddress, SpecTransferAddressExists};
```

**Referenced structs (from other modules):**
```move
/// A self-custodial object holding the staked SUI tokens.
public struct StakedSui has key, store {
    id: UID,
    /// ID of the staking pool we are staking with.
    pool_id: ID,
    /// The epoch at which the stake becomes active.
    stake_activation_epoch: u64,
    /// The staked SUI tokens.
    principal: Balance<SUI>,
}
```

```move
#[spec(prove, target=staking_pool::pool_id, ignore_abort, no_opaque)]
fun pool_id_spec(
    staked_sui: &StakedSui,
): ID {
    staking_pool::pool_id(staked_sui)
}
```

## Assistant

I need to analyze the `pool_id_spec` function body directly.

The function body contains a single call: `staking_pool::pool_id(staked_sui)` — this is a function call, not a field access. The parameter `staked_sui: &StakedSui` is passed by reference to the function, which is just an argument pass, not a field access.

There are no static field accesses, dynamic field accesses, transfers, or emits directly in this function body.

```json
{"reads": [], "writes": [], "dynamic_reads": [], "dynamic_writes": [], "transfers": [], "emits": []}
```

> 412 in, 152 out, 8007 cache create

---

# Session Complete

| | |
|---|---|
| status | success |
| stop_reason | end_turn |
| model | claude-sonnet-4-6 |
| prompt | 550ca333122fa91857467a30ba42324dcecf631304e8ab83944a55ebbb4a5ebb |
| parent | /Users/cos/asymptotic/agent/clients/mysten/sui/.foxy/sessions/spec_writeup_20260615_234357_2491 |
| turns | 1 |
| tool_calls | 0 |
| time | 5.00s |
| ended_at | 2026-06-15T23:44:05.543314 |
| cost | $0.0335 |
| tokens | 412 in / 152 out |
| cache | 0 read / 8007 create |


## Knowledge Output

- field: function/local_accesses/staking_pool_specs::pool_id_spec
- data: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/pool_id_spec/function.local_accesses.json
- meta: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/pool_id_spec/local_accesses.meta.yaml
