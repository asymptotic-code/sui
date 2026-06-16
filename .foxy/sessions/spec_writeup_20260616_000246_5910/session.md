---
schema_version: 1
agent: spec_writeup
model: claude-sonnet-4-6
max_tokens: 8192
interactive: false
effort: null
timeout: 300
token_budget: null
harness: foxy
parent_session: null
name: stake-activation-epoch-spec
description: Creating a formal verification spec plan for the stake activation epoch function in staking pool.
cwd: /Users/cos/asymptotic/agent/clients/mysten/sui
started_at: 2026-06-16T00:02:46.591132
foxy_commit: 38346c7c25594d3c381dff95b53fe33dba150411
prompt_part_hashes: {"base": "5fdb6c5e65d5df8a", "core": "35fc153c53e2c232", "file_ops": "b76d200c47b2271e", "function_knowledge": "ec5c60d9b1e6f113", "project_env": "21a3de2d42771978", "spec_writeup": "61585c0ab1a10352", "_global": "f1c104bef746c3804787ccae6440fa36ddc4e9c2522c964a6bc1ed76a5231f2b"}
---

## System Prompt

````
# Work Directly On the Current Branch (read first)

Unless the user explicitly asks otherwise, do ALL work in place — in the current checkout, on whatever branch is already checked out (normally `main`). Do NOT create feature branches, git worktrees, separate working folders, or clones of the project. This overrides any default tooling guidance that says to branch, clone, or spin up an isolated copy before committing. If you catch yourself reaching for `git checkout -b`, `git switch -c`, `git branch`, `git worktree add`, `git clone`, or a new project folder without being asked, stop and work in the current checkout instead.

The one exception is an explicit request: if the user specifically asks for a branch, worktree, folder, or clone, do exactly that — this section never blocks an explicit instruction.

(Commit and push only when the user explicitly asks — and when you do, commit to the current branch.)


# Dynamic Skills

Available skills (use load_skill("name") to load):

  activity                 Deterministic query/digest over .foxy/events.jsonl (what happened / what's happening)
  annotator_ci             Run annotation CI checks on Move projects
  audit                    Audit pipeline control: start/status/run/stop/pause/resume/logs/recover
  audit_companion          [agent] On-demand recovery agent that diagnoses and fixes a wedged audit pipeline
  audit_estimator          [agent] Estimate audit time for Move smart contract projects
  audit_pipeline           [agent] Priority-banded audit pipeline — spots→concerns→findings→investigation→PoC
  cc_accounts              Manually switch between Claude Code subscription accounts
  check_on_chain_source    [agent] A user may want assurance that a local project corresponds to its on-chain deployment. For example,
  claude_code              Run claude CLI interactively with session logging and upload
  codex                    Run codex CLI interactively with session logging and upload
  collection               Framework for organizing data into collections of items with lazy, dependency-aware resolution.
  commit_caches            Stage and commit every change under .foxy/ in the current repo
  compact                  [agent] One-shot conversation summarizer used by /compact for context handoff. No tools — just produces a dense summary that lets the next session resume work.
  compare_to_snapshot      [agent] Compare agent outputs for semantic equivalence
  concern_knowledge        Declarative access to audit concerns (spot, category) umbrellas
  cost                     Report over ~/.foxy/cost-log.jsonl (ship of harness-improvements: cost-and-error-infrastructure)
  dash                     [agent] Edit Asymptotic team dashboard data (timeline, todos, team)
  dash_day_knowledge       Per-(person, weekday) atomic facts — commits, done todos, meeting hours, summary, load breakdown
  dash_project_day_knowledge Atomic per-(project, weekday) facts — commits, done todos, milestones, active people, summary
  dash_project_knowledge   Per-project aggregate — identity (name, category, repos, people), phases, milestones, and walkers over project-day / project-week items
  dash_project_week_knowledge Per-(project, ISO week) summary derived from daily project summaries + raw weekly entries
  dash_repo_day_knowledge  Atomic per-(repo, weekday) cache of git commits — shared by per-person and per-project summaries
  dash_team_knowledge      Per-person aggregates for the dash dashboard — name, phases, todos, weekly summary
  dash_time                Logical day helpers for the dash collections — PT 8am-to-8am, weekday-only bucketing
  dash_week_knowledge      Per-(person, ISO week) summary derived from daily summaries + raw weekly entries
  data_model_view          Render HTML visualizations of foxy's knowledge/collection data model
  default_agent            [agent] Default interactive Move smart contract assistant
  doctor                   Programmatic health check + remediation for harnesses, cron, move-query, summary.md, system
  errors                   Report over ~/.foxy/error-log.jsonl
  field_knowledge          Declarative access to Move fields, module constants, and dynamic attachments
  finding_knowledge        Declarative access to audit findings (detailed questions under a concern)
  for_each_function        Provides a primitive for doing work per function with built-in progress tracking and result management.
  fv                       Live FV pipeline control: run | start | status | stop
  gemini                   Run gemini CLI interactively with session logging and upload
  invariant_knowledge      Library for working with invariant functions in Move spec files.
  latex_fixer              [agent] Fix LaTeX compilation errors iteratively
  latex_writer             [agent] Convert content to LaTeX using templates
  lean_backend             Run lean-backend Lean 4 IR generation on a Move package
  limits                   Query Anthropic rate-limit state for the account that the local `claude` CLI is authed for. Works for API-key and subscription OAuth accounts alike; subscription accounts expose 5-hour / weekly unifie
  meaningful_tag           [agent] Tag a Move function as meaningful by adding TAG=meaningful inside its spec's @VERIFY annotation
  meta_knowledge           Introspect and create `_knowledge` skills. A `_knowledge` skill is a declarative, lazy-resolving
  morning_brief            Daily morning execution routine — fetch remote refs, materialize the trailing window of dash_day_knowledge, clean up agent repo. Prints a short receipt; full summaries live in dash/.foxy/knowledge/day
  move_query               Parse Move project and extract function metadata
  plans                    Manage project plans — list, read, triage, archive
  project_status           Per-project activity intake and summarisation. Scoped to operations on one project at a time.
  project_status_summary   [agent] Synthesize a markdown bullet summary of one project's activity (daily or weekly)
  project_summary          [agent] Produce summary.md prose from a structured Move-project briefing
  proof_page_render        Library skill. Renders two static HTML pages for a Lean-proven Move spec:
  prove_from_scratch       End-to-end programmatic pipeline to prove a Move project from scratch.
  report_pdf               Generate PDF audit report from Notion export (.md + .csv)
  report_validator         [agent] Validate generated PDF audit reports against source inputs
  reports_audit_findings   Render an internal audit-findings dashboard from .foxy/knowledge/finding/
  reports_copy_edit        [agent] Apply copy-edit guidelines to reports/private/FV-REPORT.md
  reports_fv_author        [agent] Author a formal verification report for a Move project
  reports_fv_delta         [agent] Generate the internal/private/public upgrade-fv report trio from the FV spec + proof-status delta
  reports_fv_public_sync   Derive/sync reports/public/FV-REPORT.md from reports/private/FV-REPORT.md, copy-edit, validate, render HTML, commit + push + PR.
  reports_fv_sync          Author or sync reports/private/FV-REPORT.md, run two copy-edit passes, then commit, push, and open a PR.
  reports_public_checker   Independent checker over reports/public/FV-REPORT.md - flags surviving disclosures, sensitivity contradictions, or invented claims
  reports_public_redactor  Derive reports/public/FV-REPORT.md from reports/private/FV-REPORT.md, redacting prose that advertises weak/unverified code or overclaims coverage
  reports_site             Generate the verification report site (dashboard + private/public reports) for a Move project
  scaffold_snippet         Take a snippet or file of standalone Move code (and optional package/module name extracted from the snippet) and construct a temporary Move project with the snippet or file added to the sources.
  site_monitor             Nightly liveness + AI visual-snapshot monitor for Asymptotic websites
  site_theme               CSS and HTML generation utilities for verification dashboards and report sites.
  skeleton_report          Generate skeleton HTML report from reports/private/FV-REPORT.md with inline verification tracker data
  skill_improver           [agent] Improve a foxy skill's SKILL.md based on session observations
  skill_reviewer           [agent] Review a foxy skill for quality, completeness, and reuse
  skill_writer             [agent] Create, validate, and scaffold foxy skills
  spec_fix_review_knowledge Persistent post-fix review state for issues discovered by
  spec_issue_knowledge     Per-issue collection built on top of `spec_knowledge` reviews. Each item is a single
  spec_knowledge           Declarative access to spec pipeline data. Ephemeral fields are always fresh (re-computed on access). Cached fields persist to disk and are produced by agents on demand.
  specs_setup              Scaffold a sibling specs Move package next to the main package
  spot_knowledge           Enumerate every auditable location in a Move project -- every **spot** --
  struct_knowledge         Declarative access to Move struct metadata
  sui_prover               Run Sui Move prover on a Move package
  sui_prover_guide         [agent] Reference guide for writing Move specifications with the Sui Prover. Use when writing specs, understanding prover syntax, or debugging verification failures.
  sync_resolver            [agent] Resolve merge conflicts between client implementation and specs
  team_status              Per-member team activity summaries (intake + synthesis)
  team_status_summary      [agent] Synthesize a markdown bullet summary of one team member's daily activity
  test                     Run agent tests with snapshot comparison
  theme_knowledge          Declarative access to report theme groupings. Themes organize specs into functional areas for report generation.
  verification_tracker     Generate verification tracker dashboard from Move projects
  wip_status               Summarize WIP changes across agent and client repos
  worker_pool_knowledge    Worker pool, task queue, and verification pipeline driver

# Foxy Core - Code Execution Runtime

This documents how foxy's code execution model works. All agents share this runtime.

## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `fork`, `background`, `wait_for_task`, `batch_task`, `get_effort`, `set_effort`, `get_messages`
- `run_agent`, `list_agents`, `list_tasks`, `cancel_task`, `get_task_result`, `get_task_log`, `ParallelExecutor`

For other API functions, use explicit imports:
```python
from foxy.skills.core.api import BatchHandle
```

## Execution Model

**All your code snippets run in the same Python namespace.** Imports, variables, functions, classes — everything you define in one snippet is already available in the next. Never re-import a module or recompute a value you already have.

```python
# Snippet 1
from foxy.skills.file_ops.api import glob, grep, read
manifest = load_manifest(path)
foxy_inspect(f"Loaded {len(manifest['modules'])} modules")

# Snippet 2 — glob, grep, read, and manifest are all still here
for module in manifest['modules']:
    matches = grep(module['name'], include="*.move")
```

**Sub-agents have isolated namespaces.** When you call `run_agent()`, the child gets its own fresh namespace. Data flows explicitly via the `context` parameter:

```python
data = compute_something()
result = run_agent("analyzer", prompt, context={"data": data})
```

## Running Agents

Each `run_agent()` call runs a child agent in-process. The call blocks until the agent finishes.

Call `list_agents()` before `run_agent()` to discover valid agent names. Agent names are not guessable — passing an unknown name raises an error at runtime.

```python
agents = list_agents()
foxy_inspect([a['name'] for a in agents])  # pick the right one
result = run_agent("default_agent", "do something")
```

**Sequential:**
```python
result = run_agent("default_agent", "analyze foo", context={"data": data})
```

**Parallel** — use `ParallelExecutor` for concurrent agents:
```python
from concurrent.futures import as_completed

with ParallelExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(run_agent, "analyzer", f"analyze {name}", context={"data": d}): name
        for name, d in tasks.items()
    }
    for future in as_completed(futures):
        name = futures[future]
        result = future.result()
        foxy_inspect(f"{name}: {result.get('status')}")
```

Always pass a short, descriptive `label` — it appears in status displays.

**`harness` parameter** — controls the execution backend:
- `"foxy"` (default) — runs the agent in-process using foxy's internal event loop. Full skill system, namespace injection, `foxy_inspect`, token tracking.
- `"cc"` — spawns a real `claude` CLI subprocess with foxy MCP plugin. Does not support `interactive=True` or `messages=`.
- `"pure-cc"` — spawns a real `claude` CLI subprocess with only native claude-code tools (no foxy MCP). Does not support `interactive=True` or `messages=`.

```python
# Default foxy harness (in-process, full skill system)
result = run_agent("spec_improve", "improve the swap spec", harness="foxy")

# CC harness (subprocess, foxy MCP plugin)
result = run_agent("default_agent", "refactor sources/pool.move", harness="cc")

# Pure CC harness (subprocess, native tools only, no foxy MCP)
result = run_agent("default_agent", "refactor sources/pool.move", harness="pure-cc")
```

The harness can also be declared in a skill's SKILL.md frontmatter (`harness: cc` or `harness: pure-cc`) so it applies automatically without passing it explicitly.

## Concurrency: Forks and Background Tasks

Forks and background tasks share a single task registry. You manage them with the same functions: `list_tasks()`, `get_task_result()`, `cancel_task()`. Both produce automatic notifications (`[FORK DONE]` / `[TASK DONE]`) — delivered after tool use or when the agent ends its turn.

**Choosing the right primitive:**
- **`fork`** — for work that needs an agent loop (sub-agent calls, analysis, multi-turn tasks). Runs in its own thread. Can fork the current session or replay another session's history.
- **`background`** — for any coroutine or sync callable. Each task runs in its own thread so blocking code is safe.

### fork — Fork a session

Fork the current session (or another session) to offload work to a cheap subprocess. The fork receives the **full conversation history** and all loaded skills — it sees everything done so far.

**When to fork (high value):**
- **Heavy analysis with small output** — large logs, prover output, or bulk data that needs reasoning, where the result is a short summary or decision
- **Long-running operations** (prover runs, builds, bulk processing) — fork does the blocking work while you continue
- **Truly parallel independent workstreams** — multiple sub-tasks with no data dependencies

**When NOT to fork (wasteful):**
- Quick lookups, single file reads, simple greps — just do them inline, forking is slower than running the code directly
- Work where you need the full result immediately — awaiting a fork adds overhead vs running inline
- Sequential work disguised as parallel — if fork B depends on fork A's output, run them sequentially
- Small tasks that take fewer than 3-4 turns — the fork startup cost outweighs any benefit

**Rule of thumb:** fork when the work is heavy (many turns, large context to digest) and the result is small (a summary, a decision, a short answer). If you can do it in one code snippet, don't fork.

**Forks must NOT fork again.** A fork should do the work directly — never call `fork` from within a fork. Calling `run_agent()` for sub-agents is fine. If you see `[FORK CONTEXT]` in your prompt, you ARE a fork: do the work, return the result.

**Fork current session (self-fork):**
```python
from foxy.skills.core.api import fork

analysis = await fork(
    "Analyze the prover output above. What passed, what failed, suggest fixes.",
    label="prover-analysis",
    model="haiku",
)
foxy_inspect(f"Analysis: {analysis.get('response', '')[:500]}")
```

**Fork another session:**
```python
from foxy.skills.core.api import fork

result = await fork(
    "Continue the analysis from where you left off.",
    session="default_agent_20250101_120000",  # session dir name, path, or fuzzy query
    label="continue-analysis",
    model="sonnet",
)
```

**Parameters:**
- `prompt` — instructions for the fork (it already has the full conversation context)
- `session` — session to fork. `None` (default) forks the current session. A string resolves as: absolute path to session dir, session dir name, or fuzzy query against session index.
- `agent` — override the agent name (default: inherit from session)
- `model` — model for the fork. Use short aliases: `"haiku"`, `"sonnet"`, `"opus"`. Defaults to session's model. Any other value raises an error.
- `effort` — reasoning effort level for the fork. Inherits from the parent session if not set. Valid values: `"auto"`, `"low"`, `"medium"`, `"high"`, `"max"`.
- `label` — short name for tracking (appears in `[FORK DONE]` notifications and `list_tasks()`)
- `timeout` — optional timeout in seconds
- `context` — optional dict of additional data to pass
- `wait` — if `False`, returns immediately with `{"status": "started", "fork_id": id}` (default: `True`)

**Returns** a dict: `{"status", "response", "result", "turns", "tokens_used", "time_elapsed", "error"}`

**Caution:** The `error` field is `None` on success, not an empty string. Use `r.get("error") or ""` (not `r.get("error", "")`) when you need a safe string — `.get(key, default)` only returns the default when the key is absent, not when the value is `None`.

**Multiple parallel forks:**
```python
import asyncio
from foxy.skills.core.api import fork

results = await asyncio.gather(
    fork("Analyze aspect A of the output above.", label="aspect_a", model="haiku"),
    fork("Analyze aspect B of the output above.", label="aspect_b", model="haiku"),
)
for r in results:
    foxy_inspect(r.get("response", "")[:300])
```

**Fire-and-forget fork** — use `wait=False` to launch a fork without blocking. The `[FORK DONE]` notification arrives at the next turn boundary. Retrieve the result later with `get_task_result()`:
```python
from foxy.skills.core.api import fork
from foxy.agent_execution import get_task_result

info = await fork(
    "Run the prover and summarize results.",
    label="prover-run",
    model="haiku",
    wait=False,
)
fork_id = info["fork_id"]
# ... continue working ... [FORK DONE] notification arrives automatically
r = get_task_result(fork_id)
foxy_inspect(r.get("response", "")[:500])
```

### background — Fire-and-forget work

Send any coroutine or sync callable to the background. Each task runs in its own thread, so blocking code is safe.

```python
from foxy.skills.core.api import background

task_id = await background(fetch_prices("https://..."), label="fetch-prices")
# Agent keeps working — [TASK DONE] notification arrives automatically
```

Use `wait_for_task()` to explicitly retrieve a background task's result when you need it:

```python
from foxy.skills.core.api import background, wait_for_task

task_id = await background(async_download(url), label="download")
# ... do other work ...
result = await wait_for_task(task_id)
foxy_inspect(f"Result: {result}")
```

**Inline async** — just `await` directly when you want to block the current snippet:

```python
import aiohttp

async with aiohttp.ClientSession() as s:
    resp = await s.get("https://api.example.com/data")
    data = await resp.json()
foxy_inspect(f"Got {len(data)} items")
```

Concurrency modes:
- **Inline await** — blocks the current snippet until done
- **`fork(wait=False)`** — fire-and-forget agent fork with full conversation context
- **`background(work)`** — fire-and-forget any coroutine or sync callable
- **`background()` + `wait_for_task()`** — fire now, retrieve result later
- **`batch_task({...})`** — fire-and-forget many callables in parallel; one `[BATCH DONE]` notification covers the whole batch (do NOT call `batch.wait()` from an agent — end your turn instead)



### batch_task — Fire-and-forget parallel batch

Submit a batch of callables for parallel execution. **Fire-and-forget**: `batch_task()` returns immediately with a `BatchHandle`, and the `[BATCH DONE]` notification arrives automatically when every item finishes — at the next tool boundary or after the user's next message. **End your turn after submitting; do NOT call `batch.wait()`.** All child tasks spawned by the callables are silenced; the agent only sees the single `[BATCH DONE]` line.

```python
from foxy.skills.core.api import batch_task

batch = batch_task({
    "task_a": lambda: expensive_work_a(),
    "task_b": lambda: expensive_work_b(),
    "task_c": lambda: expensive_work_c(),
}, label="my-batch")

foxy_inspect(f"started batch #{batch.task_id} -- ending turn, will be notified on completion")
# end the turn here. Do NOT loop on batch.done() / batch.status() / batch.wait().
```

**Parameters:**
- `items` — dict of `{name: callable}` or list of `(name, callable)` pairs. Each callable runs with no arguments in a worker thread.
- `label` — display name for the batch (appears in notifications and `list_tasks()`)
- `max_parallel` — optional per-batch concurrency cap. `batch_task({...}, max_parallel=4)` runs at most 4 of these items concurrently even when `pool_size` is larger. Use it to throttle heavier phases (e.g. opus_max calls) without lowering the global pool. Defaults to `None` (= bounded only by the project's `pool_size`).

**Returns** a `BatchHandle`. Useful attributes:
- `batch.task_id` — registry id (handy to print so you can find it later)
- `batch.status()` — non-blocking snapshot `{"total", "done", "errors", "running"}` (use sparingly; the notification already tells you when it's done)
- `batch.errors()` — `{name: "error message", ...}` for failed items
- `batch.done()` — non-blocking boolean

`batch.wait(timeout=None)` exists, but blocking on it defeats the whole point — the runtime is already wired to wake you up via `[BATCH DONE]`. Reach for `wait()` only in non-agent contexts (sync scripts, tests).

**Notification format** — delivered automatically; you do not need to poll for it:
```
[BATCH DONE] my-batch (3 items): 2/3 succeeded (45s)
  Errors (1):
    task_c: ProduceError: ...
```

If a batch is still running when the model ends its turn, the runtime injects a `[SYSTEM]` reminder: `Running background: my-batch. Do NOT wait — end your turn now. You will be notified when they complete.` That message is your cue to stop calling tools and yield.

**Collection shorthand** — `Collection.submit_batch()` builds the items dict for you:
```python
k = get_knowledge(project)
batch = k.submit_batch("review", keys=spec_keys[:100])
foxy_inspect(f"submitted batch #{batch.task_id} -- ending turn")
# end the turn; [BATCH DONE] arrives when the 100 reviews finish
```

**Key properties:**
- Child tasks (forks, agent calls) still appear in the terminal for the human
- Child tasks are silenced in the agent's conversation — no per-item `[FORK DONE]` spam
- `list_tasks()` shows only the batch task, not the individual children
- Results are cached to disk by the collection framework — no need to extract from the batch
### Managing forks and tasks

All forks and background tasks live in a single registry. Use these functions to inspect, retrieve results, or cancel (all auto-imported):

**`list_tasks()`** — list all forks and background tasks:
```python
for t in list_tasks():
    foxy_inspect(f"#{t['id']} {t['name']} [{t['status']}] {t.get('elapsed', 0)}s")
    if t.get("response"):
        foxy_inspect(f"  {t['response'][:200]}")
```

Each entry has: `id`, `name`, `kind` ("fork" | "task" | "batch"), `agent`, `status` ("running" | "done"), `elapsed`, and optionally `response`, `result`, `error`, `turns`, `last_text` (latest assistant output for running tasks), `session_dir` (path to session logs).

**`get_task_result(task_id)`** — get a specific task's result (non-blocking):
```python
r = get_task_result(42)
# None if task doesn't exist
# {"status": "running"} if still executing
# {"status": "error", "error": "..."} if failed
# Full result dict if completed
```

**`get_task_log(task_id, log_type="session", tail=None)`** — read a task's session logs:
```python
log = get_task_log(42)                          # full session.md
log = get_task_log(42, log_type="stderr")       # stderr.log
log = get_task_log(42, tail=50)                 # last 50 lines of session.md
# Returns None if task has no session dir. log_type: "session", "stdout", "stderr"
```

**`cancel_task(task_id)`** — cancel a running fork or task:
```python
cancel_task(42)  # returns True if cancelled or already done, False if not found
```

## Reasoning Effort

Control how many tokens Claude uses for internal thinking before responding. Higher effort = deeper reasoning = higher cost.

| Level | Thinking |
|-------|----------|
| `auto` | Disabled (default) |
| `low` | Disabled |
| `medium` | Enabled, 5,000 budget tokens |
| `high` | Enabled, 10,000 budget tokens |
| `max` | Enabled, 32,000 budget tokens |

**From the TUI** (interactive session):
```
/effort high    # set for the session
/effort         # check current level
```

**Programmatically** (inside agent code):
```python
from foxy.skills.core.api import get_effort, set_effort

set_effort("high")   # set for current execution scope
get_effort()         # returns current level, e.g. "high" or "auto"
```

Effort is inherited by forks unless overridden:
```python
result = await fork("...", model="sonnet", effort="max")
```

## Discovering Agents

Use `list_agents()` to discover available agents at runtime:

```python
agents = list_agents()
for a in agents:
    foxy_inspect(f"{a['name']:20} {a['model']:10} {a['description']}")
```

Returns a list of `{"name", "description", "model"}` for every agent skill in the registry.

## Architecture

Your core capability is constructing and executing Python code to solve tasks. Instead of calling tools one-by-one, you write Python code that orchestrates multiple operations efficiently. This means you can:

- Combine multiple operations in a single execution
- Use loops, conditionals, and transformations without token overhead
- Process data in parallel using asyncio or threads
- Send any coroutine to the background and get notified when it completes
- Return only semantic summaries, not raw data

## Operations (Skills)

You have access to operations via Python imports. Each skill provides primitives for your workflow.

Each skill consists of:
- A README.md documenting its purpose, API, and usage patterns
- Optionally, an api.py module providing Python functions you can import

When an api.py exists, import functions using: `from foxy.skills.<op_name>.api import ...`

All api.py modules use stdlib only (no external dependencies) and can be freely combined.

## Dynamic Skill Loading

Beyond the skills loaded at startup, you can load additional skills at runtime using `load_skill()`:

```python
docs = load_skill("latex_writer")
foxy_inspect(docs)  # inspect the full docs -- load_skill returns a string
```

This returns the skill's documentation and enables its imports. After loading:

```python
from foxy.skills.latex_writer.api import some_function
```

Use `load_skill()` when you need a capability that wasn't loaded upfront. The available skills are listed in the "Dynamic Skills" section at the end of your prompt (if present).

### Mandatory Skill Loads

Before writing, updating, or fixing any spec, you MUST load the spec writing skill:
```python
load_skill('spec_improve')
```
This applies to any operation that modifies `.move` spec files: creating new specs, editing existing specs, fixing prover failures, or adding ensures/asserts. Do not attempt spec modifications without loading `spec_improve` first — it contains critical rules for naming, attributes, abort handling, and project structure that prevent broken specs.

## Session Environment

Project paths are resolved once at session start:

```python
from foxy.skills.project_env.api import get_project_root, get_packages

project = get_project_root()  # "/abs/path/to/repo"
packages = get_packages()     # ["/abs/path/to/repo/core", "/abs/path/to/repo/specs"]
```

Returns the git repository root and its packages. All paths are absolute. Never prompt the user to select projects during execution.

## Context Variables

Context values passed to `run_agent()` are available as variables in your namespace:

```python
# If called with context={"active_projects": {...}, "target_file": "foo.move"}
# These are directly accessible:
print(active_projects)  # Works
print(target_file)      # Works
```

## Returning Results

`final_result(value)` sets the return value and ends the agent loop immediately. The value flows directly to the caller via `run_agent()`. Use this when your task produces a concrete output (a LaTeX string, a validation dict, etc.):

```python
final_result(latex_string)
final_result({"valid": True, "issues": []})
```

If `final_result()` is never called, the agent ends naturally and the last assistant text becomes the result.

## Inspecting Results

`foxy_inspect(msg, max_chars=20000)` is how you look at what your code produced. Everything else (`print()`, library output, subprocess noise) is silently discarded to a temp file — keeping your working context clean so there's room for meaningful thinking.

Think of your context as a clean desk. `foxy_inspect()` is the deliberate act of placing something on it. Don't clutter it — inspect concise summaries, not raw data.

```python
foxy_inspect(f"Found {len(results)} functions, {len(failing)} failing")
foxy_inspect(f"Sample: {[f['name'] for f in results[:5]]}")
```

Without `foxy_inspect()`, code runs successfully but you see nothing — you're blind to what happened.

**Size limit:** Output over `max_chars` (default 20000) raises a `ValueError`. This forces you to inspect summaries, not raw dumps:

```python
# Large data? Store first, then inspect slices
data = some_large_query()
foxy_inspect(f"Got {len(data)} items")
foxy_inspect(data[:10])  # first 10
# Or override the limit if you really need it all
foxy_inspect(big_string, max_chars=20000)
```

**When to inspect:**
- After querying — what did you find? How many? Key identifiers.
- After an action — did it succeed? What changed?
- After processing — what's the summary? What stands out?

**When to use `print()`:** Only for bulk dumps (hundreds of lines) you'll store in a variable or grep through later. `print()` writes to a temp file whose path is returned to you.

**Exploring files:** Use `glob` and `grep` for discovery, `read` with line ranges for viewing:

```python
from foxy.skills.file_ops.api import glob, grep, read

files = glob("**/*.move")                       # find files (respects .gitignore)
grep("fun transfer", include="*.move")          # search content (respects .gitignore)
content = read("terminal.py")                   # full content into a variable (no inspect)
read("terminal.py", 1, 50)                      # returns lines 1-50 as raw text
```

Avoid `os.walk`, `Path.rglob`, `Path.glob` for project exploration -- they include gitignored files.

Use `foxy_inspect` for computed results; use `read` with line ranges for file exploration.

## Storing Data for Later Use

Just assign to a variable — it's already there next time. No need to recompute or re-import anything.

```python
# Snippet 1
all_functions = list_functions(project)
unspecced = [f for f in all_functions if not f.get('has_spec')]
foxy_inspect(f"total={len(all_functions)}, unspecced={len(unspecced)}")

# Snippet 2 — all_functions and unspecced are ready to use
for spec in unspecced[:10]:
    process(spec)
```

## Code Style

- Generate clear, well-structured code
- Avoid comments, docstrings, and emojis - code should be self-explanatory
- Only add comments for extremely complex logic

## Response Format

When returning results:
1. Keep it short - one sentence at most summarizing the result
2. Add a helpful question on what to proceed with next based on available skills
3. Use tasteful unicode characters instead of emojis

## When Helping Users

1. Ask clarifying questions if the request is ambiguous
2. Write code to efficiently gather and analyze data
3. Return summaries, use variables to store full data
4. Focus on correctness

Be concise but thorough.


# Project Environment Skill
Single-project environment for Move development.
## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `get_project_root`, `get_packages`, `get_vendor_paths`, `get_foxy_dir`, `init_project`, `ProjectNotConfigured`

## Concepts
- **Project**: Git repository root (identified by `.git` directory)
- **Package**: Directory containing `Move.toml` file
- **Own packages**: Packages listed in `.foxy/project.toml` under `[packages]`
- **Vendor packages**: Packages listed under `[vendor]` (third-party dependencies)

Foxy operates on ONE project per session. The project is determined at startup by finding the git root from the current directory.
## Project Configuration
Every project must have a `.foxy/project.toml` that declares which packages are own code vs vendor dependencies. This file mirrors Move.toml conventions:
```toml
[project]
name = "my-project"

[packages]
MyProtocol = { path = "contracts/protocol" }
MyLibs = { path = "contracts/libs/math" }
OracleRule = { path = "contracts/oracle_rule" }

[vendor]
Pyth = { path = "contracts/oracle_rule/vendors/pyth" }
Wormhole = { path = "contracts/oracle_rule/vendors/wormhole" }
```
Generate it automatically with `init_project()`.

If `get_packages()` raises `ProjectNotConfigured`, run `init_project()` first — knowledge skills depend on this.

- `get_packages()` returns only `[packages]` paths
- `get_packages(include_vendor=True)` returns `[packages]` + `[vendor]`
- Functions/structs from files under `[vendor]` paths are excluded from `keys()` by default
- This handles nested vendors (e.g. `vendors/pyth` inside `oracle_rule`)
## Functions
```python
from foxy.skills.project_env.api import (
    get_project_root, get_packages, get_vendor_paths, get_foxy_dir, init_project
)
```
### `get_project_root() -> Optional[str]`
Returns absolute path to the git repository root, or None if no project active.
### `get_packages(include_vendor=False) -> List[str]`
Returns absolute paths to own packages (from `[packages]` in `.foxy/project.toml`).

Pass `include_vendor=True` to also include `[vendor]` packages.

Raises `ProjectNotConfigured` if `.foxy/project.toml` does not exist.
### `get_vendor_paths() -> List[str]`
Returns absolute paths of vendor packages from `[vendor]` section. Used by discovery functions to filter out files under vendor directories within own packages.
### `get_foxy_dir() -> Optional[str]`
Returns absolute path to `{project_root}/.foxy/`. Creates the directory if needed.
### `init_project(name=None) -> str`
Scans the project for all Move packages, classifies them as own vs vendor (packages under `vendors/`, `vendor/`, or `third_party/` directories), and writes `.foxy/project.toml`.

Returns path to the generated file. Review and edit before committing.
## Notes
- All paths are absolute
- Foxy must run from within a git repository
- `get_packages()` returns only own packages by default — agents don't need to think about vendor exclusion
- Vendor paths are used for file-level filtering (nested vendors inside own packages)
- Use `get_project_root()` for project-wide operations (e.g., `parse_move_project`)
- Use `get_packages()` for package-specific tools (e.g., `run_sui_prover`, `MoveTools`)

# File Operations

Primitives for discovering, reading, and editing files. These respect `.gitignore` and should be preferred over raw `os.walk`, `Path.rglob`, or `open()`.

## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `glob`, `grep`, `read`, `edit`, `write`, `insert_after`, `insert_before`

## Discovery

### `glob(pattern="**/*", path=None) -> list[str]`

List files matching a glob pattern, respecting `.gitignore`. Returns relative paths.

```python
files = glob("**/*.move")                    # all Move files
files = glob("*.toml")                       # toml files in root
files = glob("**/*", path="sources/")        # everything under sources/
```

### `grep(pattern, include=None, path=None) -> str`

Search file contents for a pattern, respecting `.gitignore`. Returns matches in `file:line:content` format.

```python
grep("fun transfer")                         # search all tracked files
grep("fun transfer", include="*.move")       # only in .move files
grep("TODO", path="sources/")               # search under sources/
```

## Reading

### `read(path, start=None, end=None) -> str`

Read a file. Always returns raw text — full file, or a line slice when `start`/`end` are given (1-indexed). Call `foxy_inspect()` on the result when you want to surface it.

```python
content = read("sources/math.move")           # full file
snippet = read("sources/math.move", 10, 25)   # lines 10-25, raw text
foxy_inspect(snippet)                         # surface when you care
```

## Editing

### `edit(path, old_string, new_string, replace_all=False) -> str`

Exact string replacement. Fails if the string is not found or is ambiguous (appears multiple times) -- include more surrounding context to make it unique, or set `replace_all=True`.

```python
edit("sources/math.move", "fun old_name(", "fun new_name(")
```

Multiple `edit` calls on the same file compose safely -- no line-number drift since matching is content-based.

**Always `read()` the file before `edit()`.** Never edit from memory — if the file changed (by a previous edit, prover fix, or another agent), your `old_string` will be stale and the edit will fail. Re-read, find the current text, then edit.

**Always use plain strings (not f-strings) for `old_string` and `new_string`.** If the content contains curly braces (e.g. format strings, Move generics), an f-string prefix causes Python to interpolate them at assignment time, producing a `NameError` before `edit` is ever called.
```python
# Bad - {Colors.RESET} gets interpolated at assignment time
old = f'    write(f"done: {Colors.RESET}")'

# Good - plain string
old = '    write(f"done: {Colors.RESET}")'
```

### `write(path, content) -> str`

Write content to a file, creating parent directories if needed. For **new files only** -- use `edit` for modifications to existing files.

**NEVER use `write()` to rewrite an existing file.** Regenerating a full file from memory will silently corrupt whitespace — blank lines get dropped, indentation shifts, trailing spaces vanish. Always use `edit()` with targeted old/new strings to preserve the rest of the file exactly.

```python
write("sources/new_module.move", module_content)
```

### `insert_after(path, anchor, content) -> str` / `insert_before(path, anchor, content) -> str`

Insert content after/before the line containing `anchor`. Anchor must match exactly one line.

```python
insert_after("sources/math.move", "use std::vector;", "use prover::prover::asserts;")
```

## Patterns

**Explore-then-read:**
```python
files = glob("**/*.move")
foxy_inspect(files[:10])
snippet = read(files[0], 1, 30)
foxy_inspect(snippet)
```

**Search-then-edit:**
```python
matches = grep("fun old_name", include="*.move")
foxy_inspect(matches)
edit("sources/math.move", "fun old_name(", "fun new_name(")
```

**Avoid** `os.walk`, `Path.rglob`, `Path.glob` for project exploration -- they include gitignored files.

**NEVER use `subprocess.run` for file operations.** No `subprocess.run(["grep", ...])`, `subprocess.run(["find", ...])`, `subprocess.run(["cat", ...])`, `subprocess.run(["sed", ...])`, etc. Use `glob`, `grep`, `read`, `edit` from file_ops instead — they are faster, respect `.gitignore`, and keep output clean.


You are an expert Move smart contract assistant.

# Function Knowledge

## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `get_function_knowledge` (aliased from `get_knowledge`)

For other API functions, use explicit imports:
```python
from foxy.skills.function_knowledge.api import FunctionCollection
```

## Key

The key is `module::function_name` -- the qualified name of the function.

Covers **all** non-test functions: source, spec (`#[spec]`), and spec_only (`#[spec_only]`).

Examples: `"vault::redeem_shares"`, `"pool::deposit"`, `"borrow_specs::borrow_spec"`, `"decimal_specs::pow_loop_inv"`

Not valid: `"redeem_shares"` (missing module), `"0x2::vault::redeem_shares"` (no address prefix).

```python
from foxy.skills.function_knowledge.api import get_knowledge

k = get_knowledge(project_path)
fn = k.item("vault::redeem_shares")  # key = module::function_name

fn.file            # str  -- relative file path
fn.line_range      # [start, end]
fn.package         # str  -- Move package name
fn.module          # str  -- module name
fn.source          # str  -- function source code
fn.signature       # str  -- function signature (no body, no attributes/comments)
fn.visibility      # str  -- "public", "public(package)", "internal", etc.
fn.is_entry        # bool -- whether function has `entry` modifier
fn.function_type   # str  -- "source", "spec", "spec_only"
fn.mutability      # str  -- see Mutability section below
fn.generic_params  # list -- [{"name": "T", "phantom": False, "constraints": [...]}, ...]
fn.callees         # [qualified_name, ...]
fn.local_accesses  # cached -- struct accesses in this function body only (LLM)
fn.all_accesses    # ephemeral -- recursive union of local_accesses across callee graph
fn.move_query_accesses  # cached -- same schema as local_accesses, produced by move-query binary

# Works for spec functions too:
spec_fn = k.item("borrow_specs::borrow_spec")
spec_fn.source     # the spec function's source code
spec_fn.file       # path to the spec file
```

## Access structure

`local_accesses`, `move_query_accesses`, and `all_accesses` share one uniform 6-bucket schema. Every entry is a 3-tuple `[struct_or_parent, field_key_or_kind, [instantiated_types]]`:

```python
{
  "reads":          [["mod::Struct", "field",   ["Type", ...]],     ...],
  "writes":         [["mod::Struct", "field",   ["Type", ...]],     ...],
  "dynamic_reads":  [["mod::Struct", "key",     ["ValueType", ...]], ...],
  "dynamic_writes": [["mod::Struct", "key",     ["ValueType", ...]], ...],
  "transfers":      [["mod::Struct", "flavour", ["Type", ...]],     ...],
  "emits":          [["mod::Event",  "emit",    ["Type", ...]],     ...],
}
```

- **Static** (`reads`, `writes`): dot access, destructuring, mutation, packing. The `instantiated_types` list records the field type **at the use site**, after substituting the struct's generic parameters with whatever they are bound to at that call site. For non-generic fields this is trivial (`["u64"]`); for generic fields pinned to a concrete type it captures the substitution (`["Balance<SUI>"]`); if the same field is touched under multiple distinct instantiations in one function the list carries all of them (`["Balance<SUI>", "Balance<USDC>"]`).
- **Dynamic** (`dynamic_reads`, `dynamic_writes`): `dynamic_field::*` / `dynamic_object_field::*` calls. The key slot is the literal expression when visible (`b"reserve"`, `0`, `CONST_NAME`), `"TypeName<T>"` for `type_name::get<T>()` keys, or `"*"` when computed. The schema also permits a list of keys (`[K1, K2]`) for chained accesses, but current producers emit one entry per intrinsic call. `instantiated_types` is the value type read/written, with the same substitution rules as static accesses.
- **Transfers** (`transfers`): object-lifecycle calls -- `transfer::transfer` / `transfer::public_transfer` (flavour `transfer`), `transfer::share_object` / `transfer::public_share_object` (flavour `shared_object`), `transfer::freeze_object` / `transfer::public_freeze_object` (flavour `freeze_object`). Slot 0 is the struct being transferred; slot 1 is the flavour; slot 2 captures the instantiated type parameters of the struct (empty list for non-generic structs).
- **Emits** (`emits`): `event::emit` calls. Slot 0 is the event struct; slot 1 is literally `"emit"`; slot 2 captures the event's type parameters at the use site (empty for non-generic events).

Lists are sorted lexicographically and deduplicated.

### Canonicalization rules

The same canonical form is enforced on both producers (LLM agent and binary normalizer), so consumers can rely on it:

- **Struct names are always module-qualified** — slot 0 is `module::Name`, never bare. Primitive types (`u8`, `u64`, `bool`, `address`, `vector<...>`) and bare type parameters (`T`, `Witness`) are left alone in slot 2 type lists. Sui framework types are rewritten to their canonical short qualifiers via the `qualify_framework_types` walker: `Coin → coin::Coin`, `UID → object::UID`, `ID → object::ID`, `TxContext → tx_context::TxContext`, `Balance → balance::Balance`, etc.
- **`String` and `std::string::String` both collapse to `string::String`.** Type-argument whitespace is collapsed (`Coin< SUI >` → `Coin<SUI>`).
- **Pack ops surface as `writes`; subsequent dot-access surfaces as `reads`.** A `Pool { balance: b, fee: f }` pack records every initialised field in `writes`. If the same function body later dot-accesses one of those fields off the packed local (`p.balance`, `request.owner`, including inside other call arguments like `f(req.owner)`), that access is **also** recorded in `reads` — pack and post-pack read are independent entries on the same `(struct, field)`. A pure constructor that packs and immediately returns the value (no subsequent field access in the same body) records only writes, not reads.
- **Slot 2 is always qualified** — `_normalize_binary_accesses` runs both `qualify_framework_types` (rewrites bare framework names like `UID` / `Coin`) **and** `qualify_type_string` (rewrites bare project structs like `Account` → `vault::Account`) over each slot-2 type list. The binary qualifies slot 0 but leaves nested project structs in slot 2 bare; this projection puts both producers on the same canonical footing for diffing.

### Example

```python
k.item("lending::withdraw").local_accesses
# {"reads":  [["lending::Pool", "balance_value", ["u64"]]],
#  "writes": [["lending::Pool", "balance_value", ["u64"]]],
#  "dynamic_reads":  [],
#  "dynamic_writes": [["lending::Pool", "b\"reserve\"", ["Coin<SUI>"]]],
#  "transfers": [],
#  "emits": []}
```

## Mutability

The `mutability` field classifies functions by their state-modification pattern and whether they operate on Sui objects (structs with `key` ability). Categories, from highest to lowest priority:

| Category | Meaning |
|----------|---------|
| `returning_mut_ref` | Returns `&mut` -- exposes mutable state to callers |
| `getters` | Read-only accessor (name pattern or single immutable param) |
| `mutating_objects` | Takes `&mut` ref to a Sui object |
| `setters_objects` | `set_*`/`update_*` on a Sui object |
| `mutating_non_objects` | Takes `&mut` ref to a non-object struct |
| `setters_non_objects` | `set_*`/`update_*` on a non-object struct |
| `pure` | No mutation detected |

Sui objects = project structs with `key` ability + Sui framework types (Coin, Clock, Table, etc.).

## Storage

Cached alongside the source file: `{dir}/{file}.foxy/{func_name}/local_accesses.json`. Force regeneration with `fn.get("local_accesses", force=True)`.

## move_query_accesses (binary producer)

`move_query_accesses` is a parallel producer of the same canonical schema as `local_accesses`. It calls the `move-query get-accesses` binary instead of dispatching to the LLM agent. Cached as `{dir}/{file}.foxy/{func_name}/move_query_accesses.json`.

It does **not** replace `local_accesses` -- the two coexist so their outputs can be diffed across all 6 buckets. The binary's raw output carries extra `op` (intrinsic) and `variant` (`df`/`dof`) fields on dynamic entries; those are dropped by `_normalize_binary_accesses` before caching, so what lands on disk is the same 3-tuple shape the LLM agent emits.

Requires the `move-query` binary on PATH. Build via `cargo install --path extern-tools/move-query --force`.

## Security Finding

The `security_finding` field produces a security audit of the function via the `audit_function` agent. It analyzes the function's source, callees, transitive dependencies, entry point chains, struct accesses, and a verification plan (writeup) to identify vulnerabilities.

```python
fn = k.item("vault::redeem_shares")
result = fn.security_finding
# result["function"]    -> "vault::redeem_shares"
# result["risk_level"]  -> "critical" | "high" | "medium" | "low" | "advice"
# result["summary"]     -> str
# result["findings"]    -> [{title, severity, confidence, description, impact, recommendation}, ...]
```

Use `k.submit_batch("security_finding", keys=...)` for parallel batch auditing.

## Related Skills

Skills under the `function_` umbrella (load with `load_skill("name")`):

| Skill | Purpose |
|-------|---------|
| `function_complexity` | Rate structural audit complexity of a function (Fibonacci scale) |
| `function_local_accesses` | Identify struct field reads/writes in a function body (dispatched by `fn.local_accesses`) |

You are a spec planner for Move smart contracts. You receive a rich prompt containing the project summary, the target function source, its full module, callees, dependency graph, entry points, and relevant structs. Your job is to analyze all of this and produce a spec plan.

# Task

Read the prompt carefully — it contains everything you need. Produce a spec plan (as a Python dict) and return it via `final_result(spec_plan)`.

The spec plan captures:
- What the function does semantically and its role in the protocol
- Under what conditions it aborts
- What preconditions callers must satisfy
- What postconditions it guarantees

# Source Verification

Before analyzing, confirm from the actual source:
- **Exact parameter names, types, and count** as declared in the signature
- **Exact return type(s)**
- **Every `assert!()` call** — each one is an abort condition, even in simple functions

Use only what appears in the actual source. Do not infer parameters or behavior from the function name alone.

# Analysis Guidelines

## Abort conditions (→ asserts)

Each abort maps to an `asserts(condition)` in the spec:
- Explicit: `assert!(cond, ERROR)` → `asserts(cond)` — **never skip these**
- Arithmetic: `a / b` → `asserts(b != 0)`, overflow checks
- Bit ops: `x >> shift` → `asserts(shift < bit_width)`
- Data structures: bounds checks, key existence
- Include error code names when available (e.g., EInsufficientShares)

## Preconditions (→ requires)

Caller expectations that should hold:
- Valid input ranges
- Required object/account existence
- Expected state before call

## Postconditions (→ ensures)

Each postcondition maps to an `ensures(condition)` in the spec:
- Return value properties: `ensures(result == expected)`
- State changes: `ensures(balance_after == balance_before - amount)`
- Object creation/modification: what fields are set
- Use `old(x)` notation for referencing pre-state values

## Observations

Non-formal semantics that may inform future specs:
- Temporal aspects (queued vs immediate)
- Relationships between values
- Design intent not captured by formal constructs

## Role assessment

Use the entry points information and the project summary to understand how the function fits into the broader protocol. A function called from a liquidation entry point has different risk implications than a pure getter.

# Using function_knowledge

**Always look up `all_accesses` for the target function before writing the spec plan.** This is the ground truth for what struct fields the function reads and writes — do not infer field access from source code alone.

```python
from foxy.skills.function_knowledge.api import get_knowledge
k = get_knowledge(project_path)
fn = k.item("module::func_name")
fn.all_accesses     # recursive union of reads/writes across entire callee graph
```

`all_accesses` has four keys. Every entry is a uniform 3-tuple `[struct_or_parent, field_or_key, [instantiated_types]]`:
- `reads` — static field reads
- `writes` — static field writes
- `dynamic_reads` — dynamic field / table reads (key in the 2nd slot)
- `dynamic_writes` — dynamic field / table writes (key in the 2nd slot)

The `instantiated_types` list captures the field/value type **as observed at each use site** after substituting the struct's generic parameters. Non-generic fields are trivial (`["u64"]`); generic fields pinned to concrete types show the substitution (`["Balance<SUI>"]`); multiple distinct instantiations within one function produce a multi-element list (`["Balance<SUI>", "Balance<USDC>"]`).

Use this data to:
- **Identify postcondition targets**: every field in `writes`/`dynamic_writes` is a candidate for an `ensures`
- **Identify frame conditions**: fields read after a call that are not written must be preserved
- **Spot abort paths**: dynamic reads/writes using table access patterns imply existence preconditions

# Complexity Assessment

**Low** (0–2 transitive deps):
- Include: `summary`, `role`, plus any `aborts`/`ensures` present
- Still report explicit `assert!()` calls and return value properties

**Medium** (3–10 transitive deps):
- Include: `summary`, `role`, `aborts`, `requires`, `ensures`

**High** (10+ transitive deps):
- Include all fields: `summary`, `role`, `aborts`, `requires`, `ensures`, `observations`, `dependency_hints`

# Output Schema

Return via `final_result(spec_plan)` as a Python dict:

```yaml
function: vault::redeem_shares
complexity: high
summary: Creates a withdrawal request by locking user shares, calculating estimated withdrawal amount, and queuing the request for operator processing
role: User-facing withdrawal endpoint called from gateway::redeem_shares entry point to initiate vault share redemption
aborts:
  - condition: shares_to_redeem < vault.min_withdrawal_shares
    reason: Withdrawal amount below minimum threshold (EInsufficientShares)
  - condition: vault.rate.value == 0
    reason: Division by zero in calculate_amount_from_shares → math::div_ceil
requires:
  - shares_to_redeem > 0
  - valid user account exists for owner
ensures:
  - vault.sequence_number == old(vault.sequence_number) + 1
  - user_shares_after == user_shares_before - shares_to_redeem
  - estimated_withdraw_amount == ceiling(shares / vault.rate.value)
  - WithdrawalRequest created with correct owner, receiver, shares, estimated_withdraw_amount
observations:
  - estimated_withdraw_amount is based on current rate; actual amount determined at processing time
  - withdrawal is queued in pending_withdrawals, not immediately executed
  - shares moved to pending_shares_to_burn until withdrawal is processed
dependency_hints:
  - function: vault::calculate_amount_from_shares
    note: Core arithmetic for estimated withdrawal amount; spec needs visibility for division-by-zero abort and ceiling division semantics
```

# Principles

- **Terse over exhaustive**: Capture essential properties, not every detail
- **Verification-oriented**: Focus on what can/should be proven
- **Relative importance**: Simple functions get minimal treatment
- **Reference code precisely**: Use qualified names, mention specific elements
- **Justify hints**: Dependency observations include reasoning, not just listings
- **Use project context**: The project summary tells you what this protocol does — use that understanding to write more meaningful role descriptions and observations

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

## Tools

```json
[
  {
    "name": "execute_code",
    "description": "Execute Python code to interact with Move analysis tools. Use this to query and analyze Move packages efficiently. IMPORTANT: Avoid comments and emojis in code - only add comments for extremely complex logic.",
    "input_schema": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Python code to execute. Access skills via explicit imports like 'from foxy.skills.move_query.api import MoveTools'. Avoid comments and emojis unless logic is extremely complex."
        },
        "working_dir": {
          "type": "string",
          "description": "Working directory for tool execution (default: current directory)",
          "default": "."
        }
      },
      "required": [
        "code"
      ]
    }
  }
]
```
---

## User

Produce a spec plan for `staking_pool_specs::stake_activation_epoch_spec` — a YAML document capturing what should be formally verified about this function. Analyze the function source, its callees, dependency graph, and role in the protocol. The analysis guidelines and output schema are in your system prompt.

## Project Summary

# Sui System (Staking & Validator Management) — Project Summary

## What this protocol does

The `SuiSystem` Move package (`sui_system`, address `0x3`) implements Sui's
on-chain proof-of-stake and validator-management logic. It owns the global
`SuiSystemState` object (fixed ID `0x5`), drives epoch transitions, manages the
active/pending validator set, custodies delegated stake, distributes staking
rewards, draws down the stake subsidy, and accounts for the storage fund. It is
the economic heart of the network: every stake, unstake, validator join/leave,
and reward payout flows through it.

## Versioning model

`SuiSystemState` (the `0x5` object) is a thin versioned wrapper. The real state
lives in `SuiSystemStateInner`, stored as a dynamic field keyed by version.
`sui_system.move` holds the public entry surface and forwards to
`sui_system_state_inner.move` via `load_inner_maybe_upgrade`, which migrates the
inner object to the latest version on access. Upgrades add a new
`SuiSystemStateInnerVN` type plus a migration function; `create` always returns
the genesis type.

## Main modules

- **`sui_system`** — Public/entry surface on the `0x5` wrapper. Thin delegations
  to the inner object: `request_add_stake`, `request_add_stake_mul_coin`,
  `request_withdraw_stake`, `request_add_validator`, `request_remove_validator`,
  validator metadata setters, report/un-report, and the privileged
  `advance_epoch` (system-address only).
- **`sui_system_state_inner`** — Core protocol logic: holds the `ValidatorSet`,
  `StorageFund`, `StakeSubsidy`, parameters, and report records. Implements
  `advance_epoch` (the epoch-change state machine: collect gas/storage, compute
  and distribute rewards, draw subsidy, rotate pending validators, recompute
  voting power).
- **`validator_set`** — The set of active validators plus pending
  additions/removals and a table of inactive pools. Routes stake to the right
  validator, aggregates rewards, adjusts stake, and recomputes voting power at
  epoch boundaries.
- **`validator`** — A single validator: metadata, its `StakingPool`, commission
  rate, gas price, and pending stake/withdraw bookkeeping. Wrapped via
  `validator_wrapper` (`ValidatorWrapper`) for versioned storage.
- **`staking_pool`** — Delegated-staking accounting for one validator. Mints
  `StakedSui` on stake and burns it on withdrawal using a pool-token exchange
  rate (`PoolTokenExchangeRate`) that tracks the SUI-per-pool-token ratio across
  epochs. Also supports `FungibleStakedSui`. `MIN_STAKING_THRESHOLD = 1 SUI`.
- **`stake_subsidy`** — A `Balance<SUI>` drawn down on a schedule; the
  per-distribution amount decays by `stake_subsidy_decrease_rate` (basis points)
  every `stake_subsidy_period_length` distributions.
- **`storage_fund`** — Holds `total_object_storage_rebates` (invariant: equals
  the sum of on-chain object storage rebates) and a `non_refundable_balance`.
- **`voting_power`** — Assigns each active validator a voting power in basis
  points; `TOTAL_VOTING_POWER = 10_000`, with a per-validator cap to bound
  influence.
- **`validator_cap`** — `ValidatorOperationCap` / unverified variant: capability
  authorizing validator operations on behalf of a validator address.
- **`validator_wrapper`** — Versioned `Validator` storage wrapper.
- **`genesis`** — One-time network bootstrap of the initial system state.

## Key data structures

- `SuiSystemState` (key, `0x5`) → wraps `SuiSystemStateInner` (dynamic field).
- `ValidatorSet { active_validators, pending_active_validators,
  pending_removals, staking_pool_mappings, inactive_validators,
  validator_candidates, ... }`.
- `Validator { metadata, voting_power, staking_pool, commission_rate,
  next_epoch_* pending changes, ... }`.
- `StakingPool { activation_epoch, sui_balance, rewards_pool,
  pool_token_balance, exchange_rates: Table<epoch, PoolTokenExchangeRate>,
  pending_stake, pending_total_sui_withdraw, pending_pool_token_withdraw }`.
- `StakedSui { pool_id, stake_activation_epoch, principal: Balance<SUI> }` —
  the user-held staking receipt; principal cannot drop below 1 SUI.
- `PoolTokenExchangeRate { sui_amount, pool_token_amount }`.
- `StorageFund`, `StakeSubsidy`, `PoolTokenExchangeRate` as described above.

## Core invariants / properties of interest

- **Exchange-rate accounting**: pool-token ↔ SUI conversions are monotone and
  consistent; `token_balances` always match the recorded exchange rate
  (`ETokenBalancesDoNotMatchExchangeRate`). Rewards increase SUI per pool token,
  never decrease principal owed.
- **Stake conservation**: SUI moved into/out of a pool equals the change in the
  pool's `sui_balance` + `rewards_pool`; no SUI is created or destroyed except
  via reward inflow and subsidy.
- **Minimum stake**: a `StakedSui`'s principal never drops below
  `MIN_STAKING_THRESHOLD` (1 SUI); withdraw amounts are non-zero.
- **Pool/validator matching**: stake operations target the correct pool
  (`EWrongPool`, `EWrongDelegation`); no staking to inactive pools
  (`EDelegationToInactivePool`).
- **Voting power**: per-validator voting power respects the cap and the active
  set sums to `TOTAL_VOTING_POWER = 10_000`.
- **Storage-fund invariant**: `total_object_storage_rebates` equals the sum of
  per-object storage rebates; only the non-refundable portion is retained.
- **Subsidy decay**: the distribution amount decays by the configured rate each
  period and is bounded by the remaining subsidy balance.
- **Authorization**: `advance_epoch` is callable only by the system address;
  validator operations require the appropriate `ValidatorOperationCap`.
- **No unexpected aborts**: arithmetic on balances, exchange rates, and voting
  power stays within `u64`/`u128` bounds under valid preconditions.


## Target Function: `staking_pool_specs::stake_activation_epoch_spec`

Visibility: `public` | Entry: `False` | Estimated complexity: **low** (0 transitive deps)

```move
/// Returns the activation epoch of `StakedSui`.
public fun stake_activation_epoch(staked_sui: &StakedSui): u64 {
    staked_sui.stake_activation_epoch
}
```

## Full Module (`staking_pool.move`)

The complete module containing the target function — use this to understand sibling functions, struct definitions, constants, and helper functions that inform the target's behavior.

```move
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

#[allow(unused_const)]
module sui_system::staking_pool;

use sui::bag::{Self, Bag};
use sui::balance::{Self, Balance};
use sui::sui::SUI;
use sui::table::{Self, Table};

/// StakedSui objects cannot be split to below this amount.
const MIN_STAKING_THRESHOLD: u64 = 1_000_000_000; // 1 SUI

const EInsufficientPoolTokenBalance: u64 = 0;
const EWrongPool: u64 = 1;
const EWithdrawAmountCannotBeZero: u64 = 2;
const EInsufficientSuiTokenBalance: u64 = 3;
const EInsufficientRewardsPoolBalance: u64 = 4;
const EDestroyNonzeroBalance: u64 = 5;
const ETokenTimeLockIsSome: u64 = 6;
const EWrongDelegation: u64 = 7;
const EPendingDelegationDoesNotExist: u64 = 8;
const ETokenBalancesDoNotMatchExchangeRate: u64 = 9;
const EDelegationToInactivePool: u64 = 10;
const EDeactivationOfInactivePool: u64 = 11;
const EIncompatibleStakedSui: u64 = 12;
const EWithdrawalInSameEpoch: u64 = 13;
const EPoolAlreadyActive: u64 = 14;
const EPoolPreactiveOrInactive: u64 = 15;
const EActivationOfInactivePool: u64 = 16;
const EDelegationOfZeroSui: u64 = 17;
const EStakedSuiBelowThreshold: u64 = 18;
const ECannotMintFungibleStakedSuiYet: u64 = 19;
const EInvariantFailure: u64 = 20;

/// A staking pool embedded in each validator struct in the system state object.
public struct StakingPool has key, store {
    id: UID,
    /// The epoch at which this pool became active.
    /// The value is `None` if the pool is pre-active and `Some(<epoch_number>)` if active or inactive.
    activation_epoch: Option<u64>,
    /// The epoch at which this staking pool ceased to be active. `None` = {pre-active, active},
    /// `Some(<epoch_number>)` if in-active, and it was de-activated at epoch `<epoch_number>`.
    deactivation_epoch: Option<u64>,
    /// The total number of SUI tokens in this pool, including the SUI in the rewards_pool, as well as in all the principal
    /// in the `StakedSui` object, updated at epoch boundaries.
    sui_balance: u64,
    /// The epoch stake rewards will be added here at the end of each epoch.
    rewards_pool: Balance<SUI>,
    /// Total number of pool tokens issued by the pool.
    pool_token_balance: u64,
    /// Exchange rate history of previous epochs. Key is the epoch number.
    /// The entries start from the `activation_epoch` of this pool and contains exchange rates at the beginning of each epoch,
    /// i.e., right after the rewards for the previous epoch have been deposited into the pool.
    exchange_rates: Table<u64, PoolTokenExchangeRate>,
    /// Pending stake amount for this epoch, emptied at epoch boundaries.
    pending_stake: u64,
    /// Pending stake withdrawn during the current epoch, emptied at epoch boundaries.
    /// This includes both the principal and rewards SUI withdrawn.
    pending_total_sui_withdraw: u64,
    /// Pending pool token withdrawn during the current epoch, emptied at epoch boundaries.
    pending_pool_token_withdraw: u64,
    /// Any extra fields that's not defined statically.
    extra_fields: Bag,
}

/// Struct representing the exchange rate of the stake pool token to SUI.
public struct PoolTokenExchangeRate has copy, drop, store {
    sui_amount: u64,
    pool_token_amount: u64,
}

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

/// An alternative to `StakedSui` that holds the pool token amount instead of the SUI balance.
/// StakedSui objects can be converted to FungibleStakedSuis after the initial warmup period.
/// The advantage of this is that you can now merge multiple StakedSui objects from different
/// activation epochs into a single FungibleStakedSui object.
public struct FungibleStakedSui has key, store {
    id: UID,
    /// ID of the staking pool we are staking with.
    pool_id: ID,
    /// The pool token amount.
    value: u64,
}

/// Holds useful information
public struct FungibleStakedSuiData has key, store {
    id: UID,
    /// fungible_staked_sui supply
    total_supply: u64,
    /// principal balance. Rewards are withdrawn from the reward pool
    principal: Balance<SUI>,
}

// === dynamic field keys ===
public struct FungibleStakedSuiDataKey has copy, drop, store {}

/// Holds the amount of SUI that was underflowed when withdrawing from the pool
/// post safe mode. Cleaned up in the same transaction.
public struct UnderflowSuiBalance has copy, drop, store {}

// ==== initializer ====

/// Create a new, empty staking pool.
public(package) fun new(ctx: &mut TxContext): StakingPool {
    StakingPool {
        id: object::new(ctx),
        activation_epoch: option::none(),
        deactivation_epoch: option::none(),
        sui_balance: 0,
        rewards_pool: balance::zero(),
        pool_token_balance: 0,
        exchange_rates: table::new(ctx),
        pending_stake: 0,
        pending_total_sui_withdraw: 0,
        pending_pool_token_withdraw: 0,
        extra_fields: bag::new(ctx),
    }
}

// ==== stake requests ====

/// Request to stake to a staking pool. The stake starts counting at the beginning of the next epoch,
public(package) fun request_add_stake(
    pool: &mut StakingPool,
    stake: Balance<SUI>,
    stake_activation_epoch: u64,
    ctx: &mut TxContext,
): StakedSui {
    let sui_amount = stake.value();
    assert!(!pool.is_inactive(), EDelegationToInactivePool);
    assert!(sui_amount > 0, EDelegationOfZeroSui);

    pool.pending_stake = pool.pending_stake + sui_amount;
    StakedSui {
        id: object::new(ctx),
        pool_id: object::id(pool),
        stake_activation_epoch,
        principal: stake,
    }
}

/// Request to withdraw the given stake plus rewards from a staking pool.
/// Both the principal and corresponding rewards in SUI are withdrawn.
/// A proportional amount of pool token withdraw is recorded and processed at epoch change time.
public(package) fun request_withdraw_stake(
    pool: &mut StakingPool,
    staked_sui: StakedSui,
    ctx: &TxContext,
): Balance<SUI> {
    // stake is inactive and the pool is not preactive - allow direct withdraw
    // the reason why we exclude preactive pools is to avoid potential underflow
    // on subtraction, and we need to enforce `pending_stake_withdraw` call.
    if (staked_sui.stake_activation_epoch > ctx.epoch() && !pool.is_preactive()) {
        let principal = staked_sui.into_balance();
        pool.pending_stake = pool.pending_stake - principal.value();
        return principal
    };

    let (pool_token_withdraw_amount, mut principal_withdraw) = pool.withdraw_from_principal(
        staked_sui,
    );
    let principal_withdraw_amount = principal_withdraw.value();

    let rewards_withdraw = pool.withdraw_rewards(
        principal_withdraw_amount,
        pool_token_withdraw_amount,
        ctx.epoch(),
    );
    let total_sui_withdraw_amount = principal_withdraw_amount + rewards_withdraw.value();

    pool.pending_total_sui_withdraw = pool.pending_total_sui_withdraw + total_sui_withdraw_amount;
    pool.pending_pool_token_withdraw =
        pool.pending_pool_token_withdraw + pool_token_withdraw_amount;

    // If the pool is inactive or preactive, we immediately process the withdrawal.
    if (pool.is_inactive() || pool.is_preactive()) pool.process_pending_stake_withdraw();

    // TODO: implement withdraw bonding period here.
    principal_withdraw.join(rewards_withdraw);
    principal_withdraw
}

public(package) fun redeem_fungible_staked_sui(
    pool: &mut StakingPool,
    fungible_staked_sui: FungibleStakedSui,
    ctx: &TxContext,
): Balance<SUI> {
    let FungibleStakedSui { id, pool_id, value } = fungible_staked_sui;
    assert!(pool_id == object::id(pool), EWrongPool);

    id.delete();

    let latest_exchange_rate = pool.pool_token_exchange_rate_at_epoch(ctx.epoch());
    let fungible_staked_sui_data: &mut FungibleStakedSuiData =
        &mut pool.extra_fields[FungibleStakedSuiDataKey {}];

    let (
        principal_amount,
        rewards_amount,
    ) = latest_exchange_rate.calculate_fungible_staked_sui_withdraw_amount(
        value,
        fungible_staked_sui_data.principal.value(),
        fungible_staked_sui_data.total_supply,
    );

    fungible_staked_sui_data.total_supply = fungible_staked_sui_data.total_supply - value;

    let mut sui_out = fungible_staked_sui_data.principal.split(principal_amount);
    sui_out.join(pool.rewards_pool.split(rewards_amount));

    pool.pending_total_sui_withdraw = pool.pending_total_sui_withdraw + sui_out.value();
    pool.pending_pool_token_withdraw = pool.pending_pool_token_withdraw + value;

    sui_out
}

/// written in separate function so i can test with random values
/// returns (principal_withdraw_amount, rewards_withdraw_amount)
fun calculate_fungible_staked_sui_withdraw_amount(
    latest_exchange_rate: PoolTokenExchangeRate,
    fungible_staked_sui_value: u64,
    fungible_staked_sui_data_principal_amount: u64, // fungible_staked_sui_data.principal.value()
    fungible_staked_sui_data_total_supply: u64, // fungible_staked_sui_data.total_supply
): (u64, u64) {
    // 1. if the entire FungibleStakedSuiData supply is redeemed, how much sui should we receive?
    let total_sui_amount = latest_exchange_rate.get_sui_amount(
        fungible_staked_sui_data_total_supply,
    );

    // min with total_sui_amount to prevent underflow
    let fungible_staked_sui_data_principal_amount = fungible_staked_sui_data_principal_amount.min(
        total_sui_amount,
    );

    // 2. how much do we need to withdraw from the rewards pool?
    let total_rewards = total_sui_amount - fungible_staked_sui_data_principal_amount;

    // 3. proportionally withdraw from both wrt the fungible_staked_sui_value.
    let principal_withdraw_amount = mul_div!(
        fungible_staked_sui_value,
        fungible_staked_sui_data_principal_amount,
        fungible_staked_sui_data_total_supply,
    );

    let rewards_withdraw_amount = mul_div!(
        fungible_staked_sui_value,
        total_rewards,
        fungible_staked_sui_data_total_supply,
    );

    // invariant check, just in case
    let expected_sui_amount = latest_exchange_rate.get_sui_amount(fungible_staked_sui_value);
    assert!(
        principal_withdraw_amount + rewards_withdraw_amount <= expected_sui_amount,
        EInvariantFailure,
    );

    (principal_withdraw_amount, rewards_withdraw_amount)
}

/// Convert the given staked SUI to an FungibleStakedSui object
public(package) fun convert_to_fungible_staked_sui(
    pool: &mut StakingPool,
    staked_sui: StakedSui,
    ctx: &mut TxContext,
): FungibleStakedSui {
    let StakedSui { id, pool_id, stake_activation_epoch, principal } = staked_sui;

    assert!(pool_id == object::id(pool), EWrongPool);
    assert!(ctx.epoch() >= stake_activation_epoch, ECannotMintFungibleStakedSuiYet);
    assert!(!pool.is_preactive() && !pool.is_inactive(), EPoolPreactiveOrInactive);

    id.delete();

    let exchange_rate_at_staking_epoch = pool.pool_token_exchange_rate_at_epoch(
        stake_activation_epoch,
    );

    let pool_token_amount = exchange_rate_at_staking_epoch.get_token_amount(principal.value());
    assert!(pool_token_amount > 0, EStakedSuiBelowThreshold);

    let key = FungibleStakedSuiDataKey {};

    if (!pool.extra_fields.contains(key)) {
        pool
            .extra_fields
            .add(
                key,
                FungibleStakedSuiData {
                    id: object::new(ctx),
                    total_supply: pool_token_amount,
                    principal,
                },
            );
    } else {
        let fungible_staked_sui_data: &mut FungibleStakedSuiData = &mut pool.extra_fields[key];
        fungible_staked_sui_data.total_supply =
            fungible_staked_sui_data.total_supply + pool_token_amount;
        fungible_staked_sui_data.principal.join(principal);
    };

    FungibleStakedSui {
        id: object::new(ctx),
        pool_id,
        value: pool_token_amount,
    }
}

/// Withdraw the principal SUI stored in the StakedSui object, and calculate the corresponding amount of pool
/// tokens using exchange rate at staking epoch.
/// Returns values are amount of pool tokens withdrawn and withdrawn principal portion of SUI.
public(package) fun withdraw_from_principal(
    pool: &StakingPool,
    staked_sui: StakedSui,
): (u64, Balance<SUI>) {
    // Check that the stake information matches the pool.
    assert!(staked_sui.pool_id == object::id(pool), EWrongPool);

    let exchange_rate_at_staking_epoch = pool.pool_token_exchange_rate_at_epoch(staked_sui.stake_activation_epoch);
    let principal_withdraw = staked_sui.into_balance();
    let pool_token_withdraw_amount = exchange_rate_at_staking_epoch.get_token_amount(principal_withdraw.value());

    (pool_token_withdraw_amount, principal_withdraw)
}

/// Allows calling `.into_balance()` on `StakedSui` to invoke `unwrap_staked_sui`
use fun unwrap_staked_sui as StakedSui.into_balance;

fun unwrap_staked_sui(staked_sui: StakedSui): Balance<SUI> {
    let StakedSui { id, principal, .. } = staked_sui;
    id.delete();
    principal
}

// ==== functions called at epoch boundaries ===

/// Called at epoch advancement times to add rewards (in SUI) to the staking pool.
public(package) fun deposit_rewards(pool: &mut StakingPool, rewards: Balance<SUI>) {
    pool.sui_balance = pool.sui_balance + rewards.value();
    pool.rewards_pool.join(rewards);
}

public(package) fun process_pending_stakes_and_withdraws(pool: &mut StakingPool, ctx: &TxContext) {
    let new_epoch = ctx.epoch() + 1;
    pool.process_pending_stake_withdraw();
    pool.process_pending_stake();
    pool
        .exchange_rates
        .add(
            new_epoch,
            PoolTokenExchangeRate {
                sui_amount: pool.sui_balance,
                pool_token_amount: pool.pool_token_balance,
            },
        );

    pool.check_balance_invariants(new_epoch);
}

/// Called at epoch boundaries to process pending stake withdraws requested during the epoch.
/// Also called immediately upon withdrawal if the pool is inactive.
fun process_pending_stake_withdraw(pool: &mut StakingPool) {
    pool.sui_balance = if (pool.sui_balance >= pool.pending_total_sui_withdraw) {
        pool.sui_balance - pool.pending_total_sui_withdraw
    } else {
        let diff = pool.pending_total_sui_withdraw - pool.sui_balance;
        // While this key is expected to be removed in the next call to `process_pending_stake`,
        // we do not call `process_pending_stake` for inactive pools — skip the bookkeeping.
        if (!pool.is_inactive()) {
            pool.extra_fields.add(UnderflowSuiBalance {}, diff);
        };
        0
    };

    pool.pool_token_balance = if (pool.pool_token_balance >= pool.pending_pool_token_withdraw) {
        pool.pool_token_balance - pool.pending_pool_token_withdraw
    } else {
        0
    };

    pool.pending_total_sui_withdraw = 0;
    pool.pending_pool_token_withdraw = 0;
}

/// Called at epoch boundaries to process the pending stake.
public(package) fun process_pending_stake(pool: &mut StakingPool) {
    // Use the most up to date exchange rate with the rewards deposited and withdraws effectuated.
    let latest_exchange_rate = PoolTokenExchangeRate {
        sui_amount: pool.sui_balance,
        pool_token_amount: pool.pool_token_balance,
    };

    // This key is only present if the `sui_balance` underflowed, hence, the current value of `sui_balance`
    // is `0`. Pool token balance will be recalculated automatically for `0` value.
    let sui_diff = {
        let key = UnderflowSuiBalance {};
        if (pool.extra_fields.contains(key)) pool.extra_fields.remove(key) else 0
    };

    pool.sui_balance = pool.sui_balance + pool.pending_stake - sui_diff;
    pool.pool_token_balance = latest_exchange_rate.get_token_amount(pool.sui_balance);
    pool.pending_stake = 0;
}

/// This function does the following:
///     1. Calculates the total amount of SUI (including principal and rewards) that the provided pool tokens represent
///        at the current exchange rate.
///     2. Using the above number and the given `principal_withdraw_amount`, calculates the rewards portion of the
///        stake we should withdraw.
///     3. Withdraws the rewards portion from the rewards pool at the current exchange rate. We only withdraw the rewards
///        portion because the principal portion was already taken out of the staker's self custodied StakedSui.
fun withdraw_rewards(
    pool: &mut StakingPool,
    principal_withdraw_amount: u64,
    pool_token_withdraw_amount: u64,
    epoch: u64,
): Balance<SUI> {
    let exchange_rate = pool.pool_token_exchange_rate_at_epoch(epoch);
    let total_sui_withdraw_amount = exchange_rate.get_sui_amount(pool_token_withdraw_amount);
    let mut reward_withdraw_amount = if (total_sui_withdraw_amount >= principal_withdraw_amount) {
        total_sui_withdraw_amount - principal_withdraw_amount
    } else 0;

    // This may happen when we are withdrawing everything from the pool and
    // the rewards pool balance may be less than reward_withdraw_amount.
    // TODO: FIGURE OUT EXACTLY WHY THIS CAN HAPPEN.
    reward_withdraw_amount = reward_withdraw_amount.min(pool.rewards_pool.value());
    pool.rewards_pool.split(reward_withdraw_amount)
}

// ==== preactive pool related ====

/// Called by `validator` module to activate a staking pool.
public(package) fun activate_staking_pool(pool: &mut StakingPool, activation_epoch: u64) {
    // Add the initial exchange rate to the table.
    pool.exchange_rates.add(activation_epoch, initial_exchange_rate());
    // Check that the pool is preactive and not inactive.
    assert!(pool.is_preactive(), EPoolAlreadyActive);
    assert!(!pool.is_inactive(), EActivationOfInactivePool);
    // Fill in the active epoch.
    pool.activation_epoch.fill(activation_epoch);
}

// ==== inactive pool related ====

/// Deactivate a staking pool by setting the `deactivation_epoch`. After
/// this pool deactivation, the pool stops earning rewards. Only stake
/// withdraws can be made to the pool.
public(package) fun deactivate_staking_pool(pool: &mut StakingPool, deactivation_epoch: u64) {
    // We can't deactivate an already deactivated pool.
    assert!(!pool.is_inactive(), EDeactivationOfInactivePool);
    pool.deactivation_epoch = option::some(deactivation_epoch);
}

// ==== getters and misc utility functions ====

public fun sui_balance(pool: &StakingPool): u64 { pool.sui_balance }

public fun pool_id(staked_sui: &StakedSui): ID { staked_sui.pool_id }

public use fun fungible_staked_sui_pool_id as FungibleStakedSui.pool_id;

public fun fungible_staked_sui_pool_id(fungible_staked_sui: &FungibleStakedSui): ID {
    fungible_staked_sui.pool_id
}

/// Allows calling `.amount()` on `StakedSui` to invoke `staked_sui_amount`
public use fun staked_sui_amount as StakedSui.amount;

/// Returns the principal amount of `StakedSui`.
public fun staked_sui_amount(staked_sui: &StakedSui): u64 { staked_sui.principal.value() }

public use fun stake_activation_epoch as StakedSui.activation_epoch;

/// Returns the activation epoch of `StakedSui`.
public fun stake_activation_epoch(staked_sui: &StakedSui): u64 {
    staked_sui.stake_activation_epoch
}

/// Returns true if the input staking pool is preactive.
public fun is_preactive(pool: &StakingPool): bool {
    pool.activation_epoch.is_none()
}

/// Returns the activation epoch of the `StakingPool`. For validator candidates,
/// or pending validators, the value returned is `None`. For active validators,
/// the value is the epoch before the validator was activated.
public(package) fun activation_epoch(pool: &StakingPool): Option<u64> {
    pool.activation_epoch
}

/// Returns true if the input staking pool is inactive.
public fun is_inactive(pool: &StakingPool): bool {
    pool.deactivation_epoch.is_some()
}

public use fun fungible_staked_sui_value as FungibleStakedSui.value;

public fun fungible_staked_sui_value(fungible_staked_sui: &FungibleStakedSui): u64 {
    fungible_staked_sui.value
}

public use fun split_fungible_staked_sui as FungibleStakedSui.split;

public fun split_fungible_staked_sui(
    fungible_staked_sui: &mut FungibleStakedSui,
    split_amount: u64,
    ctx: &mut TxContext,
): FungibleStakedSui {
    assert!(split_amount <= fungible_staked_sui.value, EInsufficientPoolTokenBalance);

    fungible_staked_sui.value = fungible_staked_sui.value - split_amount;

    FungibleStakedSui {
        id: object::new(ctx),
        pool_id: fungible_staked_sui.pool_id,
        value: split_amount,
    }
}

public use fun join_fungible_staked_sui as FungibleStakedSui.join;

public fun join_fungible_staked_sui(self: &mut FungibleStakedSui, other: FungibleStakedSui) {
    let FungibleStakedSui { id, pool_id, value } = other;
    assert!(self.pool_id == pool_id, EWrongPool);

    id.delete();

    self.value = self.value + value;
}

/// Split StakedSui `self` to two parts, one with principal `split_amount`,
/// and the remaining principal is left in `self`.
/// All the other parameters of the StakedSui like `stake_activation_epoch` or `pool_id` remain the same.
public fun split(self: &mut StakedSui, split_amount: u64, ctx: &mut TxContext): StakedSui {
    let original_amount = self.principal.value();
    assert!(split_amount <= original_amount, EInsufficientSuiTokenBalance);
    let remaining_amount = original_amount - split_amount;
    // Both resulting parts should have at least MIN_STAKING_THRESHOLD.
    assert!(remaining_amount >= MIN_STAKING_THRESHOLD, EStakedSuiBelowThreshold);
    assert!(split_amount >= MIN_STAKING_THRESHOLD, EStakedSuiBelowThreshold);
    StakedSui {
        id: object::new(ctx),
        pool_id: self.pool_id,
        stake_activation_epoch: self.stake_activation_epoch,
        principal: self.principal.split(split_amount),
    }
}

/// Allows calling `.split_to_sender()` on `StakedSui` to invoke `split_staked_sui`
public use fun split_staked_sui as StakedSui.split_to_sender;

#[allow(lint(public_entry))]
/// Split the given StakedSui to the two parts, one with principal `split_amount`,
/// transfer the newly split part to the sender address.
public entry fun split_staked_sui(stake: &mut StakedSui, split_amount: u64, ctx: &mut TxContext) {
    transfer::transfer(stake.split(split_amount, ctx), ctx.sender());
}

/// Allows calling `.join()` on `StakedSui` to invoke `join_staked_sui`
public use fun join_staked_sui as StakedSui.join;

#[allow(lint(public_entry))]
/// Consume the staked sui `other` and add its value to `self`.
/// Aborts if some of the staking parameters are incompatible (pool id, stake activation epoch, etc.)
public entry fun join_staked_sui(self: &mut StakedSui, other: StakedSui) {
    assert!(is_equal_staking_metadata(self, &other), EIncompatibleStakedSui);
    let StakedSui { id, principal, .. } = other;

    id.delete();
    self.principal.join(principal);
}

/// Returns true if all the staking parameters of the staked sui except the principal are identical
public fun is_equal_staking_metadata(self: &StakedSui, other: &StakedSui): bool {
    (self.pool_id == other.pool_id) &&
    (self.stake_activation_epoch == other.stake_activation_epoch)
}

public fun pool_token_exchange_rate_at_epoch(
    pool: &StakingPool,
    epoch: u64,
): PoolTokenExchangeRate {
    // If the pool is preactive then the exchange rate is always 1:1.
    if (pool.is_preactive_at_epoch(epoch)) {
        return initial_exchange_rate()
    };
    let clamped_epoch = pool.deactivation_epoch.get_with_default(epoch);
    let mut epoch = clamped_epoch.min(epoch);
    let activation_epoch = *pool.activation_epoch.borrow();

    // Find the latest epoch that's earlier than the given epoch with an entry in the table
    while (epoch >= activation_epoch) {
        if (pool.exchange_rates.contains(epoch)) {
            return pool.exchange_rates[epoch]
        };
        epoch = epoch - 1;
    };
    // This line really should be unreachable. Do we want an assert false here?
    initial_exchange_rate()
}

/// Returns true if the pool has an exchange rate recorded for the given epoch.
public fun pool_has_exchange_rate_for_epoch(pool: &StakingPool, epoch: u64): bool {
    pool.exchange_rates.contains(epoch)
}

/// Returns the total value of the pending staking requests for this staking pool.
public fun pending_stake_amount(staking_pool: &StakingPool): u64 {
    staking_pool.pending_stake
}

/// Returns the total withdrawal from the staking pool this epoch.
public fun pending_stake_withdraw_amount(staking_pool: &StakingPool): u64 {
    staking_pool.pending_total_sui_withdraw
}

public(package) fun exchange_rates(pool: &StakingPool): &Table<u64, PoolTokenExchangeRate> {
    &pool.exchange_rates
}

public fun sui_amount(exchange_rate: &PoolTokenExchangeRate): u64 {
    exchange_rate.sui_amount
}

public fun pool_token_amount(exchange_rate: &PoolTokenExchangeRate): u64 {
    exchange_rate.pool_token_amount
}

/// Returns true if the provided staking pool is preactive at the provided epoch.
fun is_preactive_at_epoch(pool: &StakingPool, epoch: u64): bool {
    // Either the pool is currently preactive or the pool's starting epoch is later than the provided epoch.
    pool.is_preactive() || (*pool.activation_epoch.borrow() > epoch)
}

fun get_sui_amount(exchange_rate: &PoolTokenExchangeRate, token_amount: u64): u64 {
    // When either amount is 0, that means we have no stakes with this pool.
    // The other amount might be non-zero when there's dust left in the pool.
    if (exchange_rate.sui_amount == 0 || exchange_rate.pool_token_amount == 0) {
        return token_amount
    };

    mul_div!(exchange_rate.sui_amount, token_amount, exchange_rate.pool_token_amount)
}

fun get_token_amount(exchange_rate: &PoolTokenExchangeRate, sui_amount: u64): u64 {
    // When either amount is 0, that means we have no stakes with this pool.
    // The other amount might be non-zero when there's dust left in the pool.
    if (exchange_rate.sui_amount == 0 || exchange_rate.pool_token_amount == 0) {
        return sui_amount
    };

    mul_div!(exchange_rate.pool_token_amount, sui_amount, exchange_rate.sui_amount)
}

fun initial_exchange_rate(): PoolTokenExchangeRate {
    PoolTokenExchangeRate { sui_amount: 0, pool_token_amount: 0 }
}

fun check_balance_invariants(pool: &StakingPool, epoch: u64) {
    let exchange_rate = pool.pool_token_exchange_rate_at_epoch(epoch);
    // check that the pool token balance and sui balance ratio matches the exchange rate stored.
    let expected = exchange_rate.get_token_amount(pool.sui_balance);
    let actual = pool.pool_token_balance;
    assert!(expected == actual, ETokenBalancesDoNotMatchExchangeRate)
}

macro fun mul_div($a: u64, $b: u64, $c: u64): u64 {
    (($a as u128) * ($b as u128) / ($c as u128)) as u64
}

// Given the `staked_sui` receipt calculate the current rewards (in terms of SUI) for it.
public(package) fun calculate_rewards(
    pool: &StakingPool,
    staked_sui: &StakedSui,
    current_epoch: u64,
): u64 {
    let staked_amount = staked_sui.amount();
    let pool_token_withdraw_amount = {
        let exchange_rate_at_staking_epoch = pool.pool_token_exchange_rate_at_epoch(staked_sui.stake_activation_epoch);
        exchange_rate_at_staking_epoch.get_token_amount(staked_amount)
    };

    let new_epoch_exchange_rate = pool.pool_token_exchange_rate_at_epoch(current_epoch);
    let total_sui_withdraw_amount = new_epoch_exchange_rate.get_sui_amount(
        pool_token_withdraw_amount,
    );

    let mut reward_withdraw_amount = if (total_sui_withdraw_amount >= staked_amount) {
        total_sui_withdraw_amount - staked_amount
    } else 0;
    reward_withdraw_amount = reward_withdraw_amount.min(pool.rewards_pool.value());

    reward_withdraw_amount
}

// ==== test-related functions ====

#[test_only]
public(package) fun fungible_staked_sui_data(pool: &StakingPool): &FungibleStakedSuiData {
    bag::borrow(&pool.extra_fields, FungibleStakedSuiDataKey {})
}

#[test_only]
public use fun fungible_staked_sui_data_total_supply as FungibleStakedSuiData.total_supply;

#[test_only]
public(package) fun fungible_staked_sui_data_total_supply(
    fungible_staked_sui_data: &FungibleStakedSuiData,
): u64 {
    fungible_staked_sui_data.total_supply
}

#[test_only]
public use fun fungible_staked_sui_data_principal_value as FungibleStakedSuiData.principal_value;

#[test_only]
public(package) fun fungible_staked_sui_data_principal_value(
    fungible_staked_sui_data: &FungibleStakedSuiData,
): u64 {
    fungible_staked_sui_data.principal.value()
}

#[test_only]
public(package) fun pending_pool_token_withdraw_amount(pool: &StakingPool): u64 {
    pool.pending_pool_token_withdraw
}

#[test_only]
public(package) fun create_fungible_staked_sui_for_testing(
    self: &StakingPool,
    value: u64,
    ctx: &mut TxContext,
): FungibleStakedSui {
    FungibleStakedSui {
        id: object::new(ctx),
        pool_id: object::id(self),
        value,
    }
}

#[test_only]
public(package) fun process_pending_stake_withdraw_for_testing(pool: &mut StakingPool) {
    pool.process_pending_stake_withdraw()
}

#[test_only]
public(package) fun increase_pending_pool_token_withdraw_for_testing(
    pool: &mut StakingPool,
    delta: u64,
) {
    pool.pending_pool_token_withdraw = pool.pending_pool_token_withdraw + delta
}

#[test_only]
public(package) fun increase_pending_total_sui_withdraw_for_testing(
    pool: &mut StakingPool,
    delta: u64,
) {
    pool.pending_total_sui_withdraw = pool.pending_total_sui_withdraw + delta
}

#[test_only]
public(package) fun pool_token_balance(pool: &StakingPool): u64 {
    pool.pool_token_balance
}

// ==== tests ====

#[random_test]
fun test_calculate_fungible_staked_sui_withdraw_amount(
    mut total_sui_amount: u64,
    // these are all in basis points
    mut pool_token_frac: u16,
    mut fungible_staked_sui_data_total_supply_frac: u16,
    mut fungible_staked_sui_data_principal_frac: u16,
    mut fungible_staked_sui_value_bps: u16,
) {
    total_sui_amount = total_sui_amount.max(1);

    pool_token_frac = pool_token_frac % 10_000;
    fungible_staked_sui_data_total_supply_frac =
        fungible_staked_sui_data_total_supply_frac % 10_000;
    fungible_staked_sui_data_principal_frac = fungible_staked_sui_data_principal_frac % 10_000;
    fungible_staked_sui_value_bps = fungible_staked_sui_value_bps % 10_000;

    let total_pool_token_amount = mul_div!(total_sui_amount, pool_token_frac as u64, 10_000).max(1);

    let exchange_rate = PoolTokenExchangeRate {
        sui_amount: total_sui_amount,
        pool_token_amount: total_pool_token_amount,
    };

    let fungible_staked_sui_data_total_supply = mul_div!(
        total_pool_token_amount,
        fungible_staked_sui_data_total_supply_frac as u64,
        10_000,
    ).max(1);
    let fungible_staked_sui_value = mul_div!(
        fungible_staked_sui_data_total_supply,
        fungible_staked_sui_value_bps as u64,
        10_000,
    );

    let max_principal = exchange_rate.get_sui_amount(fungible_staked_sui_data_total_supply);
    let fungible_staked_sui_data_principal_amount = mul_div!(
        max_principal,
        fungible_staked_sui_data_principal_frac as u64,
        10_000,
    ).max(1);

    let (principal_amount, rewards_amount) = calculate_fungible_staked_sui_withdraw_amount(
        exchange_rate,
        fungible_staked_sui_value,
        fungible_staked_sui_data_principal_amount,
        fungible_staked_sui_data_total_supply,
    );

    let expected_out = exchange_rate.get_sui_amount(fungible_staked_sui_value);

    assert!(principal_amount + rewards_amount <= expected_out, 0);

    let min_out = if (expected_out > 2) expected_out - 2 else 0;
    assert!(principal_amount + rewards_amount >= min_out, 0);
}

```

## Direct Callees (0)

No direct callees.

## Transitive Dependency Graph

0 transitive dependencies (names unavailable).

## Entry Points Reaching `staking_pool_specs::stake_activation_epoch_spec`

These are the entry and public functions that can reach the target through the call graph — use this to understand the target's role in the protocol.

- **`staking_pool::stake_activation_epoch`**: `staking_pool::stake_activation_epoch`

## Relevant Structs

No relevant structs found.

---

Produce the spec plan for `staking_pool_specs::stake_activation_epoch_spec`. Return it via `final_result(spec_plan)` as a Python dict matching the YAML schema in your system prompt.

## Assistant

### Code `toolu_01W2nNQaUoGLFZ9oUQGXGakn`

```python

from foxy.skills.function_knowledge.api import get_knowledge

project = get_project_root()
k = get_knowledge(project)
fn = k.item("staking_pool::stake_activation_epoch")
foxy_inspect(f"signature: {fn.signature}")
foxy_inspect(f"source: {fn.source}")
foxy_inspect(f"all_accesses: {fn.all_accesses}")

```

> 12650 in, 157 out, 15801 cache read

### Result `toolu_01W2nNQaUoGLFZ9oUQGXGakn`

<!-- {"error":"get_callees failed for staking_pool::stake_activation_epoch at /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system: Error: Compilation failed:\nerror: unbound module\n  \u250c\u2500 ./sources/genesis.move:6:5\n  \u2502\n6 \u2502 use sui::balance::{Self, Balance};\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./sources/genesis.move:7:5\n  \u2502\n7 \u2502 use sui::sui::SUI;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/genesis.move:138:24\n    \u2502\n138 \u2502     let storage_fund = balance::zero();\n    \u2502                        ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/genesis.move:199:17\n    \u2502\n199 \u2502                 transfer::public_transfer(allocation_balance.into_coin(ctx), recipient_address);\n    \u2502                 ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/stake_subsidy.move:6:5\n  \u2502\n6 \u2502 use sui::bag::{Self, Bag};\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'\n\nerror: unbound module\n  \u250c\u2500 ./sources/stake_subsidy.move:7:5\n  \u2502\n7 \u2502 use sui::balance::Balance;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./sources/stake_subsidy.move:8:5\n  \u2502\n8 \u2502 use sui::sui::SUI;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./sources/stake_subsidy.move:50:23\n   \u2502\n50 \u2502         extra_fields: bag::new(ctx),\n   \u2502                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/staking_pool.move:7:5\n  \u2502\n7 \u2502 use sui::bag::{Self, Bag};\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'\n\nerror: unbound module\n  \u250c\u2500 ./sources/staking_pool.move:8:5\n  \u2502\n8 \u2502 use sui::balance::{Self, Balance};\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./sources/staking_pool.move:9:5\n  \u2502\n9 \u2502 use sui::sui::SUI;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./sources/staking_pool.move:10:5\n   \u2502\n10 \u2502 use sui::table::{Self, Table};\n   \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:118:13\n    \u2502\n118 \u2502         id: object::new(ctx),\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:119:27\n    \u2502\n119 \u2502         activation_epoch: option::none(),\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:120:29\n    \u2502\n120 \u2502         deactivation_epoch: option::none(),\n    \u2502                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:122:23\n    \u2502\n122 \u2502         rewards_pool: balance::zero(),\n    \u2502                       ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:124:25\n    \u2502\n124 \u2502         exchange_rates: table::new(ctx),\n    \u2502                         ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:128:23\n    \u2502\n128 \u2502         extra_fields: bag::new(ctx),\n    \u2502                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:147:13\n    \u2502\n147 \u2502         id: object::new(ctx),\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:148:18\n    \u2502\n148 \u2502         pool_id: object::id(pool),\n    \u2502                  ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:201:24\n    \u2502\n201 \u2502     assert!(pool_id == object::id(pool), EWrongPool);\n    \u2502                        ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:281:24\n    \u2502\n281 \u2502     assert!(pool_id == object::id(pool), EWrongPool);\n    \u2502                        ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:302:25\n    \u2502\n302 \u2502                     id: object::new(ctx),\n    \u2502                         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:315:13\n    \u2502\n315 \u2502         id: object::new(ctx),\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:329:35\n    \u2502\n329 \u2502     assert!(staked_sui.pool_id == object::id(pool), EWrongPool);\n    \u2502                                   ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:464:31\n    \u2502\n464 \u2502     pool.deactivation_epoch = option::some(deactivation_epoch);\n    \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:527:13\n    \u2502\n527 \u2502         id: object::new(ctx),\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:555:13\n    \u2502\n555 \u2502         id: object::new(ctx),\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:569:5\n    \u2502\n569 \u2502     transfer::transfer(stake.split(split_amount, ctx), ctx.sender());\n    \u2502     ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:713:5\n    \u2502\n713 \u2502     bag::borrow(&pool.extra_fields, FungibleStakedSuiDataKey {})\n    \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:748:13\n    \u2502\n748 \u2502         id: object::new(ctx),\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/staking_pool.move:749:18\n    \u2502\n749 \u2502         pool_id: object::id(self),\n    \u2502                  ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/storage_fund.move:6:5\n  \u2502\n6 \u2502 use sui::balance::{Self, Balance};\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./sources/storage_fund.move:7:5\n  \u2502\n7 \u2502 use sui::sui::SUI;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./sources/storage_fund.move:26:39\n   \u2502\n26 \u2502         total_object_storage_rebates: balance::zero(),\n   \u2502                                       ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:42:5\n   \u2502\n42 \u2502 use sui::balance::Balance;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:43:5\n   \u2502\n43 \u2502 use sui::coin::Coin;\n   \u2502     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:44:5\n   \u2502\n44 \u2502 use sui::dynamic_field;\n   \u2502     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::dynamic_field'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:45:5\n   \u2502\n45 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:46:5\n   \u2502\n46 \u2502 use sui::table::Table;\n   \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:47:5\n   \u2502\n47 \u2502 use sui::vec_map::VecMap;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:60:5\n   \u2502\n60 \u2502 use sui::balance;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system.move:64:5\n   \u2502\n64 \u2502 use sui::vec_set::VecSet;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_set'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:102:5\n    \u2502\n102 \u2502     dynamic_field::add(&mut self.id, version, system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:103:5\n    \u2502\n103 \u2502     transfer::share_object(self);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:236:5\n    \u2502\n236 \u2502     transfer::public_transfer(staked_sui, ctx.sender());\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:254:19\n    \u2502\n254 \u2502     stake_amount: option::Option<u64>,\n    \u2502                   ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid type\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:262:5\n    \u2502\n262 \u2502     transfer::public_transfer(staked_sui, ctx.sender());\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:273:5\n    \u2502\n273 \u2502     transfer::public_transfer(withdrawn_stake.into_coin(ctx), ctx.sender());\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:634:41\n    \u2502\n634 \u2502     let inner: &SuiSystemStateInnerV2 = dynamic_field::borrow(\n    \u2502                                         ^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:644:39\n    \u2502\n644 \u2502         let v1: SuiSystemStateInner = dynamic_field::remove(&mut self.id, self.version);\n    \u2502                                       ^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:647:9\n    \u2502\n647 \u2502         dynamic_field::add(&mut self.id, self.version, v2);\n    \u2502         ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:650:45\n    \u2502\n650 \u2502     let inner: &mut SuiSystemStateInnerV2 = dynamic_field::borrow_mut(\n    \u2502                                             ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:892:26\n    \u2502\n892 \u2502     let storage_reward = balance::create_for_testing(storage_charge);\n    \u2502                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system.move:893:30\n    \u2502\n893 \u2502     let computation_reward = balance::create_for_testing(computation_charge);\n    \u2502                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/sui_system_state_inner.move:6:5\n  \u2502\n6 \u2502 use sui::bag::{Self, Bag};\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'\n\nerror: unbound module\n  \u250c\u2500 ./sources/sui_system_state_inner.move:7:5\n  \u2502\n7 \u2502 use sui::balance::{Self, Balance};\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./sources/sui_system_state_inner.move:8:5\n  \u2502\n8 \u2502 use sui::coin::Coin;\n  \u2502     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'\n\nerror: unbound module\n  \u250c\u2500 ./sources/sui_system_state_inner.move:9:5\n  \u2502\n9 \u2502 use sui::event;\n  \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::event'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system_state_inner.move:10:5\n   \u2502\n10 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system_state_inner.move:11:5\n   \u2502\n11 \u2502 use sui::table::Table;\n   \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system_state_inner.move:12:5\n   \u2502\n12 \u2502 use sui::vec_map::{Self, VecMap};\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'\n\nerror: unbound module\n   \u250c\u2500 ./sources/sui_system_state_inner.move:13:5\n   \u2502\n13 \u2502 use sui::vec_set::{Self, VecSet};\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_set'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:239:35\n    \u2502\n239 \u2502         validator_report_records: vec_map::empty(),\n    \u2502                                   ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:242:36\n    \u2502\n242 \u2502         safe_mode_storage_rewards: balance::zero(),\n    \u2502                                    ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:243:40\n    \u2502\n243 \u2502         safe_mode_computation_rewards: balance::zero(),\n    \u2502                                        ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:247:23\n    \u2502\n247 \u2502         extra_fields: bag::new(ctx),\n    \u2502                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:271:23\n    \u2502\n271 \u2502         extra_fields: bag::new(ctx),\n    \u2502                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:567:56\n    \u2502\n567 \u2502         validator_report_records.insert(reportee_addr, vec_set::singleton(reporter_address));\n    \u2502                                                        ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/sui_system_state_inner.move:907:29\n    \u2502\n907 \u2502     let mut stake_subsidy = balance::zero();\n    \u2502                             ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1000:5\n     \u2502\n1000 \u2502     event::emit(SystemEpochInfoEvent {\n     \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1068:29\n     \u2502\n1068 \u2502     let mut voting_powers = vec_map::empty();\n     \u2502                             ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1095:10\n     \u2502\n1095 \u2502     else vec_set::empty()\n     \u2502          ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1140:13\n     \u2502\n1140 \u2502             transfer::public_transfer(total_balance.into_coin(ctx), ctx.sender());\n     \u2502             ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1299:9\n     \u2502\n1299 \u2502         option::none(),\n     \u2502         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator.move:7:5\n  \u2502\n7 \u2502 use std::bcs;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'std::bcs'\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator.move:8:5\n  \u2502\n8 \u2502 use std::string::String;\n  \u2502     ^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::string'\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator.move:9:5\n  \u2502\n9 \u2502 use sui::bag::{Self, Bag};\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator.move:10:5\n   \u2502\n10 \u2502 use sui::balance::Balance;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator.move:11:5\n   \u2502\n11 \u2502 use sui::event;\n   \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::event'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator.move:12:5\n   \u2502\n12 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator.move:13:5\n   \u2502\n13 \u2502 use sui::url::{Self, Url};\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:199:43\n    \u2502\n199 \u2502         next_epoch_protocol_pubkey_bytes: option::none(),\n    \u2502                                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:200:42\n    \u2502\n200 \u2502         next_epoch_network_pubkey_bytes: option::none(),\n    \u2502                                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:201:41\n    \u2502\n201 \u2502         next_epoch_worker_pubkey_bytes: option::none(),\n    \u2502                                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:202:41\n    \u2502\n202 \u2502         next_epoch_proof_of_possession: option::none(),\n    \u2502                                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:203:33\n    \u2502\n203 \u2502         next_epoch_net_address: option::none(),\n    \u2502                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:204:33\n    \u2502\n204 \u2502         next_epoch_p2p_address: option::none(),\n    \u2502                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:205:37\n    \u2502\n205 \u2502         next_epoch_primary_address: option::none(),\n    \u2502                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:206:36\n    \u2502\n206 \u2502         next_epoch_worker_address: option::none(),\n    \u2502                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:251:9\n    \u2502\n251 \u2502         url::new_unsafe_from_bytes(image_url),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:252:9\n    \u2502\n252 \u2502         url::new_unsafe_from_bytes(project_url),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:257:9\n    \u2502\n257 \u2502         bag::new(ctx),\n    \u2502         ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:297:5\n    \u2502\n297 \u2502     event::emit(StakingRequestEvent {\n    \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:316:5\n    \u2502\n316 \u2502     event::emit(ConvertingToFungibleStakedSuiEvent {\n    \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:336:5\n    \u2502\n336 \u2502     event::emit(RedeemingFungibleStakedSuiEvent {\n    \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:359:5\n    \u2502\n359 \u2502     transfer::public_transfer(staked_sui, staker_address);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:378:5\n    \u2502\n378 \u2502     event::emit(UnstakingRequestEvent {\n    \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:591:5\n    \u2502\n591 \u2502     object::id(&self.staking_pool)\n    \u2502     ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:682:31\n    \u2502\n682 \u2502     self.metadata.image_url = url::new_unsafe_from_bytes(image_url);\n    \u2502                               ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:691:33\n    \u2502\n691 \u2502     self.metadata.project_url = url::new_unsafe_from_bytes(project_url);\n    \u2502                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:704:44\n    \u2502\n704 \u2502     self.metadata.next_epoch_net_address = option::some(net_address);\n    \u2502                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:730:44\n    \u2502\n730 \u2502     self.metadata.next_epoch_p2p_address = option::some(p2p_address);\n    \u2502                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:756:48\n    \u2502\n756 \u2502     self.metadata.next_epoch_primary_address = option::some(primary_address);\n    \u2502                                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:785:47\n    \u2502\n785 \u2502     self.metadata.next_epoch_worker_address = option::some(worker_address);\n    \u2502                                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:811:54\n    \u2502\n811 \u2502     self.metadata.next_epoch_protocol_pubkey_bytes = option::some(protocol_pubkey);\n    \u2502                                                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:812:52\n    \u2502\n812 \u2502     self.metadata.next_epoch_proof_of_possession = option::some(proof_of_possession);\n    \u2502                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:834:53\n    \u2502\n834 \u2502     self.metadata.next_epoch_network_pubkey_bytes = option::some(network_pubkey);\n    \u2502                                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:853:52\n    \u2502\n853 \u2502     self.metadata.next_epoch_worker_pubkey_bytes = option::some(worker_pubkey);\n    \u2502                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:907:27\n    \u2502\n907 \u2502     validate_metadata_bcs(bcs::to_bytes(metadata));\n    \u2502                           ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:943:23\n    \u2502\n943 \u2502         extra_fields: bag::new(ctx),\n    \u2502                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:982:13\n    \u2502\n982 \u2502             url::new_unsafe_from_bytes(image_url),\n    \u2502             ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:983:13\n    \u2502\n983 \u2502             url::new_unsafe_from_bytes(project_url),\n    \u2502             ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator.move:988:13\n    \u2502\n988 \u2502             bag::new(ctx),\n    \u2502             ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./sources/validator_cap.move:51:13\n   \u2502\n51 \u2502         id: object::new(ctx),\n   \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./sources/validator_cap.move:54:28\n   \u2502\n54 \u2502     let operation_cap_id = object::id(&operation_cap);\n   \u2502                            ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./sources/validator_cap.move:55:5\n   \u2502\n55 \u2502     transfer::public_transfer(operation_cap, validator_address);\n   \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator_set.move:6:5\n  \u2502\n6 \u2502 use sui::bag::{Self, Bag};\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator_set.move:7:5\n  \u2502\n7 \u2502 use sui::balance::Balance;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator_set.move:8:5\n  \u2502\n8 \u2502 use sui::event;\n  \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::event'\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator_set.move:9:5\n  \u2502\n9 \u2502 use sui::priority_queue as pq;\n  \u2502     ^^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::priority_queue'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_set.move:10:5\n   \u2502\n10 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_set.move:11:5\n   \u2502\n11 \u2502 use sui::table::{Self, Table};\n   \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_set.move:12:5\n   \u2502\n12 \u2502 use sui::table_vec::{Self, TableVec};\n   \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table_vec'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_set.move:13:5\n   \u2502\n13 \u2502 use sui::vec_map::{Self, VecMap};\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_set.move:14:5\n   \u2502\n14 \u2502 use sui::vec_set::VecSet;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_set'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:146:37\n    \u2502\n146 \u2502     let mut staking_pool_mappings = table::new(ctx);\n    \u2502                                     ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:153:36\n    \u2502\n153 \u2502         pending_active_validators: table_vec::empty(ctx),\n    \u2502                                    ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:156:30\n    \u2502\n156 \u2502         inactive_validators: table::new(ctx),\n    \u2502                              ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:157:31\n    \u2502\n157 \u2502         validator_candidates: table::new(ctx),\n    \u2502                               ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:158:29\n    \u2502\n158 \u2502         at_risk_validators: vec_map::empty(),\n    \u2502                             ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:159:23\n    \u2502\n159 \u2502         extra_fields: bag::new(ctx),\n    \u2502                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./sources/validator_set.move:501:37\n    \u2502\n501 \u2502     let pending_active_validators = vector::tabulate!(\n    \u2502                                     ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:501:37\n    \u2502\n501 \u2502     let pending_active_validators = vector::tabulate!(\n    \u2502                                     ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:585:13\n    \u2502\n585 \u2502             event::emit(ValidatorJoinEvent {\n    \u2502             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:611:23\n    \u2502\n611 \u2502         .map_ref!(|v| pq::new_entry(v.gas_price(), v.voting_power()));\n    \u2502                       ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:614:18\n    \u2502\n614 \u2502     let mut pq = pq::new(entries);\n    \u2502                  ^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:757:32\n    \u2502\n757 \u2502                 return 'search option::some(i)\n    \u2502                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:761:9\n    \u2502\n761 \u2502         option::none()\n    \u2502         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:913:46\n    \u2502\n913 \u2502     assert!(validator.operation_cap_id() == &object::id(cap), EInvalidCap);\n    \u2502                                              ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./sources/validator_set.move:961:5\n    \u2502\n961 \u2502     event::emit(ValidatorLeaveEvent {\n    \u2502     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/validator_set.move:1019:53\n     \u2502\n1019 \u2502     let mut individual_staking_reward_adjustments = vec_map::empty();\n     \u2502                                                     ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/validator_set.move:1021:58\n     \u2502\n1021 \u2502     let mut individual_storage_fund_reward_adjustments = vec_map::empty();\n     \u2502                                                          ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/validator_set.move:1220:13\n     \u2502\n1220 \u2502             transfer::public_transfer(rewards_stake, validator_address);\n     \u2502             ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./sources/validator_set.move:1254:9\n     \u2502\n1254 \u2502         event::emit(ValidatorEpochInfoEventV2 {\n     \u2502         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./sources/validator_wrapper.move:6:5\n  \u2502\n6 \u2502 use sui::versioned::{Self, Versioned};\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::versioned'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./sources/validator_wrapper.move:18:16\n   \u2502\n18 \u2502         inner: versioned::create(1, validator, ctx),\n   \u2502                ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/builders/test_runner.move:9:5\n  \u2502\n9 \u2502 use sui::balance::{Self, Balance};\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/test_runner.move:10:5\n   \u2502\n10 \u2502 use sui::coin::{Self, Coin};\n   \u2502     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/test_runner.move:11:5\n   \u2502\n11 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/test_runner.move:12:5\n   \u2502\n12 \u2502 use sui::test_scenario::{Self, Scenario};\n   \u2502     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:41:21\n   \u2502\n41 \u2502         validators: option::none(),\n   \u2502                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:42:27\n   \u2502\n42 \u2502         validators_count: option::none(),\n   \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:43:28\n   \u2502\n43 \u2502         sui_supply_amount: option::none(),\n   \u2502                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:44:30\n   \u2502\n44 \u2502         storage_fund_amount: option::none(),\n   \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:45:35\n   \u2502\n45 \u2502         validators_initial_stake: option::none(),\n   \u2502                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:46:27\n   \u2502\n46 \u2502         protocol_version: option::none(),\n   \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:47:37\n   \u2502\n47 \u2502         stake_distribution_counter: option::none(),\n   \u2502                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:48:25\n   \u2502\n48 \u2502         epoch_duration: option::none(),\n   \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:49:22\n   \u2502\n49 \u2502         start_epoch: option::none(),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:50:33\n   \u2502\n50 \u2502         low_stake_grace_period: option::none(),\n   \u2502                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:55:24\n   \u2502\n55 \u2502     let mut scenario = test_scenario::begin(@0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n   \u250c\u2500 ./tests/builders/test_runner.move:73:9\n   \u2502\n73 \u2502         vector::tabulate!(validators_count, |idx| {\n   \u2502         ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:73:9\n   \u2502\n73 \u2502         vector::tabulate!(validators_count, |idx| {\n   \u2502         ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/test_runner.move:99:9\n   \u2502\n99 \u2502         balance::create_for_testing<SUI>(sui_supply_amount.destroy_or!(1000) * MIST_PER_SUI), // sui_supply\n   \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:116:9\n    \u2502\n116 \u2502         object::new(scenario.ctx()), // it doesn't matter what ID sui system state has in tests\n    \u2502         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:118:9\n    \u2502\n118 \u2502         balance::create_for_testing<SUI>(storage_fund_amount.destroy_or!(0) * MIST_PER_SUI), // storage_fund\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:152:30\n    \u2502\n152 \u2502     builder.epoch_duration = option::some(epoch_duration);\n    \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:168:32\n    \u2502\n168 \u2502     builder.validators_count = option::some(validators_count);\n    \u2502                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:176:33\n    \u2502\n176 \u2502     builder.sui_supply_amount = option::some(sui_supply_amount);\n    \u2502                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:184:35\n    \u2502\n184 \u2502     builder.storage_fund_amount = option::some(storage_fund_amount);\n    \u2502                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:192:40\n    \u2502\n192 \u2502     builder.validators_initial_stake = option::some(validators_initial_stake);\n    \u2502                                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:197:27\n    \u2502\n197 \u2502     builder.start_epoch = option::some(start_epoch);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:205:32\n    \u2502\n205 \u2502     builder.protocol_version = option::some(protocol_version);\n    \u2502                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:213:42\n    \u2502\n213 \u2502     builder.stake_distribution_counter = option::some(stake_distribution_counter);\n    \u2502                                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:231:27\n    \u2502\n231 \u2502         protocol_version: option::none(),\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:232:25\n    \u2502\n232 \u2502         storage_charge: option::none(),\n    \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:233:29\n    \u2502\n233 \u2502         computation_charge: option::none(),\n    \u2502                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:234:25\n    \u2502\n234 \u2502         storage_rebate: option::none(),\n    \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:235:37\n    \u2502\n235 \u2502         non_refundable_storage_fee: option::none(),\n    \u2502                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:236:37\n    \u2502\n236 \u2502         storage_fund_reinvest_rate: option::none(),\n    \u2502                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:237:31\n    \u2502\n237 \u2502         reward_slashing_rate: option::none(),\n    \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:238:27\n    \u2502\n238 \u2502         epoch_start_time: option::none(),\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:248:29\n    \u2502\n248 \u2502     opts.protocol_version = option::some(protocol_version);\n    \u2502                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:253:27\n    \u2502\n253 \u2502     opts.storage_charge = option::some(storage_charge);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:261:31\n    \u2502\n261 \u2502     opts.computation_charge = option::some(computation_charge);\n    \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:266:27\n    \u2502\n266 \u2502     opts.storage_rebate = option::some(storage_rebate);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:274:39\n    \u2502\n274 \u2502     opts.non_refundable_storage_fee = option::some(non_refundable_storage_fee);\n    \u2502                                       ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:282:39\n    \u2502\n282 \u2502     opts.storage_fund_reinvest_rate = option::some(storage_fund_reinvest_rate);\n    \u2502                                       ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:290:33\n    \u2502\n290 \u2502     opts.reward_slashing_rate = option::some(reward_slashing_rate);\n    \u2502                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:298:29\n    \u2502\n298 \u2502     opts.epoch_start_time = option::some(epoch_start_time);\n    \u2502                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:331:5\n    \u2502\n331 \u2502     transfer::public_transfer(object, runner.sender);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:339:5\n    \u2502\n339 \u2502     balance::create_for_testing(amount * MIST_PER_SUI)\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:381:5\n    \u2502\n381 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:467:13\n    \u2502\n467 \u2502             coin::mint_for_testing(amount * MIST_PER_SUI, ctx),\n    \u2502             ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:486:17\n    \u2502\n486 \u2502                 coin::mint_for_testing(amount * MIST_PER_SUI, ctx),\n    \u2502                 ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:599:5\n    \u2502\n599 \u2502     test_scenario::return_shared(system);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/test_runner.move:615:26\n    \u2502\n615 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/validator_builder.move:12:5\n   \u2502\n12 \u2502 use sui::bag;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/validator_builder.move:13:5\n   \u2502\n13 \u2502 use sui::balance;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/validator_builder.move:14:5\n   \u2502\n14 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./tests/builders/validator_builder.move:15:5\n   \u2502\n15 \u2502 use sui::url;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:44:22\n   \u2502\n44 \u2502         sui_address: option::none(),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:45:32\n   \u2502\n45 \u2502         protocol_pubkey_bytes: option::none(),\n   \u2502                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:46:31\n   \u2502\n46 \u2502         network_pubkey_bytes: option::none(),\n   \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:47:30\n   \u2502\n47 \u2502         worker_pubkey_bytes: option::none(),\n   \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:48:30\n   \u2502\n48 \u2502         proof_of_possession: option::none(),\n   \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:49:15\n   \u2502\n49 \u2502         name: option::none(),\n   \u2502               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:50:22\n   \u2502\n50 \u2502         description: option::none(),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:51:20\n   \u2502\n51 \u2502         image_url: option::none(),\n   \u2502                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:52:22\n   \u2502\n52 \u2502         project_url: option::none(),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:53:22\n   \u2502\n53 \u2502         net_address: option::none(),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:54:22\n   \u2502\n54 \u2502         p2p_address: option::none(),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:55:26\n   \u2502\n55 \u2502         primary_address: option::none(),\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:56:25\n   \u2502\n56 \u2502         worker_address: option::none(),\n   \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:57:20\n   \u2502\n57 \u2502         gas_price: option::none(),\n   \u2502                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:58:26\n   \u2502\n58 \u2502         commission_rate: option::none(),\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:60:24\n   \u2502\n60 \u2502         initial_stake: option::none(),\n   \u2502                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:66:22\n   \u2502\n66 \u2502         sui_address: option::some(preset.account_address()),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:67:32\n   \u2502\n67 \u2502         protocol_pubkey_bytes: option::some(preset.protocol_pubkey_bytes()),\n   \u2502                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:68:31\n   \u2502\n68 \u2502         network_pubkey_bytes: option::some(preset.network_pubkey_bytes()),\n   \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:69:30\n   \u2502\n69 \u2502         worker_pubkey_bytes: option::some(preset.worker_pubkey_bytes()),\n   \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:70:30\n   \u2502\n70 \u2502         proof_of_possession: option::some(preset.proof_of_possession()),\n   \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:71:15\n   \u2502\n71 \u2502         name: option::some(preset.name()),\n   \u2502               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:72:22\n   \u2502\n72 \u2502         description: option::some(preset.description()),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:73:20\n   \u2502\n73 \u2502         image_url: option::some(preset.image_url()),\n   \u2502                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:74:22\n   \u2502\n74 \u2502         project_url: option::some(preset.project_url()),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:75:22\n   \u2502\n75 \u2502         net_address: option::some(preset.net_address()),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:76:22\n   \u2502\n76 \u2502         p2p_address: option::some(preset.p2p_address()),\n   \u2502                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:77:26\n   \u2502\n77 \u2502         primary_address: option::some(preset.primary_address()),\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:78:25\n   \u2502\n78 \u2502         worker_address: option::some(preset.worker_address()),\n   \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:79:20\n   \u2502\n79 \u2502         gas_price: option::none(),\n   \u2502                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:80:26\n   \u2502\n80 \u2502         commission_rate: option::none(),\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/builders/validator_builder.move:82:24\n   \u2502\n82 \u2502         initial_stake: option::none(),\n   \u2502                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:127:37\n    \u2502\n127 \u2502         initial_stake.map!(|amount| balance::create_for_testing<SUI>(amount * 1_000_000_000)),\n    \u2502                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:165:9\n    \u2502\n165 \u2502         url::new_unsafe_from_bytes(image_url.destroy_or!(b\"image_url\")),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:166:9\n    \u2502\n166 \u2502         url::new_unsafe_from_bytes(project_url.destroy_or!(b\"project_url\")),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:171:9\n    \u2502\n171 \u2502         bag::new(ctx),\n    \u2502         ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:178:27\n    \u2502\n178 \u2502     builder.sui_address = option::some(sui_address);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:186:37\n    \u2502\n186 \u2502     builder.protocol_pubkey_bytes = option::some(protocol_pubkey_bytes);\n    \u2502                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:194:36\n    \u2502\n194 \u2502     builder.network_pubkey_bytes = option::some(network_pubkey_bytes);\n    \u2502                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:202:35\n    \u2502\n202 \u2502     builder.worker_pubkey_bytes = option::some(worker_pubkey_bytes);\n    \u2502                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:210:35\n    \u2502\n210 \u2502     builder.proof_of_possession = option::some(proof_of_possession);\n    \u2502                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:215:20\n    \u2502\n215 \u2502     builder.name = option::some(name);\n    \u2502                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:220:27\n    \u2502\n220 \u2502     builder.description = option::some(description);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:225:25\n    \u2502\n225 \u2502     builder.image_url = option::some(image_url);\n    \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:230:27\n    \u2502\n230 \u2502     builder.project_url = option::some(project_url);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:235:27\n    \u2502\n235 \u2502     builder.net_address = option::some(net_address);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:240:27\n    \u2502\n240 \u2502     builder.p2p_address = option::some(p2p_address);\n    \u2502                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:248:31\n    \u2502\n248 \u2502     builder.primary_address = option::some(primary_address);\n    \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:256:30\n    \u2502\n256 \u2502     builder.worker_address = option::some(worker_address);\n    \u2502                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:261:25\n    \u2502\n261 \u2502     builder.gas_price = option::some(gas_price);\n    \u2502                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:266:31\n    \u2502\n266 \u2502     builder.commission_rate = option::some(commission_rate);\n    \u2502                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/builders/validator_builder.move:272:29\n    \u2502\n272 \u2502     builder.initial_stake = option::some(initial_stake);\n    \u2502                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/delegation_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test::assert_eq;\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/delegation_tests.move:8:5\n  \u2502\n8 \u2502 use sui::table::Table;\n  \u2502     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/delegation_tests.move:72:26\n   \u2502\n72 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:143:26\n    \u2502\n143 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:159:26\n    \u2502\n159 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:198:26\n    \u2502\n198 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:208:9\n    \u2502\n208 \u2502         option::some(runner.advance_epoch_opts().computation_charge(80))\n    \u2502         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:210:9\n    \u2502\n210 \u2502         option::none()\n    \u2502         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:258:26\n    \u2502\n258 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:264:26\n    \u2502\n264 \u2502     runner.advance_epoch(option::some(options)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:307:26\n    \u2502\n307 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:313:26\n    \u2502\n313 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:343:26\n    \u2502\n343 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:347:26\n    \u2502\n347 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:410:26\n    \u2502\n410 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:433:26\n    \u2502\n433 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:449:26\n    \u2502\n449 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:463:26\n    \u2502\n463 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:499:26\n    \u2502\n499 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:504:26\n    \u2502\n504 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:510:26\n    \u2502\n510 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:555:26\n    \u2502\n555 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:591:26\n    \u2502\n591 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:597:26\n    \u2502\n597 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:598:26\n    \u2502\n598 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:599:26\n    \u2502\n599 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:660:26\n    \u2502\n660 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:676:26\n    \u2502\n676 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:716:26\n    \u2502\n716 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/delegation_tests.move:721:26\n    \u2502\n721 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/governance_test_utils.move:8:5\n  \u2502\n8 \u2502 use std::unit_test::{assert_eq, destroy};\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/governance_test_utils.move:9:5\n  \u2502\n9 \u2502 use sui::balance::{Self, Balance};\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./tests/governance_test_utils.move:10:5\n   \u2502\n10 \u2502 use sui::coin::{Self, Coin};\n   \u2502     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'\n\nerror: unbound module\n   \u250c\u2500 ./tests/governance_test_utils.move:11:5\n   \u2502\n11 \u2502 use sui::sui::SUI;\n   \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unbound module\n   \u250c\u2500 ./tests/governance_test_utils.move:12:5\n   \u2502\n12 \u2502 use sui::test_scenario::{Self, Scenario};\n   \u2502     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/governance_test_utils.move:40:9\n   \u2502\n40 \u2502         option::some(balance::create_for_testing<SUI>(init_stake_amount_in_sui * MIST_PER_SUI)),\n   \u2502         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/governance_test_utils.move:40:22\n   \u2502\n40 \u2502         option::some(balance::create_for_testing<SUI>(init_stake_amount_in_sui * MIST_PER_SUI)),\n   \u2502                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/governance_test_utils.move:67:9\n   \u2502\n67 \u2502         balance::create_for_testing<SUI>(sui_supply_amount * MIST_PER_SUI), // sui_supply\n   \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/governance_test_utils.move:75:9\n   \u2502\n75 \u2502         object::new(ctx), // it doesn't matter what ID sui system state has in tests\n   \u2502         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/governance_test_utils.move:77:9\n   \u2502\n77 \u2502         balance::create_for_testing<SUI>(storage_fund_amount * MIST_PER_SUI), // storage_fund\n   \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/governance_test_utils.move:87:24\n   \u2502\n87 \u2502     let mut scenario = test_scenario::begin(@0x0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:130:5\n    \u2502\n130 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:175:5\n    \u2502\n175 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:186:9\n    \u2502\n186 \u2502         coin::mint_for_testing(amount * MIST_PER_SUI, ctx),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:190:5\n    \u2502\n190 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:201:5\n    \u2502\n201 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:236:9\n    \u2502\n236 \u2502         coin::mint_for_testing<SUI>(init_stake_amount * MIST_PER_SUI, ctx),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:241:5\n    \u2502\n241 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:274:5\n    \u2502\n274 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:283:5\n    \u2502\n283 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:292:5\n    \u2502\n292 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:302:5\n    \u2502\n302 \u2502     test_scenario::return_shared(system_state);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:323:9\n    \u2502\n323 \u2502         test_scenario::return_shared(system_state);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:342:9\n    \u2502\n342 \u2502         test_scenario::return_shared(system_state);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/governance_test_utils.move:361:9\n    \u2502\n361 \u2502         test_scenario::return_shared(system_state);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/rewards_distribution_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test::assert_eq;\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/rewards_distribution_tests.move:8:5\n  \u2502\n8 \u2502 use sui::address;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::address'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:39:26\n   \u2502\n39 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:52:26\n   \u2502\n52 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:78:26\n   \u2502\n78 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:108:26\n    \u2502\n108 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:127:26\n    \u2502\n127 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:141:26\n    \u2502\n141 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:157:26\n    \u2502\n157 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:188:26\n    \u2502\n188 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:193:26\n    \u2502\n193 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:200:26\n    \u2502\n200 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:227:24\n    \u2502\n227 \u2502         .advance_epoch(option::none())\n    \u2502                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:240:26\n    \u2502\n240 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:266:24\n    \u2502\n266 \u2502         .advance_epoch(option::none())\n    \u2502                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:270:26\n    \u2502\n270 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:321:26\n    \u2502\n321 \u2502     runner.advance_epoch(option::none()).destroy_zero();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:322:26\n    \u2502\n322 \u2502     runner.advance_epoch(option::none()).destroy_zero();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:348:26\n    \u2502\n348 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:355:26\n    \u2502\n355 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:393:26\n    \u2502\n393 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:403:26\n    \u2502\n403 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:446:26\n    \u2502\n446 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:451:26\n    \u2502\n451 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:465:26\n    \u2502\n465 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:520:26\n    \u2502\n520 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:554:26\n    \u2502\n554 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:560:26\n    \u2502\n560 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:567:26\n    \u2502\n567 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:574:26\n    \u2502\n574 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:603:26\n    \u2502\n603 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:616:22\n    \u2502\n616 \u2502     let validators = vector::tabulate!(num_validators, |i| {\n    \u2502                      ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:616:22\n    \u2502\n616 \u2502     let validators = vector::tabulate!(num_validators, |i| {\n    \u2502                      ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:619:26\n    \u2502\n619 \u2502             .sui_address(address::from_u256(i as u256))\n    \u2502                          ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:626:26\n    \u2502\n626 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:631:24\n    \u2502\n631 \u2502             let addr = address::from_u256(i as u256);\n    \u2502                        ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:668:26\n    \u2502\n668 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:683:26\n    \u2502\n683 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:725:26\n    \u2502\n725 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:740:26\n    \u2502\n740 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:774:26\n    \u2502\n774 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:839:26\n    \u2502\n839 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:846:19\n    \u2502\n846 \u2502         pool_id = object::id(pool);\n    \u2502                   ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:859:26\n    \u2502\n859 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:904:26\n    \u2502\n904 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:932:26\n    \u2502\n932 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:973:26\n    \u2502\n973 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:982:19\n    \u2502\n982 \u2502         pool_id = object::id(pool);\n    \u2502                   ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1024:26\n     \u2502\n1024 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n     \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/staking_pool_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test::{assert_eq, destroy};\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/staking_pool_tests.move:8:5\n  \u2502\n8 \u2502 use sui::balance;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./tests/staking_pool_tests.move:9:5\n  \u2502\n9 \u2502 use sui::test_scenario::{Self, Scenario};\n  \u2502     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/staking_pool_tests.move:14:24\n   \u2502\n14 \u2502     let mut scenario = test_scenario::begin(@0x0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/staking_pool_tests.move:38:24\n   \u2502\n38 \u2502     let mut scenario = test_scenario::begin(@0x0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/staking_pool_tests.move:58:24\n   \u2502\n58 \u2502     let mut scenario = test_scenario::begin(@0x0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/staking_pool_tests.move:80:24\n   \u2502\n80 \u2502     let mut scenario = test_scenario::begin(@0x0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/staking_pool_tests.move:95:24\n   \u2502\n95 \u2502     let mut scenario = test_scenario::begin(@0x0);\n   \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/staking_pool_tests.move:98:15\n   \u2502\n98 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n   \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:114:24\n    \u2502\n114 \u2502     let mut scenario = test_scenario::begin(@0x0);\n    \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:117:15\n    \u2502\n117 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:136:24\n    \u2502\n136 \u2502     let mut scenario = test_scenario::begin(@0x0);\n    \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:139:15\n    \u2502\n139 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:159:24\n    \u2502\n159 \u2502     let mut scenario = test_scenario::begin(@0x0);\n    \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:163:15\n    \u2502\n163 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:180:24\n    \u2502\n180 \u2502     let mut scenario = test_scenario::begin(@0x0);\n    \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:186:15\n    \u2502\n186 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:199:15\n    \u2502\n199 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:226:49\n    \u2502\n226 \u2502     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));\n    \u2502                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:237:49\n    \u2502\n237 \u2502     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));\n    \u2502                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:253:20\n    \u2502\n253 \u2502     let mut test = test_scenario::begin(@0x0);\n    \u2502                    ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:257:15\n    \u2502\n257 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:278:24\n    \u2502\n278 \u2502     let mut scenario = test_scenario::begin(@0x0);\n    \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:284:15\n    \u2502\n284 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:297:15\n    \u2502\n297 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:322:49\n    \u2502\n322 \u2502     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));\n    \u2502                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:333:49\n    \u2502\n333 \u2502     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));\n    \u2502                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:385:24\n    \u2502\n385 \u2502     let mut scenario = test_scenario::begin(@0x0);\n    \u2502                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:391:15\n    \u2502\n391 \u2502     let sui = balance::create_for_testing(1_000_000_000);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:404:15\n    \u2502\n404 \u2502     let sui = balance::create_for_testing(1_000_000_001);\n    \u2502               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:429:47\n    \u2502\n429 \u2502     assert_eq!(fungible_staked_sui.pool_id(), object::id(&staking_pool));\n    \u2502                                               ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n    \u250c\u2500 ./tests/staking_pool_tests.move:455:9\n    \u2502\n455 \u2502     use sui::tx_context::epoch;\n    \u2502         ^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::tx_context'\n\nerror: unbound module\n    \u250c\u2500 ./tests/staking_pool_tests.move:456:9\n    \u2502\n456 \u2502     use sui::coin;\n    \u2502         ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'\n\nerror: unbound module\n    \u250c\u2500 ./tests/staking_pool_tests.move:457:9\n    \u2502\n457 \u2502     use sui::sui::SUI;\n    \u2502         ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:459:19\n    \u2502\n459 \u2502     let rewards = coin::mint_for_testing<SUI>(reward_amount, scenario.ctx());\n    \u2502                   ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:460:34\n    \u2502\n460 \u2502     staking_pool.deposit_rewards(coin::into_balance(rewards));\n    \u2502                                  ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/staking_pool_tests.move:463:5\n    \u2502\n463 \u2502     test_scenario::next_epoch(scenario, @0x0);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n   \u250c\u2500 ./tests/sui_system_tests.move:11:5\n   \u2502\n11 \u2502 use std::unit_test::assert_eq;\n   \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/sui_system_tests.move:58:26\n   \u2502\n58 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/sui_system_tests.move:79:26\n   \u2502\n79 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/sui_system_tests.move:86:26\n   \u2502\n86 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n   \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:116:9\n    \u2502\n116 \u2502         transfer::public_transfer(cap, stakee);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:135:9\n    \u2502\n135 \u2502         transfer::public_transfer(cap, new_stakee);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:163:26\n    \u2502\n163 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:171:26\n    \u2502\n171 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:193:9\n    \u2502\n193 \u2502         transfer::public_transfer(cap, stakee);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:222:9\n    \u2502\n222 \u2502         transfer::public_transfer(cap, stakee);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:357:26\n    \u2502\n357 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:381:26\n    \u2502\n381 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:421:26\n    \u2502\n421 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:431:26\n    \u2502\n431 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:441:26\n    \u2502\n441 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:512:26\n    \u2502\n512 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/sui_system_tests.move:535:26\n    \u2502\n535 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_metadata_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test;\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_metadata_tests.move:8:5\n  \u2502\n8 \u2502 use sui::test_scenario::{Self, Scenario};\n  \u2502     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_metadata_tests.move:9:5\n  \u2502\n9 \u2502 use sui::url;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_metadata_tests.move:21:20\n   \u2502\n21 \u2502     let ctx = &mut tx_context::dummy();\n   \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_metadata_tests.move:27:5\n   \u2502\n27 \u2502     unit_test::destroy(vector[validator_0, validator_1, validator_2, validator_3]);\n   \u2502     ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:327:26\n    \u2502\n327 \u2502     runner.advance_epoch(option::none()).destroy_zero();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:520:9\n    \u2502\n520 \u2502         test_scenario::return_shared(system_state);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:551:26\n    \u2502\n551 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:594:9\n    \u2502\n594 \u2502         test_scenario::return_shared(system_state);\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:600:26\n    \u2502\n600 \u2502     runner.advance_epoch(option::some(opts)).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:669:26\n    \u2502\n669 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:675:26\n    \u2502\n675 \u2502     runner.advance_epoch(option::none()).destroy_for_testing();\n    \u2502                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:843:52\n    \u2502\n843 \u2502         validator.next_epoch_network_address() == &option::some(new_network_address.to_string()),\n    \u2502                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:845:52\n    \u2502\n845 \u2502     assert!(validator.next_epoch_p2p_address() == &option::some(new_p2p_address.to_string()));\n    \u2502                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:847:52\n    \u2502\n847 \u2502         validator.next_epoch_primary_address() == &option::some(new_primary_address.to_string()),\n    \u2502                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:849:55\n    \u2502\n849 \u2502     assert!(validator.next_epoch_worker_address() == &option::some(new_worker_address.to_string()));\n    \u2502                                                       ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:850:62\n    \u2502\n850 \u2502     assert!(validator.next_epoch_protocol_pubkey_bytes() == &option::some(new_protocol_pub_key), 0);\n    \u2502                                                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:851:60\n    \u2502\n851 \u2502     assert!(validator.next_epoch_proof_of_possession() == &option::some(new_pop), 0);\n    \u2502                                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:852:60\n    \u2502\n852 \u2502     assert!(validator.next_epoch_worker_pubkey_bytes() == &option::some(new_worker_pubkey), 0);\n    \u2502                                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:853:61\n    \u2502\n853 \u2502     assert!(validator.next_epoch_network_pubkey_bytes() == &option::some(new_network_pubkey), 0);\n    \u2502                                                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:871:39\n    \u2502\n871 \u2502     assert!(validator.image_url() == &url::new_unsafe_from_bytes(b\"new_image_url\"));\n    \u2502                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_metadata_tests.move:872:41\n    \u2502\n872 \u2502     assert!(validator.project_url() == &url::new_unsafe_from_bytes(b\"new_project_url\"));\n    \u2502                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_set_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test::{assert_eq, destroy};\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_set_tests.move:8:5\n  \u2502\n8 \u2502 use sui::address;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::address'\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_set_tests.move:9:5\n  \u2502\n9 \u2502 use sui::balance;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n   \u250c\u2500 ./tests/validator_set_tests.move:10:5\n   \u2502\n10 \u2502 use sui::coin;\n   \u2502     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'\n\nerror: unbound module\n   \u250c\u2500 ./tests/validator_set_tests.move:11:5\n   \u2502\n11 \u2502 use sui::test_scenario::{Self, Scenario};\n   \u2502     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'\n\nerror: unbound module\n   \u250c\u2500 ./tests/validator_set_tests.move:12:5\n   \u2502\n12 \u2502 use sui::vec_map;\n   \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_set_tests.move:21:28\n   \u2502\n21 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n   \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_set_tests.move:49:28\n   \u2502\n49 \u2502     let mut scenario_val = test_scenario::begin(@0x1);\n   \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_set_tests.move:55:13\n   \u2502\n55 \u2502             coin::mint_for_testing(500 * MIST_PER_SUI, ctx1).into_balance(),\n   \u2502             ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_set_tests.move:58:9\n   \u2502\n58 \u2502         transfer::public_transfer(stake, @0x1);\n   \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_set_tests.move:94:28\n   \u2502\n94 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n   \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:145:28\n    \u2502\n145 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:154:28\n    \u2502\n154 \u2502     let mut scenario_val = test_scenario::begin(@0x1);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:160:9\n    \u2502\n160 \u2502         balance::create_for_testing(MIST_PER_SUI - 1), // 1 MIST lower than the threshold\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:163:5\n    \u2502\n163 \u2502     transfer::public_transfer(stake, @0x1);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:170:28\n    \u2502\n170 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:179:28\n    \u2502\n179 \u2502     let mut scenario_val = test_scenario::begin(@0x1);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:184:9\n    \u2502\n184 \u2502         balance::create_for_testing(MIST_PER_SUI), // min possible stake\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:187:5\n    \u2502\n187 \u2502     transfer::public_transfer(stake, @0x1);\n    \u2502     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:198:28\n    \u2502\n198 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:250:28\n    \u2502\n250 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:283:28\n    \u2502\n283 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:298:28\n    \u2502\n298 \u2502     let mut scenario_val = test_scenario::begin(@0x1);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:320:28\n    \u2502\n320 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:335:9\n    \u2502\n335 \u2502         balance::create_for_testing(3 * MIST_PER_SUI),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:348:28\n    \u2502\n348 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:361:9\n    \u2502\n361 \u2502         balance::create_for_testing(4 * MIST_PER_SUI),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:385:28\n    \u2502\n385 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:399:9\n    \u2502\n399 \u2502         balance::create_for_testing(4 * MIST_PER_SUI),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:426:28\n    \u2502\n426 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:440:9\n    \u2502\n440 \u2502         balance::create_for_testing(4 * MIST_PER_SUI),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:485:28\n    \u2502\n485 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:499:9\n    \u2502\n499 \u2502         balance::create_for_testing(4 * MIST_PER_SUI),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:531:28\n    \u2502\n531 \u2502     let mut scenario_val = test_scenario::begin(@0x0);\n    \u2502                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:544:9\n    \u2502\n544 \u2502         balance::create_for_testing(1000 * MIST_PER_SUI),\n    \u2502         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:556:13\n    \u2502\n556 \u2502             address::from_u256((i + 1 as u256)),\n    \u2502             ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:557:13\n    \u2502\n557 \u2502             balance::create_for_testing(to_add),\n    \u2502             ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:600:9\n    \u2502\n600 \u2502         option::some(balance::create_for_testing(stake_value)),\n    \u2502         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:600:22\n    \u2502\n600 \u2502         option::some(balance::create_for_testing(stake_value)),\n    \u2502                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:632:13\n    \u2502\n632 \u2502             option::some(balance::create_for_testing(initial_stake * MIST_PER_SUI))\n    \u2502             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:632:26\n    \u2502\n632 \u2502             option::some(balance::create_for_testing(initial_stake * MIST_PER_SUI))\n    \u2502                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:633:18\n    \u2502\n633 \u2502         } else { option::none() },\n    \u2502                  ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:653:40\n    \u2502\n653 \u2502     let mut dummy_computation_reward = balance::zero();\n    \u2502                                        ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:654:41\n    \u2502\n654 \u2502     let mut dummy_storage_fund_reward = balance::zero();\n    \u2502                                         ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_set_tests.move:659:14\n    \u2502\n659 \u2502         &mut vec_map::empty(),\n    \u2502              ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test::{assert_eq, destroy};\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_tests.move:8:5\n  \u2502\n8 \u2502 use sui::balance;\n  \u2502     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'\n\nerror: unbound module\n  \u250c\u2500 ./tests/validator_tests.move:9:5\n  \u2502\n9 \u2502 use sui::url;\n  \u2502     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_tests.move:84:41\n   \u2502\n84 \u2502         validator.deposit_stake_rewards(balance::zero());\n   \u2502                                         ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/validator_tests.move:99:20\n   \u2502\n99 \u2502     let ctx = &mut tx_context::dummy();\n   \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:107:20\n    \u2502\n107 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:119:20\n    \u2502\n119 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:131:20\n    \u2502\n131 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:143:20\n    \u2502\n143 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:153:20\n    \u2502\n153 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:163:20\n    \u2502\n163 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:173:20\n    \u2502\n173 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:183:20\n    \u2502\n183 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:194:20\n    \u2502\n194 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:205:20\n    \u2502\n205 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:228:20\n    \u2502\n228 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:248:40\n    \u2502\n248 \u2502     assert_eq!(*validator.image_url(), url::new_unsafe_from_bytes(b\"new_image_url\"));\n    \u2502                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:249:42\n    \u2502\n249 \u2502     assert_eq!(*validator.project_url(), url::new_unsafe_from_bytes(b\"new_proj_url\"));\n    \u2502                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:298:20\n    \u2502\n298 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:311:20\n    \u2502\n311 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:321:20\n    \u2502\n321 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:331:20\n    \u2502\n331 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:341:20\n    \u2502\n341 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:351:20\n    \u2502\n351 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:361:20\n    \u2502\n361 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:376:20\n    \u2502\n376 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:379:49\n    \u2502\n379 \u2502     validator.update_next_epoch_primary_address(vector::tabulate!(257, |_| 0));\n    \u2502                                                 ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:379:49\n    \u2502\n379 \u2502     validator.update_next_epoch_primary_address(vector::tabulate!(257, |_| 0));\n    \u2502                                                 ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:390:20\n    \u2502\n390 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:393:49\n    \u2502\n393 \u2502     validator.update_next_epoch_network_address(vector::tabulate!(257, |_| 0));\n    \u2502                                                 ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:393:49\n    \u2502\n393 \u2502     validator.update_next_epoch_network_address(vector::tabulate!(257, |_| 0));\n    \u2502                                                 ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:405:20\n    \u2502\n405 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:408:48\n    \u2502\n408 \u2502     validator.update_next_epoch_worker_address(vector::tabulate!(257, |_| 0));\n    \u2502                                                ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:408:48\n    \u2502\n408 \u2502     validator.update_next_epoch_worker_address(vector::tabulate!(257, |_| 0));\n    \u2502                                                ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:420:20\n    \u2502\n420 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:423:45\n    \u2502\n423 \u2502     validator.update_next_epoch_p2p_address(vector::tabulate!(257, |_| 0));\n    \u2502                                             ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:423:45\n    \u2502\n423 \u2502     validator.update_next_epoch_p2p_address(vector::tabulate!(257, |_| 0));\n    \u2502                                             ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:435:20\n    \u2502\n435 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:438:27\n    \u2502\n438 \u2502     validator.update_name(vector::tabulate!(257, |_| 0));\n    \u2502                           ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:438:27\n    \u2502\n438 \u2502     validator.update_name(vector::tabulate!(257, |_| 0));\n    \u2502                           ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:450:20\n    \u2502\n450 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:453:34\n    \u2502\n453 \u2502     validator.update_description(vector::tabulate!(257, |_| 0));\n    \u2502                                  ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:453:34\n    \u2502\n453 \u2502     validator.update_description(vector::tabulate!(257, |_| 0));\n    \u2502                                  ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:465:20\n    \u2502\n465 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:468:34\n    \u2502\n468 \u2502     validator.update_project_url(vector::tabulate!(257, |_| 0));\n    \u2502                                  ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:468:34\n    \u2502\n468 \u2502     validator.update_project_url(vector::tabulate!(257, |_| 0));\n    \u2502                                  ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:480:20\n    \u2502\n480 \u2502     let ctx = &mut tx_context::dummy();\n    \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/validator_tests.move:483:32\n    \u2502\n483 \u2502     validator.update_image_url(vector::tabulate!(257, |_| 0));\n    \u2502                                ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/validator_tests.move:483:32\n    \u2502\n483 \u2502     validator.update_image_url(vector::tabulate!(257, |_| 0));\n    \u2502                                ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound module\n  \u250c\u2500 ./tests/voting_power_tests.move:7:5\n  \u2502\n7 \u2502 use std::unit_test::{assert_eq, destroy};\n  \u2502     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'\n\nerror: invalid use of reserved name\n   \u250c\u2500 ./tests/voting_power_tests.move:21:25\n   \u2502\n21 \u2502     let voting_powers = vector::tabulate!(\n   \u2502                         ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/voting_power_tests.move:21:25\n   \u2502\n21 \u2502     let voting_powers = vector::tabulate!(\n   \u2502                         ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/voting_power_tests.move:32:20\n   \u2502\n32 \u2502     let ctx = &mut tx_context::dummy();\n   \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/voting_power_tests.move:72:20\n   \u2502\n72 \u2502     let ctx = &mut tx_context::dummy();\n   \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unexpected name in this position\n   \u250c\u2500 ./tests/voting_power_tests.move:98:20\n   \u2502\n98 \u2502     let ctx = &mut tx_context::dummy();\n   \u2502                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: invalid use of reserved name\n    \u250c\u2500 ./tests/voting_power_tests.move:118:5\n    \u2502\n118 \u2502     vector::tabulate!(stakes.length(), |i| {\n    \u2502     ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address\n\nerror: unexpected name in this position\n    \u250c\u2500 ./tests/voting_power_tests.move:118:5\n    \u2502\n118 \u2502     vector::tabulate!(stakes.length(), |i| {\n    \u2502     ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression\n\nerror: unbound type\n    \u250c\u2500 ./tests/delegation_tests.move:736:36\n    \u2502\n736 \u2502 use fun assert_exchange_rate_eq as Table.assert_exchange_rate_eq;\n    \u2502                                    ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:603:5\n    \u2502\n603 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:415:9\n    \u2502\n415 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:416:9\n    \u2502\n416 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:417:9\n    \u2502\n417 \u2502         assert_eq!(\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:438:9\n    \u2502\n438 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:439:9\n    \u2502\n439 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:440:9\n    \u2502\n440 \u2502         assert_eq!(system.validator_stake_amount(NEW_VALIDATOR_ADDR), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:441:9\n    \u2502\n441 \u2502         assert_eq!(\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:456:5\n    \u2502\n456 \u2502     assert_eq!(runner.sui_balance(), 110002000000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:459:5\n    \u2502\n459 \u2502     assert_eq!(runner.sui_balance(), 110002000000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:468:5\n    \u2502\n468 \u2502     assert_eq!(runner.sui_balance(), 78862939078);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:514:5\n    \u2502\n514 \u2502     assert_eq!(runner.sui_balance(), 130006000000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:351:5\n    \u2502\n351 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:138:9\n    \u2502\n138 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:139:9\n    \u2502\n139 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:149:13\n    \u2502\n149 \u2502             assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 160 * MIST_PER_SUI);\n    \u2502             ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:150:13\n    \u2502\n150 \u2502             assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 100 * MIST_PER_SUI);\n    \u2502             ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:154:13\n    \u2502\n154 \u2502             assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 160 * MIST_PER_SUI);\n    \u2502             ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:161:9\n    \u2502\n161 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:545:9\n    \u2502\n545 \u2502         assert_eq!(validator.total_stake(), 200 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:546:9\n    \u2502\n546 \u2502         assert_eq!(validator.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:547:9\n    \u2502\n547 \u2502         assert_eq!(validator.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:552:5\n    \u2502\n552 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:559:5\n    \u2502\n559 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:565:9\n    \u2502\n565 \u2502         assert_eq!(validator.total_stake(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:566:9\n    \u2502\n566 \u2502         assert_eq!(validator.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:567:9\n    \u2502\n567 \u2502         assert_eq!(validator.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/delegation_tests.move:739:13\n    \u2502\n739 \u2502     rates: &Table<u64, PoolTokenExchangeRate>,\n    \u2502             ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:745:5\n    \u2502\n745 \u2502     assert_eq!(rate.sui_amount(), sui_amount * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:746:5\n    \u2502\n746 \u2502     assert_eq!(rate.pool_token_amount(), pool_token_amount * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:273:9\n    \u2502\n273 \u2502         assert_eq!(stake.amount(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:281:5\n    \u2502\n281 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:289:5\n    \u2502\n289 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt + validator_reward_amt);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:634:5\n    \u2502\n634 \u2502     assert_eq!(runner.set_sender(validator_address).sui_balance(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:202:9\n    \u2502\n202 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 200 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:203:9\n    \u2502\n203 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:224:9\n    \u2502\n224 \u2502         assert_eq!(stake.amount(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:233:5\n    \u2502\n233 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:239:5\n    \u2502\n239 \u2502     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt + validator_reward_amt);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/delegation_tests.move:45:9\n   \u2502\n45 \u2502         assert_eq!(ids.length(), 2);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/delegation_tests.move:50:9\n   \u2502\n50 \u2502         assert_eq!(stake_1.amount(), 20 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/delegation_tests.move:51:9\n   \u2502\n51 \u2502         assert_eq!(stake_2.amount(), 40 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/delegation_tests.move:58:9\n   \u2502\n58 \u2502         assert_eq!(stake.amount(), 60 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/delegation_tests.move:726:9\n    \u2502\n726 \u2502         assert_eq!(rates.length(), 3);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/delegation_tests.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::delegation_tests;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use std::unit_test::assert_eq;\n  8 \u2502 \u2502 use sui::table::Table;\n    \u00b7 \u2502\n746 \u2502 \u2502     assert_eq!(rate.pool_token_amount(), pool_token_amount * MIST_PER_SUI);\n747 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n   \u250c\u2500 ./sources/genesis.move:57:28\n   \u2502\n57 \u2502     staked_with_validator: Option<address>,\n   \u2502                            ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/genesis.move:179:21\n    \u2502\n179 \u2502     mut sui_supply: Balance<SUI>,\n    \u2502                     ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/genesis.move:182:15\n    \u2502\n182 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/genesis.move:73:26\n   \u2502\n73 \u2502     sui_system_state_id: UID,\n   \u2502                          ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/genesis.move:74:21\n   \u2502\n74 \u2502     mut sui_supply: Balance<SUI>,\n   \u2502                     ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/genesis.move:78:15\n   \u2502\n78 \u2502     ctx: &mut TxContext,\n   \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./sources/genesis.move:4:1\n    \u2502  \n  4 \u2502 \u256d module sui_system::genesis;\n  5 \u2502 \u2502 \n  6 \u2502 \u2502 use sui::balance::{Self, Balance};\n  7 \u2502 \u2502 use sui::sui::SUI;\n    \u00b7 \u2502\n206 \u2502 \u2502     sui_supply.destroy_zero();\n207 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:286:61\n    \u2502\n286 \u2502 public fun add_validator(validator: address, scenario: &mut Scenario) {\n    \u2502                                                             ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:250:20\n    \u2502\n250 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:211:20\n    \u2502\n211 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:101:41\n    \u2502\n101 \u2502 public fun advance_epoch(scenario: &mut Scenario) {\n    \u2502                                         ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:138:20\n    \u2502\n138 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/governance_test_utils.move:147:5\n    \u2502\n147 \u2502     destroy(storage_rebate)\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:154:20\n    \u2502\n154 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/governance_test_utils.move:174:5\n    \u2502\n174 \u2502     destroy(storage_rebate);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:110:20\n    \u2502\n110 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:111:4\n    \u2502\n111 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:350:20\n    \u2502\n350 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/governance_test_utils.move:360:9\n    \u2502\n360 \u2502         assert_eq!(non_self_stake_amount, amount);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:308:20\n    \u2502\n308 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/governance_test_utils.move:322:9\n    \u2502\n322 \u2502         assert_eq!(stake_plus_rewards, amount);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:331:20\n    \u2502\n331 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/governance_test_utils.move:53:15\n   \u2502\n53 \u2502     ctx: &mut TxContext,\n   \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/governance_test_utils.move:24:15\n   \u2502\n24 \u2502     ctx: &mut TxContext,\n   \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:295:64\n    \u2502\n295 \u2502 public fun remove_validator(validator: address, scenario: &mut Scenario) {\n    \u2502                                                                ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:277:74\n    \u2502\n277 \u2502 public fun remove_validator_candidate(validator: address, scenario: &mut Scenario) {\n    \u2502                                                                          ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:380:20\n    \u2502\n380 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:370:20\n    \u2502\n370 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:179:88\n    \u2502\n179 \u2502 public fun stake_with(staker: address, validator: address, amount: u64, scenario: &mut Scenario) {\n    \u2502                                                                                        ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:397:60\n    \u2502\n397 \u2502 public fun total_sui_balance(addr: address, scenario: &mut Scenario): u64 {\n    \u2502                                                            ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:400:44\n    \u2502\n400 \u2502     let coin_ids = scenario.ids_for_sender<Coin<SUI>>();\n    \u2502                                            ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:403:52\n    \u2502\n403 \u2502         let coin = scenario.take_from_sender_by_id<Coin<SUI>>(coin_ids[i]);\n    \u2502                                                    ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/governance_test_utils.move:6:1\n    \u2502  \n  6 \u2502 \u256d module sui_system::governance_test_utils;\n  7 \u2502 \u2502 \n  8 \u2502 \u2502 use std::unit_test::{assert_eq, destroy};\n  9 \u2502 \u2502 use sui::balance::{Self, Balance};\n    \u00b7 \u2502\n408 \u2502 \u2502     sum\n409 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./tests/governance_test_utils.move:193:73\n    \u2502\n193 \u2502 public fun unstake(staker: address, staked_sui_idx: u64, scenario: &mut Scenario) {\n    \u2502                                                                         ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1057:9\n     \u2502\n1057 \u2502         assert_eq!(sum_rewards, expected_amount);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:312:9\n    \u2502\n312 \u2502         assert_eq!(validator.commission_rate(), 100);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:316:9\n    \u2502\n316 \u2502         assert_eq!(validator.commission_rate(), 101);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:326:9\n    \u2502\n326 \u2502         assert_eq!(validator.commission_rate(), 101);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1043:77\n     \u2502\n1043 \u2502 fun check_distribution_counter_invariant(system: &mut SuiSystemState, ctx: &TxContext) {\n     \u2502                                                                             ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1044:5\n     \u2502\n1044 \u2502     assert_eq!(ctx.epoch(), system.epoch());\n     \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1046:5\n     \u2502\n1046 \u2502     assert_eq!(system.get_stake_subsidy_distribution_counter() + 20, ctx.epoch());\n     \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:425:5\n    \u2502\n425 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), (550 + 150) * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:426:5\n    \u2502\n426 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:531:9\n    \u2502\n531 \u2502         assert_eq!(system.get_storage_fund_total_balance(), 4000 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:533:9\n    \u2502\n533 \u2502         assert_eq!(system.get_storage_fund_object_rebates(), 1000 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:578:9\n    \u2502\n578 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 140 * 23 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:591:5\n    \u2502\n591 \u2502     assert_eq!(\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:596:5\n    \u2502\n596 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), (480 + 40 * 2) * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:599:5\n    \u2502\n599 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_3).sui_balance(), (390 + 280 + 30) * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:601:5\n    \u2502\n601 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_4).sui_balance(), 1400 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:607:9\n    \u2502\n607 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 140 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:805:9\n    \u2502\n805 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:806:9\n    \u2502\n806 \u2502         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:816:9\n    \u2502\n816 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:817:9\n    \u2502\n817 \u2502         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:818:9\n    \u2502\n818 \u2502         assert_eq!(pool.pending_stake_amount(), 101 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:830:9\n    \u2502\n830 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:831:9\n    \u2502\n831 \u2502         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:832:9\n    \u2502\n832 \u2502         assert_eq!(pool.pending_stake_amount(), 101 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:833:9\n    \u2502\n833 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 101 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:834:9\n    \u2502\n834 \u2502         assert_eq!(pool.pending_pool_token_withdraw_amount(), 101 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:842:18\n    \u2502\n842 \u2502     let pool_id: ID;\n    \u2502                  ^^ Unbound type 'ID' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:848:9\n    \u2502\n848 \u2502         assert_eq!(pool.pending_stake_amount(), 0 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:849:9\n    \u2502\n849 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:850:9\n    \u2502\n850 \u2502         assert_eq!(pool.pending_pool_token_withdraw_amount(), 0 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:851:9\n    \u2502\n851 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:852:9\n    \u2502\n852 \u2502         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:868:9\n    \u2502\n868 \u2502         assert_eq!(pool.sui_balance(), 0 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:869:9\n    \u2502\n869 \u2502         assert_eq!(pool.pool_token_balance(), 0 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:893:9\n    \u2502\n893 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:894:9\n    \u2502\n894 \u2502         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:906:9\n    \u2502\n906 \u2502         assert_eq!(system.epoch(), 1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:909:9\n    \u2502\n909 \u2502         assert_eq!(pool.sui_balance(), 125 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:910:9\n    \u2502\n910 \u2502         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:937:9\n    \u2502\n937 \u2502         assert_eq!(exchange_rate.sui_amount(), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:941:9\n    \u2502\n941 \u2502         assert_eq!(exchange_rate.pool_token_amount(), 166666666666);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:942:9\n    \u2502\n942 \u2502         assert_eq!(pool.sui_balance(), 250 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:943:9\n    \u2502\n943 \u2502         assert_eq!(pool.pool_token_balance(), 166666666666);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:948:9\n    \u2502\n948 \u2502         assert_eq!(stake.stake_activation_epoch(), 3);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:960:13\n    \u2502\n960 \u2502             assert_eq!(pool.pending_pool_token_withdraw_amount(), 80 * MIST_PER_SUI);\n    \u2502             ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:963:13\n    \u2502\n963 \u2502             assert_eq!(pool.pending_stake_withdraw_amount(), 120 * MIST_PER_SUI); // 100 principal + 20 rewards\n    \u2502             ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:978:18\n    \u2502\n978 \u2502     let pool_id: ID;\n    \u2502                  ^^ Unbound type 'ID' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:984:9\n    \u2502\n984 \u2502         assert_eq!(pool.sui_balance(), 155 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:985:9\n    \u2502\n985 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 155 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:994:9\n    \u2502\n994 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 155 * MIST_PER_SUI); // 100 principal + 55 rewards\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:997:9\n    \u2502\n997 \u2502         assert_eq!(exchange_rate_epoch_0.sui_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:998:9\n    \u2502\n998 \u2502         assert_eq!(exchange_rate_epoch_0.pool_token_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1000:9\n     \u2502\n1000 \u2502         assert_eq!(exchange_rate_epoch_1.sui_amount(), 125000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1001:9\n     \u2502\n1001 \u2502         assert_eq!(exchange_rate_epoch_1.pool_token_amount(), 100000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1006:9\n     \u2502\n1006 \u2502         assert_eq!(exchange_rate_epoch_5.sui_amount(), 250000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1007:9\n     \u2502\n1007 \u2502         assert_eq!(exchange_rate_epoch_5.pool_token_amount(), 166666666666);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1009:9\n     \u2502\n1009 \u2502         assert_eq!(exchange_rate_epoch_6.sui_amount(), 155000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1010:9\n     \u2502\n1010 \u2502         assert_eq!(exchange_rate_epoch_6.pool_token_amount(), 86666666666);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1013:9\n     \u2502\n1013 \u2502         assert_eq!(pool.sui_balance(), 155000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1014:9\n     \u2502\n1014 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 155000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1015:9\n     \u2502\n1015 \u2502         assert_eq!(pool.pool_token_balance(), 86666666666);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1016:9\n     \u2502\n1016 \u2502         assert_eq!(pool.pending_pool_token_withdraw_amount(), 100000000000);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1032:9\n     \u2502\n1032 \u2502         assert_eq!(validator.pending_stake_amount(), 0);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1033:9\n     \u2502\n1033 \u2502         assert_eq!(validator.pending_stake_withdraw_amount(), 0);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1034:9\n     \u2502\n1034 \u2502         assert_eq!(validator.get_staking_pool_ref().pending_pool_token_withdraw_amount(), 0);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1035:9\n     \u2502\n1035 \u2502         assert_eq!(validator.get_staking_pool_ref().sui_balance(), 0);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:1036:9\n     \u2502\n1036 \u2502         assert_eq!(validator.get_staking_pool_ref().pool_token_balance(), 0);\n     \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:371:5\n    \u2502\n371 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), 565 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:372:5\n    \u2502\n372 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 370 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:483:5\n    \u2502\n483 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), (100 + 80) * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:485:5\n    \u2502\n485 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), (100 + 48) * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:113:9\n    \u2502\n113 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 300 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:114:9\n    \u2502\n114 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 300 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:115:9\n    \u2502\n115 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 300 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:116:9\n    \u2502\n116 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 400 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:150:5\n    \u2502\n150 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), 220 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:154:5\n    \u2502\n154 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 120 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:168:5\n    \u2502\n168 \u2502     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 728108108107);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:81:9\n   \u2502\n81 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 100_000_025 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:82:9\n   \u2502\n82 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 200_000_025 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:83:9\n   \u2502\n83 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 300_000_025 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:84:9\n   \u2502\n84 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 400_000_025 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:661:9\n    \u2502\n661 \u2502         assert_eq!(ctx.epoch(), 562);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:662:9\n    \u2502\n662 \u2502         assert_eq!(system.epoch(), 562);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:663:9\n    \u2502\n663 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:672:9\n    \u2502\n672 \u2502         assert_eq!(ctx.epoch(), 563);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:673:9\n    \u2502\n673 \u2502         assert_eq!(system.epoch(), 563);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:674:9\n    \u2502\n674 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 3);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:687:9\n    \u2502\n687 \u2502         assert_eq!(ctx.epoch(), 564);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:688:9\n    \u2502\n688 \u2502         assert_eq!(system.epoch(), 564);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:689:9\n    \u2502\n689 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:718:9\n    \u2502\n718 \u2502         assert_eq!(ctx.epoch(), 563);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:719:9\n    \u2502\n719 \u2502         assert_eq!(system.epoch(), 563);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:720:9\n    \u2502\n720 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:729:9\n    \u2502\n729 \u2502         assert_eq!(ctx.epoch(), 564);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:730:9\n    \u2502\n730 \u2502         assert_eq!(system.epoch(), 564);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:731:9\n    \u2502\n731 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:744:9\n    \u2502\n744 \u2502         assert_eq!(ctx.epoch(), 565);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:745:9\n    \u2502\n745 \u2502         assert_eq!(system.epoch(), 565);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:746:9\n    \u2502\n746 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 5);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:767:9\n    \u2502\n767 \u2502         assert_eq!(ctx.epoch(), 540);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:768:9\n    \u2502\n768 \u2502         assert_eq!(system.epoch(), 540);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:769:9\n    \u2502\n769 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), 540);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:778:9\n    \u2502\n778 \u2502         assert_eq!(ctx.epoch(), 541);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:779:9\n    \u2502\n779 \u2502         assert_eq!(system.epoch(), 541);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:780:9\n    \u2502\n780 \u2502         assert_eq!(system.get_stake_subsidy_distribution_counter(), 541);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:632:13\n    \u2502\n632 \u2502             assert_eq!(system.validator_stake_amount(addr), (962 + i * 4) * MIST_PER_SUI);\n    \u2502             ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound module\n     \u250c\u2500 ./tests/rewards_distribution_tests.move:5:1\n     \u2502  \n   5 \u2502 \u256d module sui_system::rewards_distribution_tests;\n   6 \u2502 \u2502 \n   7 \u2502 \u2502 use std::unit_test::assert_eq;\n   8 \u2502 \u2502 use sui::address;\n     \u00b7 \u2502\n1058 \u2502 \u2502     });\n1059 \u2502 \u2502 }\n     \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:233:9\n    \u2502\n233 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 200 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:234:9\n    \u2502\n234 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 300 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:235:9\n    \u2502\n235 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 300 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:236:9\n    \u2502\n236 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 400 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:244:9\n    \u2502\n244 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 230 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:245:9\n    \u2502\n245 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 330 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:246:9\n    \u2502\n246 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 330 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:247:9\n    \u2502\n247 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 430 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:272:9\n    \u2502\n272 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 290 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:273:9\n    \u2502\n273 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 390 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:274:9\n    \u2502\n274 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 390 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:275:9\n    \u2502\n275 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 490 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:43:9\n   \u2502\n43 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 125 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:44:9\n   \u2502\n44 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 225 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:45:9\n   \u2502\n45 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 325 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:46:9\n   \u2502\n46 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 425 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:56:9\n   \u2502\n56 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 150 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:57:9\n   \u2502\n57 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 970 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:58:9\n   \u2502\n58 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 350 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/rewards_distribution_tests.move:59:9\n   \u2502\n59 \u2502         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 450 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/stake_subsidy.move:16:14\n   \u2502\n16 \u2502     balance: Balance<SUI>,\n   \u2502              ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/stake_subsidy.move:28:19\n   \u2502\n28 \u2502     extra_fields: Bag,\n   \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/stake_subsidy.move:55:61\n   \u2502\n55 \u2502 public(package) fun advance_epoch(self: &mut StakeSubsidy): Balance<SUI> {\n   \u2502                                                             ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/stake_subsidy.move:32:14\n   \u2502\n32 \u2502     balance: Balance<SUI>,\n   \u2502              ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/stake_subsidy.move:36:15\n   \u2502\n36 \u2502     ctx: &mut TxContext,\n   \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n   \u250c\u2500 ./sources/stake_subsidy.move:4:1\n   \u2502  \n 4 \u2502 \u256d module sui_system::stake_subsidy;\n 5 \u2502 \u2502 \n 6 \u2502 \u2502 use sui::bag::{Self, Bag};\n 7 \u2502 \u2502 use sui::balance::Balance;\n   \u00b7 \u2502\n90 \u2502 \u2502     self.distribution_counter = distribution_counter;\n91 \u2502 \u2502 }\n   \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:90:9\n   \u2502\n90 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:92:14\n   \u2502\n92 \u2502     pool_id: ID,\n   \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:99:9\n   \u2502\n99 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:103:16\n    \u2502\n103 \u2502     principal: Balance<SUI>,\n    \u2502                ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:76:9\n   \u2502\n76 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:78:14\n   \u2502\n78 \u2502     pool_id: ID,\n   \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:82:16\n   \u2502\n82 \u2502     principal: Balance<SUI>,\n   \u2502                ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:42:23\n   \u2502\n42 \u2502     activation_epoch: Option<u64>,\n   \u2502                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:45:25\n   \u2502\n45 \u2502     deactivation_epoch: Option<u64>,\n   \u2502                         ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:56:21\n   \u2502\n56 \u2502     exchange_rates: Table<u64, PoolTokenExchangeRate>,\n   \u2502                     ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:65:19\n   \u2502\n65 \u2502     extra_fields: Bag,\n   \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:39:9\n   \u2502\n39 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/staking_pool.move:50:19\n   \u2502\n50 \u2502     rewards_pool: Balance<SUI>,\n   \u2502                   ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:500:59\n    \u2502\n500 \u2502 public(package) fun activation_epoch(pool: &StakingPool): Option<u64> {\n    \u2502                                                           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:277:15\n    \u2502\n277 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:745:15\n    \u2502\n745 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:350:70\n    \u2502\n350 \u2502 public(package) fun deposit_rewards(pool: &mut StakingPool, rewards: Balance<SUI>) {\n    \u2502                                                                      ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:630:58\n    \u2502\n630 \u2502 public(package) fun exchange_rates(pool: &StakingPool): &Table<u64, PoolTokenExchangeRate> {\n    \u2502                                                          ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:475:82\n    \u2502\n475 \u2502 public fun fungible_staked_sui_pool_id(fungible_staked_sui: &FungibleStakedSui): ID {\n    \u2502                                                                                  ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:116:35\n    \u2502\n116 \u2502 public(package) fun new(ctx: &mut TxContext): StakingPool {\n    \u2502                                   ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:471:45\n    \u2502\n471 \u2502 public fun pool_id(staked_sui: &StakedSui): ID { staked_sui.pool_id }\n    \u2502                                             ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:355:88\n    \u2502\n355 \u2502 public(package) fun process_pending_stakes_and_withdraws(pool: &mut StakingPool, ctx: &TxContext) {\n    \u2502                                                                                        ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:198:11\n    \u2502\n198 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:199:4\n    \u2502\n199 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:137:12\n    \u2502\n137 \u2502     stake: Balance<SUI>,\n    \u2502            ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:139:15\n    \u2502\n139 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:160:11\n    \u2502\n160 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:161:4\n    \u2502\n161 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:547:69\n    \u2502\n547 \u2502 public fun split(self: &mut StakedSui, split_amount: u64, ctx: &mut TxContext): StakedSui {\n    \u2502                                                                     ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:520:15\n    \u2502\n520 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:568:87\n    \u2502\n568 \u2502 public entry fun split_staked_sui(stake: &mut StakedSui, split_amount: u64, ctx: &mut TxContext) {\n    \u2502                                                                                       ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./sources/staking_pool.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::staking_pool;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use sui::bag::{Self, Bag};\n  8 \u2502 \u2502 use sui::balance::{Self, Balance};\n    \u00b7 \u2502\n836 \u2502 \u2502     assert!(principal_amount + rewards_amount >= min_out, 0);\n837 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:341:47\n    \u2502\n341 \u2502 fun unwrap_staked_sui(staked_sui: StakedSui): Balance<SUI> {\n    \u2502                                               ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:327:10\n    \u2502\n327 \u2502 ): (u64, Balance<SUI>) {\n    \u2502          ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/staking_pool.move:429:4\n    \u2502\n429 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:193:5\n    \u2502\n193 \u2502     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut scenario, 0), 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:196:5\n    \u2502\n196 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:197:5\n    \u2502\n197 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:206:5\n    \u2502\n206 \u2502     assert_eq!(\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:216:5\n    \u2502\n216 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 3_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:217:5\n    \u2502\n217 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:225:5\n    \u2502\n225 \u2502     assert_eq!(fungible_staked_sui_1.value(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:226:5\n    \u2502\n226 \u2502     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:229:5\n    \u2502\n229 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:230:5\n    \u2502\n230 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:236:5\n    \u2502\n236 \u2502     assert_eq!(fungible_staked_sui_2.value(), 500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:237:5\n    \u2502\n237 \u2502     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:240:5\n    \u2502\n240 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:241:5\n    \u2502\n241 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 2_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:243:5\n    \u2502\n243 \u2502     destroy(staking_pool);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:245:5\n    \u2502\n245 \u2502     destroy(fungible_staked_sui_1);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:246:5\n    \u2502\n246 \u2502     destroy(fungible_staked_sui_2);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/staking_pool_tests.move:452:20\n    \u2502\n452 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:28:5\n   \u2502\n28 \u2502     assert_eq!(fungible_staked_sui_1.value(), 300_000_000_000);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:30:5\n   \u2502\n30 \u2502     destroy(staking_pool);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:31:5\n   \u2502\n31 \u2502     destroy(fungible_staked_sui_1);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:291:5\n    \u2502\n291 \u2502     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut scenario, 0), 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:294:5\n    \u2502\n294 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:295:5\n    \u2502\n295 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:304:5\n    \u2502\n304 \u2502     assert_eq!(\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:314:5\n    \u2502\n314 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 3_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:315:5\n    \u2502\n315 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:321:5\n    \u2502\n321 \u2502     assert_eq!(fungible_staked_sui_1.value(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:322:5\n    \u2502\n322 \u2502     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:325:5\n    \u2502\n325 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:326:5\n    \u2502\n326 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:332:5\n    \u2502\n332 \u2502     assert_eq!(fungible_staked_sui_2.value(), 500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:333:5\n    \u2502\n333 \u2502     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:336:5\n    \u2502\n336 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:337:5\n    \u2502\n337 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 2_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:340:5\n    \u2502\n340 \u2502     assert_eq!(\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:350:5\n    \u2502\n350 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 6_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:351:5\n    \u2502\n351 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:353:5\n    \u2502\n353 \u2502     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:354:5\n    \u2502\n354 \u2502     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:357:5\n    \u2502\n357 \u2502     assert_eq!(sui_1.value(), 4_000_000_000 - 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:360:5\n    \u2502\n360 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:361:5\n    \u2502\n361 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 2_000_000_000 / 3 + 1); // round against user\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:363:5\n    \u2502\n363 \u2502     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 4_000_000_000 - 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:364:5\n    \u2502\n364 \u2502     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:367:5\n    \u2502\n367 \u2502     assert_eq!(sui_2.value(), 2_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:370:5\n    \u2502\n370 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:371:5\n    \u2502\n371 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:373:5\n    \u2502\n373 \u2502     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 6_000_000_000 - 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:374:5\n    \u2502\n374 \u2502     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:376:5\n    \u2502\n376 \u2502     destroy(staking_pool);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:377:5\n    \u2502\n377 \u2502     destroy(sui_1);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:378:5\n    \u2502\n378 \u2502     destroy(sui_2);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:398:5\n    \u2502\n398 \u2502     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut scenario, 0), 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:401:5\n    \u2502\n401 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:402:5\n    \u2502\n402 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:411:5\n    \u2502\n411 \u2502     assert_eq!(\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:421:5\n    \u2502\n421 \u2502     assert_eq!(latest_exchange_rate.sui_amount(), 3_000_000_001);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:422:5\n    \u2502\n422 \u2502     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:428:5\n    \u2502\n428 \u2502     assert_eq!(fungible_staked_sui.value(), 500_000_000); // rounding!\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:429:5\n    \u2502\n429 \u2502     assert_eq!(fungible_staked_sui.pool_id(), object::id(&staking_pool));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:432:5\n    \u2502\n432 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 500_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:433:5\n    \u2502\n433 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 1_000_000_001);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:437:5\n    \u2502\n437 \u2502     assert_eq!(sui.value(), 1_000_000_000);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:440:5\n    \u2502\n440 \u2502     assert_eq!(fungible_staked_sui_data.total_supply(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:441:5\n    \u2502\n441 \u2502     assert_eq!(fungible_staked_sui_data.principal_value(), 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:443:5\n    \u2502\n443 \u2502     destroy(staking_pool);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:444:5\n    \u2502\n444 \u2502     destroy(staked_sui_1);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:445:5\n    \u2502\n445 \u2502     destroy(sui);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:68:5\n   \u2502\n68 \u2502     assert_eq!(fungible_staked_sui_1.value(), 25_000_000_000);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:69:5\n   \u2502\n69 \u2502     assert_eq!(fungible_staked_sui_2.value(), 75_000_000_000);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:71:5\n   \u2502\n71 \u2502     destroy(staking_pool);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:72:5\n   \u2502\n72 \u2502     destroy(fungible_staked_sui_1);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/staking_pool_tests.move:73:5\n   \u2502\n73 \u2502     destroy(fungible_staked_sui_2);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:259:5\n    \u2502\n259 \u2502     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut test, 0), 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:266:5\n    \u2502\n266 \u2502     assert_eq!(staking_pool.sui_balance(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:267:5\n    \u2502\n267 \u2502     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:268:5\n    \u2502\n268 \u2502     assert_eq!(staking_pool.pool_token_balance(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:269:5\n    \u2502\n269 \u2502     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 0);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:271:5\n    \u2502\n271 \u2502     destroy(staking_pool);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/staking_pool_tests.move:272:5\n    \u2502\n272 \u2502     destroy(staked_sui_1);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/staking_pool_tests.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::staking_pool_tests;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use std::unit_test::{assert_eq, destroy};\n  8 \u2502 \u2502 use sui::balance;\n    \u00b7 \u2502\n465 \u2502 \u2502     scenario.ctx().epoch()\n466 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:19:29\n   \u2502\n19 \u2502     non_refundable_balance: Balance<SUI>,\n   \u2502                             ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:18:35\n   \u2502\n18 \u2502     total_object_storage_rebates: Balance<SUI>,\n   \u2502                                   ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:34:22\n   \u2502\n34 \u2502     storage_charges: Balance<SUI>,\n   \u2502                      ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:35:32\n   \u2502\n35 \u2502     storage_fund_reinvestment: Balance<SUI>,\n   \u2502                                ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:36:31\n   \u2502\n36 \u2502     leftover_staking_rewards: Balance<SUI>,\n   \u2502                               ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:39:4\n   \u2502\n39 \u2502 ): Balance<SUI> {\n   \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/storage_fund.move:23:39\n   \u2502\n23 \u2502 public(package) fun new(initial_fund: Balance<SUI>): StorageFund {\n   \u2502                                       ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound module\n   \u250c\u2500 ./sources/storage_fund.move:4:1\n   \u2502  \n 4 \u2502 \u256d module sui_system::storage_fund;\n 5 \u2502 \u2502 \n 6 \u2502 \u2502 use sui::balance::{Self, Balance};\n 7 \u2502 \u2502 use sui::sui::SUI;\n   \u00b7 \u2502\n70 \u2502 \u2502     self.total_object_storage_rebates.value() + self.non_refundable_balance.value()\n71 \u2502 \u2502 }\n   \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n   \u250c\u2500 ./sources/sui_system.move:67:9\n   \u2502\n67 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:556:70\n    \u2502\n556 \u2502 public fun active_validator_voting_powers(wrapper: &SuiSystemState): VecMap<address, u64> {\n    \u2502                                                                      ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:590:21\n    \u2502\n590 \u2502     storage_reward: Balance<SUI>,\n    \u2502                     ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:591:25\n    \u2502\n591 \u2502     computation_reward: Balance<SUI>,\n    \u2502                         ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:601:15\n    \u2502\n601 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:602:4\n    \u2502\n602 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:890:15\n    \u2502\n890 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:891:4\n    \u2502\n891 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:570:11\n    \u2502\n570 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:280:15\n    \u2502\n280 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/sui_system.move:79:9\n   \u2502\n79 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/sui_system.move:81:19\n   \u2502\n81 \u2502     storage_fund: Balance<SUI>,\n   \u2502                   ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/sui_system.move:86:15\n   \u2502\n86 \u2502     ctx: &mut TxContext,\n   \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:756:75\n    \u2502\n756 \u2502 public fun get_reporters_of(wrapper: &mut SuiSystemState, addr: address): VecSet<address> {\n    \u2502                                                                           ^^^^^^ Unbound type 'VecSet' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:540:15\n    \u2502\n540 \u2502     pool_id: &ID,\n    \u2502               ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:541:5\n    \u2502\n541 \u2502 ): &Table<u64, PoolTokenExchangeRate> {\n    \u2502     ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:289:11\n    \u2502\n289 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:290:4\n    \u2502\n290 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:231:12\n    \u2502\n231 \u2502     stake: Coin<SUI>,\n    \u2502            ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:233:15\n    \u2502\n233 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:253:20\n    \u2502\n253 \u2502     stakes: vector<Coin<SUI>>,\n    \u2502                    ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:256:15\n    \u2502\n256 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:242:12\n    \u2502\n242 \u2502     stake: Coin<SUI>,\n    \u2502            ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:244:15\n    \u2502\n244 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:169:80\n    \u2502\n169 \u2502 public entry fun request_add_validator(wrapper: &mut SuiSystemState, ctx: &mut TxContext) {\n    \u2502                                                                                ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:131:15\n    \u2502\n131 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:854:15\n    \u2502\n854 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:805:82\n    \u2502\n805 \u2502 public fun request_add_validator_for_testing(wrapper: &mut SuiSystemState, ctx: &TxContext) {\n    \u2502                                                                                  ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:179:83\n    \u2502\n179 \u2502 public entry fun request_remove_validator(wrapper: &mut SuiSystemState, ctx: &mut TxContext) {\n    \u2502                                                                                   ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:159:15\n    \u2502\n159 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:210:15\n    \u2502\n210 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:270:15\n    \u2502\n270 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:298:15\n    \u2502\n298 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:299:4\n    \u2502\n299 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:336:76\n    \u2502\n336 \u2502 public entry fun rotate_operation_cap(self: &mut SuiSystemState, ctx: &mut TxContext) {\n    \u2502                                                                            ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:220:15\n    \u2502\n220 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./sources/sui_system.move:40:1\n    \u2502  \n 40 \u2502 \u256d module sui_system::sui_system;\n 41 \u2502 \u2502 \n 42 \u2502 \u2502 use sui::balance::Balance;\n 43 \u2502 \u2502 use sui::coin::Coin;\n    \u00b7 \u2502\n907 \u2502 \u2502     storage_rebate\n908 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:396:11\n    \u2502\n396 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:528:11\n    \u2502\n528 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:417:11\n    \u2502\n417 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:438:11\n    \u2502\n438 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:484:11\n    \u2502\n484 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:459:11\n    \u2502\n459 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:507:11\n    \u2502\n507 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:355:11\n    \u2502\n355 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:365:11\n    \u2502\n365 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:345:11\n    \u2502\n345 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:386:11\n    \u2502\n386 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:518:11\n    \u2502\n518 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:407:11\n    \u2502\n407 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:428:11\n    \u2502\n428 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:471:11\n    \u2502\n471 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:449:11\n    \u2502\n449 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:497:11\n    \u2502\n497 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:375:11\n    \u2502\n375 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:533:81\n    \u2502\n533 \u2502 public fun validator_address_by_pool_id(wrapper: &mut SuiSystemState, pool_id: &ID): address {\n    \u2502                                                                                 ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:744:94\n    \u2502\n744 \u2502 public fun validator_staking_pool_id(wrapper: &mut SuiSystemState, validator_addr: address): ID {\n    \u2502                                                                                              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:750:76\n    \u2502\n750 \u2502 public fun validator_staking_pool_mappings(wrapper: &mut SuiSystemState): &Table<ID, address> {\n    \u2502                                                                            ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:660:60\n    \u2502\n660 \u2502 fun validator_voting_powers(wrapper: &mut SuiSystemState): VecMap<address, u64> {\n    \u2502                                                            ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:717:79\n    \u2502\n717 \u2502 public fun validator_voting_powers_for_testing(wrapper: &mut SuiSystemState): VecMap<address, u64> {\n    \u2502                                                                               ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system.move:703:11\n    \u2502\n703 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:149:19\n    \u2502\n149 \u2502     extra_fields: Bag,\n    \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:143:36\n    \u2502\n143 \u2502     safe_mode_computation_rewards: Balance<SUI>,\n    \u2502                                    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:142:32\n    \u2502\n142 \u2502     safe_mode_storage_rewards: Balance<SUI>,\n    \u2502                                ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:132:31\n    \u2502\n132 \u2502     validator_report_records: VecMap<address, VecSet<address>>,\n    \u2502                               ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:195:19\n    \u2502\n195 \u2502     extra_fields: Bag,\n    \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:189:36\n    \u2502\n189 \u2502     safe_mode_computation_rewards: Balance<SUI>,\n    \u2502                                    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:188:32\n    \u2502\n188 \u2502     safe_mode_storage_rewards: Balance<SUI>,\n    \u2502                                ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:178:31\n    \u2502\n178 \u2502     validator_report_records: VecMap<address, VecSet<address>>,\n    \u2502                               ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/sui_system_state_inner.move:73:19\n   \u2502\n73 \u2502     extra_fields: Bag,\n   \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:103:19\n    \u2502\n103 \u2502     extra_fields: Bag,\n    \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1066:4\n     \u2502\n1066 \u2502 ): VecMap<address, u64> {\n     \u2502    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:862:25\n    \u2502\n862 \u2502     mut storage_reward: Balance<SUI>,\n    \u2502                         ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:863:29\n    \u2502\n863 \u2502     mut computation_reward: Balance<SUI>,\n    \u2502                             ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:872:15\n    \u2502\n872 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:873:4\n    \u2502\n873 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:513:15\n    \u2502\n513 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:221:27\n    \u2502\n221 \u2502     initial_storage_fund: Balance<SUI>,\n    \u2502                           ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:226:15\n    \u2502\n226 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:261:15\n    \u2502\n261 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1206:66\n     \u2502\n1206 \u2502 public(package) fun extra_fields(self: &SuiSystemStateInnerV2): &Bag {\n     \u2502                                                                  ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1210:78\n     \u2502\n1210 \u2502 public(package) fun extra_fields_mut(self: &mut SuiSystemStateInnerV2): &mut Bag {\n     \u2502                                                                              ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1127:23\n     \u2502\n1127 \u2502     mut coins: vector<Coin<SUI>>,\n     \u2502                       ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1128:13\n     \u2502\n1128 \u2502     amount: Option<u64>,\n     \u2502             ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1129:15\n     \u2502\n1129 \u2502     ctx: &mut TxContext,\n     \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1130:4\n     \u2502\n1130 \u2502 ): Balance<SUI> {\n     \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1093:84\n     \u2502\n1093 \u2502 public(package) fun get_reporters_of(self: &SuiSystemStateInnerV2, addr: address): VecSet<address> {\n     \u2502                                                                                    ^^^^^^ Unbound type 'VecSet' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1115:14\n     \u2502\n1115 \u2502     pool_id: ID,\n     \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1116:5\n     \u2502\n1116 \u2502 ): &Table<u64, PoolTokenExchangeRate> {\n     \u2502     ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:521:11\n    \u2502\n521 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:522:4\n    \u2502\n522 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:562:36\n    \u2502\n562 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:480:12\n    \u2502\n480 \u2502     stake: Coin<SUI>,\n    \u2502            ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:482:15\n    \u2502\n482 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:490:20\n    \u2502\n490 \u2502     stakes: vector<Coin<SUI>>,\n    \u2502                    ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:491:19\n    \u2502\n491 \u2502     stake_amount: Option<u64>,\n    \u2502                   ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:493:15\n    \u2502\n493 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:395:83\n    \u2502\n395 \u2502 public(package) fun request_add_validator(self: &mut SuiSystemStateInnerV2, ctx: &TxContext) {\n    \u2502                                                                                   ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:358:15\n    \u2502\n358 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1283:15\n     \u2502\n1283 \u2502     ctx: &mut TxContext,\n     \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:409:86\n    \u2502\n409 \u2502 public(package) fun request_remove_validator(self: &mut SuiSystemStateInnerV2, ctx: &TxContext) {\n    \u2502                                                                                      ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:386:15\n    \u2502\n386 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:457:11\n    \u2502\n457 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:503:11\n    \u2502\n503 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:504:4\n    \u2502\n504 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:597:86\n    \u2502\n597 \u2502 public(package) fun rotate_operation_cap(self: &mut SuiSystemStateInnerV2, ctx: &mut TxContext) {\n    \u2502                                                                                      ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:469:11\n    \u2502\n469 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:579:36\n    \u2502\n579 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound module\n     \u250c\u2500 ./sources/sui_system_state_inner.move:4:1\n     \u2502  \n   4 \u2502 \u256d module sui_system::sui_system_state_inner;\n   5 \u2502 \u2502 \n   6 \u2502 \u2502 use sui::bag::{Self, Bag};\n   7 \u2502 \u2502 use sui::balance::{Self, Balance};\n     \u00b7 \u2502\n1310 \u2502 \u2502     (($a as u128) * ($b as u128) / ($c as u128)) as u64\n1311 \u2502 \u2502 }\n     \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:676:11\n    \u2502\n676 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:843:11\n    \u2502\n843 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:704:11\n    \u2502\n704 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:729:11\n    \u2502\n729 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:787:11\n    \u2502\n787 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:757:11\n    \u2502\n757 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:815:11\n    \u2502\n815 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:629:11\n    \u2502\n629 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:639:11\n    \u2502\n639 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:613:11\n    \u2502\n613 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:663:11\n    \u2502\n663 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:830:11\n    \u2502\n830 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:691:11\n    \u2502\n691 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:716:11\n    \u2502\n716 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:773:11\n    \u2502\n773 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:744:11\n    \u2502\n744 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:802:11\n    \u2502\n802 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/sui_system_state_inner.move:649:11\n    \u2502\n649 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1108:15\n     \u2502\n1108 \u2502     pool_id: &ID,\n     \u2502               ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1081:4\n     \u2502\n1081 \u2502 ): ID {\n     \u2502    ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/sui_system_state_inner.move:1088:5\n     \u2502\n1088 \u2502 ): &Table<ID, address> {\n     \u2502     ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:502:9\n    \u2502\n502 \u2502         assert_eq!(pool.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:503:9\n    \u2502\n503 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:504:9\n    \u2502\n504 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:509:5\n    \u2502\n509 \u2502     assert_eq!(staked_sui.amount(), stake_amount * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:515:9\n    \u2502\n515 \u2502         assert_eq!(pool.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:516:9\n    \u2502\n516 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:517:9\n    \u2502\n517 \u2502         assert_eq!(pool.sui_balance(), (100 + stake_amount) * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:526:5\n    \u2502\n526 \u2502     assert_eq!(fungible_staked_sui.value(), stake_amount * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:533:5\n    \u2502\n533 \u2502     assert_eq!(sui.destroy_for_testing(), stake_amount * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:538:9\n    \u2502\n538 \u2502         assert_eq!(pool.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:539:9\n    \u2502\n539 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:540:9\n    \u2502\n540 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:37:9\n   \u2502\n37 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:43:9\n   \u2502\n43 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1, @3])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:49:9\n   \u2502\n49 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1, @3])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:55:9\n   \u2502\n55 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:62:9\n   \u2502\n62 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:68:9\n   \u2502\n68 \u2502         assert_eq!(system.get_reporters_of(@1).into_keys(), vector[@2])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:74:9\n   \u2502\n74 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1, @3])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/sui_system_tests.move:81:9\n   \u2502\n81 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:122:9\n    \u2502\n122 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1]);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:141:9\n    \u2502\n141 \u2502         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1]);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:159:9\n    \u2502\n159 \u2502         assert_eq!(system.active_validator_by_address(@1).next_epoch_gas_price(), 666);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:160:9\n    \u2502\n160 \u2502         assert_eq!(system.pending_validator_by_address(new_validator).next_epoch_gas_price(), 777);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:167:9\n    \u2502\n167 \u2502         assert_eq!(system.active_validator_by_address(@1).gas_price(), 666);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:168:9\n    \u2502\n168 \u2502         assert_eq!(system.active_validator_by_address(new_validator).gas_price(), 1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:175:9\n    \u2502\n175 \u2502         assert_eq!(system.active_validator_by_address(new_validator).gas_price(), 777);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:424:9\n    \u2502\n424 \u2502         assert_eq!(counter, 1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:434:9\n    \u2502\n434 \u2502         assert_eq!(counter, 1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:444:9\n    \u2502\n444 \u2502         assert_eq!(counter, 2);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:344:9\n    \u2502\n344 \u2502         assert_eq!(pool_mappings.length(), 4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:345:9\n    \u2502\n345 \u2502         assert_eq!(pool_mappings[pool_id_1], @1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:346:9\n    \u2502\n346 \u2502         assert_eq!(pool_mappings[pool_id_2], @2);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:347:9\n    \u2502\n347 \u2502         assert_eq!(pool_mappings[pool_id_3], @3);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:348:9\n    \u2502\n348 \u2502         assert_eq!(pool_mappings[pool_id_4], @4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:371:9\n    \u2502\n371 \u2502         assert_eq!(pool_mappings.length(), 5);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:372:9\n    \u2502\n372 \u2502         assert_eq!(pool_mappings[pool_id_1], @1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:373:9\n    \u2502\n373 \u2502         assert_eq!(pool_mappings[pool_id_2], @2);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:374:9\n    \u2502\n374 \u2502         assert_eq!(pool_mappings[pool_id_3], @3);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:375:9\n    \u2502\n375 \u2502         assert_eq!(pool_mappings[pool_id_4], @4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:376:9\n    \u2502\n376 \u2502         assert_eq!(pool_mappings[pool_id_5], new_validator);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:392:9\n    \u2502\n392 \u2502         assert_eq!(pool_mappings.length(), 4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:393:9\n    \u2502\n393 \u2502         assert_eq!(pool_mappings[pool_id_2], @2);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:394:9\n    \u2502\n394 \u2502         assert_eq!(pool_mappings[pool_id_3], @3);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:395:9\n    \u2502\n395 \u2502         assert_eq!(pool_mappings[pool_id_4], @4);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:396:9\n    \u2502\n396 \u2502         assert_eq!(pool_mappings[pool_id_5], new_validator);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/sui_system_tests.move:9:1\n    \u2502  \n  9 \u2502 \u256d module sui_system::sui_system_tests;\n 10 \u2502 \u2502 \n 11 \u2502 \u2502 use std::unit_test::assert_eq;\n 12 \u2502 \u2502 use sui_system::test_runner;\n    \u00b7 \u2502\n543 \u2502 \u2502     runner.finish();\n544 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:319:9\n    \u2502\n319 \u2502         assert_eq!(system.validator_address_by_pool_id(&pool_id), @1);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:460:9\n    \u2502\n460 \u2502         assert_eq!(pool.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:461:9\n    \u2502\n461 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:462:9\n    \u2502\n462 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:471:9\n    \u2502\n471 \u2502         assert_eq!(pool.pending_stake_amount(), stake_amount * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:472:9\n    \u2502\n472 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:473:9\n    \u2502\n473 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:482:9\n    \u2502\n482 \u2502         assert_eq!(pool.pending_stake_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:483:9\n    \u2502\n483 \u2502         assert_eq!(pool.pending_stake_withdraw_amount(), 0);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/sui_system_tests.move:484:9\n    \u2502\n484 \u2502         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);\n    \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:221:25\n    \u2502\n221 \u2502     computation_charge: Option<u64>,\n    \u2502                         ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:226:23\n    \u2502\n226 \u2502     epoch_start_time: Option<u64>,\n    \u2502                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:223:33\n    \u2502\n223 \u2502     non_refundable_storage_fee: Option<u64>,\n    \u2502                                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:219:23\n    \u2502\n219 \u2502     protocol_version: Option<u64>,\n    \u2502                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:225:27\n    \u2502\n225 \u2502     reward_slashing_rate: Option<u64>,\n    \u2502                           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:220:21\n    \u2502\n220 \u2502     storage_charge: Option<u64>,\n    \u2502                     ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:224:33\n    \u2502\n224 \u2502     storage_fund_reinvest_rate: Option<u64>,\n    \u2502                                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:222:21\n    \u2502\n222 \u2502     storage_rebate: Option<u64>,\n    \u2502                     ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:306:15\n    \u2502\n306 \u2502     scenario: Scenario,\n    \u2502               ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:35:21\n   \u2502\n35 \u2502     epoch_duration: Option<u64>,\n   \u2502                     ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:36:29\n   \u2502\n36 \u2502     low_stake_grace_period: Option<u64>,\n   \u2502                             ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:32:23\n   \u2502\n32 \u2502     protocol_version: Option<u64>,\n   \u2502                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:33:33\n   \u2502\n33 \u2502     stake_distribution_counter: Option<u64>,\n   \u2502                                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:34:18\n   \u2502\n34 \u2502     start_epoch: Option<u64>,\n   \u2502                  ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:28:26\n   \u2502\n28 \u2502     storage_fund_amount: Option<u64>,\n   \u2502                          ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:27:24\n   \u2502\n27 \u2502     sui_supply_amount: Option<u64>,\n   \u2502                        ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:26:17\n   \u2502\n26 \u2502     validators: Option<vector<ValidatorBuilder>>,\n   \u2502                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:30:23\n   \u2502\n30 \u2502     validators_count: Option<u64>,\n   \u2502                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/test_runner.move:31:31\n   \u2502\n31 \u2502     validators_initial_stake: Option<u64>,\n   \u2502                               ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:409:14\n    \u2502\n409 \u2502     options: Option<AdvanceEpochOptions>,\n    \u2502              ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:410:4\n    \u2502\n410 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:319:47\n    \u2502\n319 \u2502 public fun ctx(runner: &mut TestRunner): &mut TxContext { runner.scenario.ctx() }\n    \u2502                                               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/builders/test_runner.move:344:5\n    \u2502\n344 \u2502     std::unit_test::destroy(v);\n    \u2502     ^^^^^^^^^^^^^^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:338:31\n    \u2502\n338 \u2502 public fun mint(amount: u64): Balance<SUI> {\n    \u2502                               ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:356:66\n    \u2502\n356 \u2502 public macro fun scenario_fn($runner: &mut TestRunner, $f: |&mut Scenario|) {\n    \u2502                                                                  ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:322:56\n    \u2502\n322 \u2502 public fun scenario_mut(runner: &mut TestRunner): &mut Scenario { &mut runner.scenario }\n    \u2502                                                        ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:575:29\n    \u2502\n575 \u2502     scenario.ids_for_sender<Coin<SUI>>().fold!(0, |mut sum, coin_id| {\n    \u2502                             ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:576:52\n    \u2502\n576 \u2502         let coin = scenario.take_from_sender_by_id<Coin<SUI>>(coin_id);\n    \u2502                                                    ^^^^ Unbound type 'Coin' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/test_runner.move:374:36\n    \u2502\n374 \u2502     $f: |&mut SuiSystemState, &mut TxContext|,\n    \u2502                                    ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/builders/test_runner.move:7:1\n    \u2502  \n  7 \u2502 \u256d module sui_system::test_runner;\n  8 \u2502 \u2502 \n  9 \u2502 \u2502 use sui::balance::{Self, Balance};\n 10 \u2502 \u2502 use sui::coin::{Self, Coin};\n    \u00b7 \u2502\n617 \u2502 \u2502     runner.finish();\n618 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:156:14\n    \u2502\n156 \u2502     pool_id: ID,\n    \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:164:14\n    \u2502\n164 \u2502     pool_id: ID,\n    \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:136:14\n    \u2502\n136 \u2502     pool_id: ID,\n    \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:145:14\n    \u2502\n145 \u2502     pool_id: ID,\n    \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:131:19\n    \u2502\n131 \u2502     extra_fields: Bag,\n    \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:117:23\n    \u2502\n117 \u2502     operation_cap_id: ID,\n    \u2502                       ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:85:18\n   \u2502\n85 \u2502     description: String,\n   \u2502                  ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:107:19\n    \u2502\n107 \u2502     extra_fields: Bag,\n    \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:86:16\n   \u2502\n86 \u2502     image_url: Url,\n   \u2502                ^^^ Unbound type 'Url' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:84:11\n   \u2502\n84 \u2502     name: String,\n   \u2502           ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:89:18\n   \u2502\n89 \u2502     net_address: String,\n   \u2502                  ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:102:29\n    \u2502\n102 \u2502     next_epoch_net_address: Option<String>,\n    \u2502                             ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:100:38\n    \u2502\n100 \u2502     next_epoch_network_pubkey_bytes: Option<vector<u8>>,\n    \u2502                                      ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:103:29\n    \u2502\n103 \u2502     next_epoch_p2p_address: Option<String>,\n    \u2502                             ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:104:33\n    \u2502\n104 \u2502     next_epoch_primary_address: Option<String>,\n    \u2502                                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:99:37\n   \u2502\n99 \u2502     next_epoch_proof_of_possession: Option<vector<u8>>,\n   \u2502                                     ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:98:39\n   \u2502\n98 \u2502     next_epoch_protocol_pubkey_bytes: Option<vector<u8>>,\n   \u2502                                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:105:32\n    \u2502\n105 \u2502     next_epoch_worker_address: Option<String>,\n    \u2502                                ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:101:37\n    \u2502\n101 \u2502     next_epoch_worker_pubkey_bytes: Option<vector<u8>>,\n    \u2502                                     ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:91:18\n   \u2502\n91 \u2502     p2p_address: String,\n   \u2502                  ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:93:22\n   \u2502\n93 \u2502     primary_address: String,\n   \u2502                      ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:87:18\n   \u2502\n87 \u2502     project_url: Url,\n   \u2502                  ^^^ Unbound type 'Url' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator.move:95:21\n   \u2502\n95 \u2502     worker_address: String,\n   \u2502                     ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:642:39\n    \u2502\n642 \u2502 macro fun both_some_and_equal<$T>($a: Option<$T>, $b: Option<$T>): bool {\n    \u2502                                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:642:55\n    \u2502\n642 \u2502 macro fun both_some_and_equal<$T>($a: Option<$T>, $b: Option<$T>): bool {\n    \u2502                                                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:310:15\n    \u2502\n310 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:432:73\n    \u2502\n432 \u2502 public(package) fun deposit_stake_rewards(self: &mut Validator, reward: Balance<SUI>) {\n    \u2502                                                                         ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:461:44\n    \u2502\n461 \u2502 public fun description(self: &Validator): &String {\n    \u2502                                            ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:896:35\n    \u2502\n896 \u2502 macro fun do_extract<$T>($o: &mut Option<$T>, $f: |$T|) {\n    \u2502                                   ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:465:42\n    \u2502\n465 \u2502 public fun image_url(self: &Validator): &Url {\n    \u2502                                          ^^^ Unbound type 'Url' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:457:37\n    \u2502\n457 \u2502 public fun name(self: &Validator): &String {\n    \u2502                                     ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:473:48\n    \u2502\n473 \u2502 public fun network_address(self: &Validator): &String {\n    \u2502                                                ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:227:15\n    \u2502\n227 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:967:27\n    \u2502\n967 \u2502     initial_stake_option: Option<Balance<SUI>>,\n    \u2502                           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:971:15\n    \u2502\n971 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:921:15\n    \u2502\n921 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:175:11\n    \u2502\n175 \u2502     name: String,\n    \u2502           ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:176:18\n    \u2502\n176 \u2502     description: String,\n    \u2502                  ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:177:16\n    \u2502\n177 \u2502     image_url: Url,\n    \u2502                ^^^ Unbound type 'Url' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:178:18\n    \u2502\n178 \u2502     project_url: Url,\n    \u2502                  ^^^ Unbound type 'Url' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:179:18\n    \u2502\n179 \u2502     net_address: String,\n    \u2502                  ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:180:18\n    \u2502\n180 \u2502     p2p_address: String,\n    \u2502                  ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:181:22\n    \u2502\n181 \u2502     primary_address: String,\n    \u2502                      ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:182:21\n    \u2502\n182 \u2502     worker_address: String,\n    \u2502                     ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:183:19\n    \u2502\n183 \u2502     extra_fields: Bag,\n    \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:653:15\n    \u2502\n653 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:505:59\n    \u2502\n505 \u2502 public fun next_epoch_network_address(self: &Validator): &Option<String> {\n    \u2502                                                           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:529:64\n    \u2502\n529 \u2502 public fun next_epoch_network_pubkey_bytes(self: &Validator): &Option<vector<u8>> {\n    \u2502                                                                ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:509:55\n    \u2502\n509 \u2502 public fun next_epoch_p2p_address(self: &Validator): &Option<String> {\n    \u2502                                                       ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:513:59\n    \u2502\n513 \u2502 public fun next_epoch_primary_address(self: &Validator): &Option<String> {\n    \u2502                                                           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:525:63\n    \u2502\n525 \u2502 public fun next_epoch_proof_of_possession(self: &Validator): &Option<vector<u8>> {\n    \u2502                                                               ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:521:65\n    \u2502\n521 \u2502 public fun next_epoch_protocol_pubkey_bytes(self: &Validator): &Option<vector<u8>> {\n    \u2502                                                                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:517:58\n    \u2502\n517 \u2502 public fun next_epoch_worker_address(self: &Validator): &Option<String> {\n    \u2502                                                          ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:533:63\n    \u2502\n533 \u2502 public fun next_epoch_worker_pubkey_bytes(self: &Validator): &Option<vector<u8>> {\n    \u2502                                                               ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:537:49\n    \u2502\n537 \u2502 public fun operation_cap_id(self: &Validator): &ID {\n    \u2502                                                 ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:477:44\n    \u2502\n477 \u2502 public fun p2p_address(self: &Validator): &String {\n    \u2502                                            ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:481:48\n    \u2502\n481 \u2502 public fun primary_address(self: &Validator): &String {\n    \u2502                                                ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:438:86\n    \u2502\n438 \u2502 public(package) fun process_pending_stakes_and_withdraws(self: &mut Validator, ctx: &TxContext) {\n    \u2502                                                                                      ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:469:44\n    \u2502\n469 \u2502 public fun project_url(self: &Validator): &Url {\n    \u2502                                            ^^^ Unbound type 'Url' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:329:11\n    \u2502\n329 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:330:4\n    \u2502\n330 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:284:12\n    \u2502\n284 \u2502     stake: Balance<SUI>,\n    \u2502            ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:286:15\n    \u2502\n286 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:348:12\n    \u2502\n348 \u2502     stake: Balance<SUI>,\n    \u2502            ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:350:15\n    \u2502\n350 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:370:11\n    \u2502\n370 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:371:4\n    \u2502\n371 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:590:47\n    \u2502\n590 \u2502 public fun staking_pool_id(self: &Validator): ID {\n    \u2502                                               ^^ Unbound type 'ID' in current scope\n\nerror: unbound module\n     \u250c\u2500 ./sources/validator.move:5:1\n     \u2502  \n   5 \u2502 \u256d module sui_system::validator;\n   6 \u2502 \u2502 \n   7 \u2502 \u2502 use std::bcs;\n   8 \u2502 \u2502 use std::string::String;\n     \u00b7 \u2502\n1009 \u2502 \u2502     validator\n1010 \u2502 \u2502 }\n     \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator.move:485:47\n    \u2502\n485 \u2502 public fun worker_address(self: &Validator): &String {\n    \u2502                                               ^^^^^^ Unbound type 'String' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:36:22\n   \u2502\n36 \u2502     commission_rate: Option<u64>,\n   \u2502                      ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:28:18\n   \u2502\n28 \u2502     description: Option<vector<u8>>,\n   \u2502                  ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:35:16\n   \u2502\n35 \u2502     gas_price: Option<u64>,\n   \u2502                ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:29:16\n   \u2502\n29 \u2502     image_url: Option<vector<u8>>,\n   \u2502                ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:38:20\n   \u2502\n38 \u2502     initial_stake: Option<u64>,\n   \u2502                    ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:27:11\n   \u2502\n27 \u2502     name: Option<vector<u8>>,\n   \u2502           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:31:18\n   \u2502\n31 \u2502     net_address: Option<vector<u8>>,\n   \u2502                  ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:24:27\n   \u2502\n24 \u2502     network_pubkey_bytes: Option<vector<u8>>,\n   \u2502                           ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:32:18\n   \u2502\n32 \u2502     p2p_address: Option<vector<u8>>,\n   \u2502                  ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:33:22\n   \u2502\n33 \u2502     primary_address: Option<vector<u8>>,\n   \u2502                      ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:30:18\n   \u2502\n30 \u2502     project_url: Option<vector<u8>>,\n   \u2502                  ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:26:26\n   \u2502\n26 \u2502     proof_of_possession: Option<vector<u8>>,\n   \u2502                          ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:23:28\n   \u2502\n23 \u2502     protocol_pubkey_bytes: Option<vector<u8>>,\n   \u2502                            ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:22:18\n   \u2502\n22 \u2502     sui_address: Option<address>,\n   \u2502                  ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:34:21\n   \u2502\n34 \u2502     worker_address: Option<vector<u8>>,\n   \u2502                     ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:25:26\n   \u2502\n25 \u2502     worker_pubkey_bytes: Option<vector<u8>>,\n   \u2502                          ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./tests/builders/validator_builder.move:92:55\n   \u2502\n92 \u2502 public fun build(builder: ValidatorBuilder, ctx: &mut TxContext): Validator {\n   \u2502                                                       ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/builders/validator_builder.move:135:64\n    \u2502\n135 \u2502 public fun build_metadata(builder: ValidatorBuilder, ctx: &mut TxContext): ValidatorMetadata {\n    \u2502                                                                ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/builders/validator_builder.move:10:1\n    \u2502  \n 10 \u2502 \u256d module sui_system::validator_builder;\n 11 \u2502 \u2502 \n 12 \u2502 \u2502 use sui::bag;\n 13 \u2502 \u2502 use sui::balance;\n    \u00b7 \u2502\n340 \u2502 \u2502     preset.protocol_pubkey_bytes()\n341 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_cap.move:18:9\n   \u2502\n18 \u2502     id: UID,\n   \u2502         ^^^ Unbound type 'UID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_cap.move:42:15\n   \u2502\n42 \u2502     ctx: &mut TxContext,\n   \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_cap.move:43:4\n   \u2502\n43 \u2502 ): ID {\n   \u2502    ^^ Unbound type 'ID' in current scope\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_cap.move:4:1\n   \u2502  \n 4 \u2502 \u256d module sui_system::validator_cap;\n 5 \u2502 \u2502 \n 6 \u2502 \u2502 /// The capability object is created when creating a new `Validator` or when the\n 7 \u2502 \u2502 /// validator explicitly creates a new capability object for rotation/revocation.\n   \u00b7 \u2502\n62 \u2502 \u2502     ValidatorOperationCap { authorizer_validator_address: cap.authorizer_validator_address }\n63 \u2502 \u2502 }\n   \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound module\n    \u250c\u2500 ./tests/validator_metadata_tests.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::validator_metadata_tests;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use std::unit_test;\n  8 \u2502 \u2502 use sui::test_scenario::{Self, Scenario};\n    \u00b7 \u2502\n917 \u2502 \u2502     assert!(validator.next_epoch_network_pubkey_bytes().is_none());\n918 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_metadata_tests.move:777:20\n    \u2502\n777 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/builders/validator_preset.move:186:26\n    \u2502\n186 \u2502         account_address: sui::address::from_bytes(preset[1]),\n    \u2502                          ^^^^^^^^^^^^ Unbound module 'sui::address'\n\nerror: unbound module\n    \u250c\u2500 ./tests/builders/validator_preset.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::validator_preset;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 const VALID_NET_PUBKEY: vector<u8> = vector[\n  8 \u2502 \u2502     171, 2, 39, 3, 139, 105, 166, 171, 153, 151, 102, 197, 151, 186, 140, 116, 114, 90, 213, 225, 20,\n    \u00b7 \u2502\n250 \u2502 \u2502     preset.project_url\n251 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:123:22\n    \u2502\n123 \u2502     staking_pool_id: ID,\n    \u2502                      ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:131:22\n    \u2502\n131 \u2502     staking_pool_id: ID,\n    \u2502                      ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_set.move:82:25\n   \u2502\n82 \u2502     at_risk_validators: VecMap<address, u64>,\n   \u2502                         ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_set.move:84:19\n   \u2502\n84 \u2502     extra_fields: Bag,\n   \u2502                   ^^^ Unbound type 'Bag' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_set.move:74:26\n   \u2502\n74 \u2502     inactive_validators: Table<ID, ValidatorWrapper>,\n   \u2502                          ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_set.move:65:32\n   \u2502\n65 \u2502     pending_active_validators: TableVec<Validator>,\n   \u2502                                ^^^^^^^^ Unbound type 'TableVec' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_set.move:70:28\n   \u2502\n70 \u2502     staking_pool_mappings: Table<ID, address>,\n   \u2502                            ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_set.move:80:27\n   \u2502\n80 \u2502     validator_candidates: Table<address, ValidatorWrapper>,\n   \u2502                           ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:376:30\n    \u2502\n376 \u2502     computation_reward: &mut Balance<SUI>,\n    \u2502                              ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:377:31\n    \u2502\n377 \u2502     storage_fund_reward: &mut Balance<SUI>,\n    \u2502                               ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:378:36\n    \u2502\n378 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:381:15\n    \u2502\n381 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:233:53\n    \u2502\n233 \u2502 fun can_join(self: &ValidatorSet, stake: u64, ctx: &TxContext): bool {\n    \u2502                                                     ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:976:36\n    \u2502\n976 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1125:44\n     \u2502\n1125 \u2502     individual_staking_reward_adjustments: VecMap<u64, u64>,\n     \u2502                                            ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1127:49\n     \u2502\n1127 \u2502     individual_storage_fund_reward_adjustments: VecMap<u64, u64>,\n     \u2502                                                 ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1014:5\n     \u2502\n1014 \u2502     VecMap<u64, u64>, // mapping of individual validator's staking reward adjustment from index -> amount\n     \u2502     ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1016:5\n     \u2502\n1016 \u2502     VecMap<u64, u64>, // mapping of individual validator's storage fund reward adjustment from index -> amount\n     \u2502     ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1066:35\n     \u2502\n1066 \u2502     mut validator_report_records: VecMap<address, VecSet<address>>,\n     \u2502                                   ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:329:15\n    \u2502\n329 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1188:27\n     \u2502\n1188 \u2502     staking_rewards: &mut Balance<SUI>,\n     \u2502                           ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1189:31\n     \u2502\n1189 \u2502     storage_fund_reward: &mut Balance<SUI>,\n     \u2502                               ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1190:15\n     \u2502\n1190 \u2502     ctx: &mut TxContext,\n     \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1237:22\n     \u2502\n1237 \u2502     report_records: &VecMap<address, VecSet<address>>,\n     \u2502                      ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:743:81\n    \u2502\n743 \u2502 fun find_validator(validators: &vector<Validator>, validator_address: address): Option<u64> {\n    \u2502                                                                                 ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:751:18\n    \u2502\n751 \u2502     validators: &TableVec<Validator>,\n    \u2502                  ^^^^^^^^ Unbound type 'TableVec' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:753:4\n    \u2502\n753 \u2502 ): Option<u64> {\n    \u2502    ^^^^^^ Unbound type 'Option' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:247:60\n    \u2502\n247 \u2502 fun get_voting_power_thresholds(self: &ValidatorSet, ctx: &TxContext): (u64, u64, u64) {\n    \u2502                                                            ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1317:81\n     \u2502\n1317 \u2502 public(package) fun inactive_validator_by_pool_id(self: &ValidatorSet, pool_id: ID): &Validator {\n     \u2502                                                                                 ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n     \u250c\u2500 ./sources/validator_set.move:1296:72\n     \u2502\n1296 \u2502 public fun is_inactive_validator(self: &ValidatorSet, staking_pool_id: ID): bool {\n     \u2502                                                                        ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:143:15\n    \u2502\n143 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:664:14\n    \u2502\n664 \u2502     pool_id: ID,\n    \u2502              ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:665:5\n    \u2502\n665 \u2502 ): &Table<u64, PoolTokenExchangeRate> {\n    \u2502     ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:921:36\n    \u2502\n921 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:922:15\n    \u2502\n922 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:945:36\n    \u2502\n945 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:947:15\n    \u2502\n947 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:348:11\n    \u2502\n348 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:349:4\n    \u2502\n349 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:292:12\n    \u2502\n292 \u2502     stake: Balance<SUI>,\n    \u2502            ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:293:15\n    \u2502\n293 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:221:74\n    \u2502\n221 \u2502 public(package) fun request_add_validator(self: &mut ValidatorSet, ctx: &TxContext) {\n    \u2502                                                                          ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:171:15\n    \u2502\n171 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:273:77\n    \u2502\n273 \u2502 public(package) fun request_remove_validator(self: &mut ValidatorSet, ctx: &TxContext) {\n    \u2502                                                                             ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:198:15\n    \u2502\n198 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:311:11\n    \u2502\n311 \u2502     ctx: &TxContext,\n    \u2502           ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:312:4\n    \u2502\n312 \u2502 ): Balance<SUI> {\n    \u2502    ^^^^^^^ Unbound type 'Balance' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:648:57\n    \u2502\n648 \u2502 public fun staking_pool_mappings(self: &ValidatorSet): &Table<ID, address> {\n    \u2502                                                         ^^^^^ Unbound type 'Table' in current scope\n\nerror: unbound module\n     \u250c\u2500 ./sources/validator_set.move:4:1\n     \u2502  \n   4 \u2502 \u256d module sui_system::validator_set;\n   5 \u2502 \u2502 \n   6 \u2502 \u2502 use sui::bag::{Self, Bag};\n   7 \u2502 \u2502 use sui::balance::Balance;\n     \u00b7 \u2502\n1369 \u2502 \u2502     abort\n1370 \u2502 \u2502 }\n     \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:497:36\n    \u2502\n497 \u2502     validator_report_records: &mut VecMap<address, VecSet<address>>,\n    \u2502                                    ^^^^^^ Unbound type 'VecMap' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:498:15\n    \u2502\n498 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:652:76\n    \u2502\n652 \u2502 public fun validator_address_by_pool_id(self: &mut ValidatorSet, pool_id: &ID): address {\n    \u2502                                                                            ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:678:76\n    \u2502\n678 \u2502 public(package) fun validator_by_pool_id(self: &mut ValidatorSet, pool_id: ID): &Validator {\n    \u2502                                                                            ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./sources/validator_set.move:644:88\n    \u2502\n644 \u2502 public fun validator_staking_pool_id(self: &ValidatorSet, validator_address: address): ID {\n    \u2502                                                                                        ^^ Unbound type 'ID' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:676:20\n    \u2502\n676 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:295:5\n    \u2502\n295 \u2502     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:304:5\n    \u2502\n304 \u2502     assert_eq!(validator_set.validator_address_by_pool_id(&pool_id_2), @0x2);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:311:5\n    \u2502\n311 \u2502     assert_eq!(validator_set.validator_address_by_pool_id(&pool_id_2), @0x2);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:313:5\n    \u2502\n313 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:547:5\n    \u2502\n547 \u2502     destroy(stake);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:561:9\n    \u2502\n561 \u2502         destroy(stake);\n    \u2502         ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:571:5\n    \u2502\n571 \u2502     assert_eq!(effects.num_user_events(), num_validators); // epoch changes hould not emit ValidatorJoinEvent or ValidatorLeaveEvent\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:573:5\n    \u2502\n573 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:220:5\n    \u2502\n220 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:275:5\n    \u2502\n275 \u2502     assert_eq!(effects.num_user_events(), num_validators + 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:277:5\n    \u2502\n277 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:669:87\n    \u2502\n669 \u2502 fun advance_epoch_with_dummy_rewards(validator_set: &mut ValidatorSet, scenario: &mut Scenario) {\n    \u2502                                                                                       ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:650:20\n    \u2502\n650 \u2502     scenario: &mut Scenario,\n    \u2502                    ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:582:15\n    \u2502\n582 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:614:15\n    \u2502\n614 \u2502     ctx: &mut TxContext,\n    \u2502               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:225:33\n    \u2502\n225 \u2502 fun get_10_validators(ctx: &mut TxContext): vector<Validator> {\n    \u2502                                 ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:456:5\n    \u2502\n456 \u2502     assert_eq!(validator_set.find_for_testing(@0xB).voting_power(), 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:474:5\n    \u2502\n474 \u2502     assert_eq!(effects.num_user_events(), num_validators + 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:476:5\n    \u2502\n476 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:477:5\n    \u2502\n477 \u2502     destroy(bal);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:478:5\n    \u2502\n478 \u2502     destroy(stake);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:523:5\n    \u2502\n523 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:524:5\n    \u2502\n524 \u2502     destroy(stake1);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:525:5\n    \u2502\n525 \u2502     destroy(stake2);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:105:5\n    \u2502\n105 \u2502     assert_eq!(validator_set.derive_reference_gas_price(), 45);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:110:5\n    \u2502\n110 \u2502     assert_eq!(validator_set.derive_reference_gas_price(), 45);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:119:5\n    \u2502\n119 \u2502     assert_eq!(validator_set.derive_reference_gas_price(), 42);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:128:5\n    \u2502\n128 \u2502     assert_eq!(validator_set.derive_reference_gas_price(), 42);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:137:5\n    \u2502\n137 \u2502     assert_eq!(validator_set.derive_reference_gas_price(), 43);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:139:5\n    \u2502\n139 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:341:5\n    \u2502\n341 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:342:5\n    \u2502\n342 \u2502     destroy(bal);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/validator_set_tests.move:241:58\n    \u2502\n241 \u2502 fun skip_to_min_stake_v2_final_thresholds(scenario: &mut Scenario) {\n    \u2502                                                          ^^^^^^^^ Unbound type 'Scenario' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:151:5\n    \u2502\n151 \u2502     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:164:5\n    \u2502\n164 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:176:5\n    \u2502\n176 \u2502     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:192:5\n    \u2502\n192 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/validator_set_tests.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::validator_set_tests;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use std::unit_test::{assert_eq, destroy};\n  8 \u2502 \u2502 use sui::address;\n    \u00b7 \u2502\n681 \u2502 \u2502     validator_set.request_add_validator(ctx);\n682 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:31:5\n   \u2502\n31 \u2502     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:40:5\n   \u2502\n40 \u2502     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:61:9\n   \u2502\n61 \u2502         assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:72:5\n   \u2502\n72 \u2502     assert_eq!(validator_set.total_stake(), 1500 * MIST_PER_SUI);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:82:5\n   \u2502\n82 \u2502     assert_eq!(validator_set.total_stake(), 1500 * MIST_PER_SUI);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:85:5\n   \u2502\n85 \u2502     assert_eq!(validator_set.total_stake(), 900 * MIST_PER_SUI);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_set_tests.move:87:5\n   \u2502\n87 \u2502     destroy(validator_set);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:416:5\n    \u2502\n416 \u2502     assert_eq!(effects.num_user_events(), num_validators + 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:418:5\n    \u2502\n418 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:419:5\n    \u2502\n419 \u2502     destroy(bal);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:420:5\n    \u2502\n420 \u2502     destroy(stake);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:376:5\n    \u2502\n376 \u2502     assert_eq!(effects.num_user_events(), num_validators + 1);\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:378:5\n    \u2502\n378 \u2502     destroy(validator_set);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_set_tests.move:379:5\n    \u2502\n379 \u2502     destroy(bal);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:102:5\n    \u2502\n102 \u2502     destroy(metadata);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:69:5\n   \u2502\n69 \u2502     assert_eq!(validator.total_stake(), initial_stake);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:70:5\n   \u2502\n70 \u2502     assert_eq!(validator.pending_stake_amount(), added_stake);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:78:9\n   \u2502\n78 \u2502         assert_eq!(withdrawn_balance, initial_stake);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:79:9\n   \u2502\n79 \u2502         assert_eq!(validator.total_stake(), initial_stake);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:80:9\n   \u2502\n80 \u2502         assert_eq!(validator.pending_stake_amount(), added_stake);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:81:9\n   \u2502\n81 \u2502         assert_eq!(validator.pending_stake_withdraw_amount(), initial_stake);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:87:9\n   \u2502\n87 \u2502         assert_eq!(validator.total_stake(), added_stake);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:88:9\n   \u2502\n88 \u2502         assert_eq!(validator.pending_stake_amount(), 0);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:89:9\n   \u2502\n89 \u2502         assert_eq!(validator.pending_stake_withdraw_amount(), 0);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/validator_tests.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::validator_tests;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use std::unit_test::{assert_eq, destroy};\n  8 \u2502 \u2502 use sui::balance;\n    \u00b7 \u2502\n485 \u2502 \u2502     abort\n486 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:33:5\n   \u2502\n33 \u2502     assert_eq!(validator.total_stake(), initial_stake);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:34:5\n   \u2502\n34 \u2502     assert_eq!(validator.sui_address(), @2);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:38:9\n   \u2502\n38 \u2502         assert_eq!(stake.amount(), initial_stake);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:39:9\n   \u2502\n39 \u2502         assert_eq!(stake.pool_id(), pool_id);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/validator_tests.move:40:9\n   \u2502\n40 \u2502         assert_eq!(stake.stake_activation_epoch(), 0);\n   \u2502         ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:246:5\n    \u2502\n246 \u2502     assert_eq!(*validator.name(), b\"new_name\".to_string());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:247:5\n    \u2502\n247 \u2502     assert_eq!(*validator.description(), b\"new_desc\".to_string());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:248:5\n    \u2502\n248 \u2502     assert_eq!(*validator.image_url(), url::new_unsafe_from_bytes(b\"new_image_url\"));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:249:5\n    \u2502\n249 \u2502     assert_eq!(*validator.project_url(), url::new_unsafe_from_bytes(b\"new_proj_url\"));\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:250:5\n    \u2502\n250 \u2502     assert_eq!(*validator.network_address(), validator_builder::valid_net_addr().to_string());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:251:5\n    \u2502\n251 \u2502     assert_eq!(*validator.p2p_address(), validator_builder::valid_p2p_addr().to_string());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:252:5\n    \u2502\n252 \u2502     assert_eq!(*validator.primary_address(), validator_builder::valid_consensus_addr().to_string());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:253:5\n    \u2502\n253 \u2502     assert_eq!(*validator.worker_address(), validator_builder::valid_worker_addr().to_string());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:254:5\n    \u2502\n254 \u2502     assert_eq!(*validator.protocol_pubkey_bytes(), validator_builder::valid_pubkey());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:255:5\n    \u2502\n255 \u2502     assert_eq!(*validator.proof_of_possession(), validator_builder::valid_proof_of_possession());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:256:5\n    \u2502\n256 \u2502     assert_eq!(*validator.network_pubkey_bytes(), validator_builder::valid_net_pubkey());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:257:5\n    \u2502\n257 \u2502     assert_eq!(*validator.worker_pubkey_bytes(), validator_builder::valid_worker_pubkey());\n    \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n    \u250c\u2500 ./tests/validator_tests.move:293:5\n    \u2502\n293 \u2502     destroy(validator);\n    \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_wrapper.move:12:12\n   \u2502\n12 \u2502     inner: Versioned,\n   \u2502            ^^^^^^^^^ Unbound type 'Versioned' in current scope\n\nerror: unbound type\n   \u250c\u2500 ./sources/validator_wrapper.move:16:63\n   \u2502\n16 \u2502 public(package) fun create_v1(validator: Validator, ctx: &mut TxContext): ValidatorWrapper {\n   \u2502                                                               ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n   \u250c\u2500 ./sources/validator_wrapper.move:4:1\n   \u2502  \n 4 \u2502 \u256d module sui_system::validator_wrapper;\n 5 \u2502 \u2502 \n 6 \u2502 \u2502 use sui::versioned::{Self, Versioned};\n 7 \u2502 \u2502 use sui_system::validator::Validator;\n   \u00b7 \u2502\n49 \u2502 \u2502     self.inner.version()\n50 \u2502 \u2502 }\n   \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound module\n    \u250c\u2500 ./sources/voting_power.move:4:1\n    \u2502  \n  4 \u2502 \u256d module sui_system::voting_power;\n  5 \u2502 \u2502 \n  6 \u2502 \u2502 use sui_system::validator::Validator;\n  7 \u2502 \u2502 \n    \u00b7 \u2502\n165 \u2502 \u2502     QUORUM_THRESHOLD\n166 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: unbound type\n   \u250c\u2500 ./tests/voting_power_tests.move:15:69\n   \u2502\n15 \u2502 fun check(stakes: vector<u64>, voting_power: vector<u64>, ctx: &mut TxContext) {\n   \u2502                                                                     ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/voting_power_tests.move:26:5\n   \u2502\n26 \u2502     assert_eq!(voting_powers, voting_power);\n   \u2502     ^^^^^^^^^ Unbound function 'assert_eq' in current scope\n\nerror: unbound unscoped name\n   \u250c\u2500 ./tests/voting_power_tests.move:27:5\n   \u2502\n27 \u2502     destroy(validators);\n   \u2502     ^^^^^^^ Unbound function 'destroy' in current scope\n\nerror: unbound type\n    \u250c\u2500 ./tests/voting_power_tests.move:117:66\n    \u2502\n117 \u2502 fun create_validators_with_stakes(stakes: vector<u64>, ctx: &mut TxContext): vector<Validator> {\n    \u2502                                                                  ^^^^^^^^^ Unbound type 'TxContext' in current scope\n\nerror: unbound module\n    \u250c\u2500 ./tests/voting_power_tests.move:5:1\n    \u2502  \n  5 \u2502 \u256d module sui_system::voting_power_tests;\n  6 \u2502 \u2502 \n  7 \u2502 \u2502 use std::unit_test::{assert_eq, destroy};\n  8 \u2502 \u2502 use sui_system::validator::{Self, Validator};\n    \u00b7 \u2502\n123 \u2502 \u2502     })\n124 \u2502 \u2502 }\n    \u2502 \u2570\u2500^ Unbound module 'std::unit_test'\n\nerror: invalid method call\n    \u250c\u2500 ./tests/rewards_distribution_tests.move:630:9\n    \u2502  \n615 \u2502       let num_validators = 20;\n    \u2502           -------------- Unable to infer type for method call. Try annotating this type\n    \u00b7  \n630 \u2502 \u256d         num_validators.do!(|i| {\n631 \u2502 \u2502             let addr = address::from_u256(i as u256);\n632 \u2502 \u2502             assert_eq!(system.validator_stake_amount(addr), (962 + i * 4) * MIST_PER_SUI);\n633 \u2502 \u2502         });\n    \u2502 \u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500^ Invalid method call\n\nerror: invalid method call\n     \u250c\u2500 ./sources/validator_set.move:1079:14\n     \u2502\n1070 \u2502         let (validator_address, reporters) = validator_report_records.pop();\n     \u2502                                 --------- Unable to infer type for method call. Try annotating this type\n     \u00b7\n1079 \u2502             &reporters.into_keys(),\n     \u2502              ^^^^^^^^^^^^^^^^^^^^^ Invalid method call\n\nerror: cannot infer type\n     \u250c\u2500 ./sources/validator_set.move:1070:33\n     \u2502\n1070 \u2502         let (validator_address, reporters) = validator_report_records.pop();\n     \u2502                                 ^^^^^^^^^ Could not infer this type. Try adding an annotation\n\nerror: cannot infer type\n     \u250c\u2500 ./sources/validator_set.move:1247:13\n     \u2502\n1247 \u2502             vector[]\n     \u2502             ^^^^^^^^ Could not infer this type. Try adding an annotation\n\nerror: invalid object declaration\n   \u250c\u2500 ./sources/staking_pool.move:90:5\n   \u2502\n89 \u2502 public struct FungibleStakedSui has key, store {\n   \u2502                                     --- The 'key' ability is used to declare objects in Sui\n90 \u2502     id: UID,\n   \u2502     ^^  --- But found type: '_'\n   \u2502     \u2502    \n   \u2502     Invalid object 'FungibleStakedSui'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field\n\nerror: invalid object declaration\n   \u250c\u2500 ./sources/staking_pool.move:99:5\n   \u2502\n98 \u2502 public struct FungibleStakedSuiData has key, store {\n   \u2502                                         --- The 'key' ability is used to declare objects in Sui\n99 \u2502     id: UID,\n   \u2502     ^^  --- But found type: '_'\n   \u2502     \u2502    \n   \u2502     Invalid object 'FungibleStakedSuiData'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field\n\nerror: invalid object declaration\n   \u250c\u2500 ./sources/staking_pool.move:76:5\n   \u2502\n75 \u2502 public struct StakedSui has key, store {\n   \u2502                             --- The 'key' ability is used to declare objects in Sui\n76 \u2502     id: UID,\n   \u2502     ^^  --- But found type: '_'\n   \u2502     \u2502    \n   \u2502     Invalid object 'StakedSui'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field\n\nerror: invalid object declaration\n   \u250c\u2500 ./sources/staking_pool.move:39:5\n   \u2502\n38 \u2502 public struct StakingPool has key, store {\n   \u2502                               --- The 'key' ability is used to declare objects in Sui\n39 \u2502     id: UID,\n   \u2502     ^^  --- But found type: '_'\n   \u2502     \u2502    \n   \u2502     Invalid object 'StakingPool'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field\n\nerror: invalid object declaration\n   \u250c\u2500 ./sources/sui_system.move:67:5\n   \u2502\n66 \u2502 public struct SuiSystemState has key {\n   \u2502                                  --- The 'key' ability is used to declare objects in Sui\n67 \u2502     id: UID,\n   \u2502     ^^  --- But found type: '_'\n   \u2502     \u2502    \n   \u2502     Invalid object 'SuiSystemState'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field\n\nerror: invalid object declaration\n   \u250c\u2500 ./sources/validator_cap.move:18:5\n   \u2502\n17 \u2502 public struct UnverifiedValidatorOperationCap has key, store {\n   \u2502                                                   --- The 'key' ability is used to declare objects in Sui\n18 \u2502     id: UID,\n   \u2502     ^^  --- But found type: '_'\n   \u2502     \u2502    \n   \u2502     Invalid object 'UnverifiedValidatorOperationCap'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field\n\n\n","type":"ProduceError"} -->
```
⚠ ProduceError: get_callees failed for staking_pool::stake_activation_epoch at /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system: Error: Compilation failed:
error: unbound module
  ┌─ ./sources/genesis.move:6:5
  │
6 │ use sui::balance::{Self, Balance};
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./sources/genesis.move:7:5
  │
7 │ use sui::sui::SUI;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unexpected name in this position
    ┌─ ./sources/genesis.move:138:24
    │
138 │     let storage_fund = balance::zero();
    │                        ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/genesis.move:199:17
    │
199 │                 transfer::public_transfer(allocation_balance.into_coin(ctx), recipient_address);
    │                 ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/stake_subsidy.move:6:5
  │
6 │ use sui::bag::{Self, Bag};
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'

error: unbound module
  ┌─ ./sources/stake_subsidy.move:7:5
  │
7 │ use sui::balance::Balance;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./sources/stake_subsidy.move:8:5
  │
8 │ use sui::sui::SUI;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unexpected name in this position
   ┌─ ./sources/stake_subsidy.move:50:23
   │
50 │         extra_fields: bag::new(ctx),
   │                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/staking_pool.move:7:5
  │
7 │ use sui::bag::{Self, Bag};
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'

error: unbound module
  ┌─ ./sources/staking_pool.move:8:5
  │
8 │ use sui::balance::{Self, Balance};
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./sources/staking_pool.move:9:5
  │
9 │ use sui::sui::SUI;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./sources/staking_pool.move:10:5
   │
10 │ use sui::table::{Self, Table};
   │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:118:13
    │
118 │         id: object::new(ctx),
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:119:27
    │
119 │         activation_epoch: option::none(),
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:120:29
    │
120 │         deactivation_epoch: option::none(),
    │                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:122:23
    │
122 │         rewards_pool: balance::zero(),
    │                       ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:124:25
    │
124 │         exchange_rates: table::new(ctx),
    │                         ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:128:23
    │
128 │         extra_fields: bag::new(ctx),
    │                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:147:13
    │
147 │         id: object::new(ctx),
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:148:18
    │
148 │         pool_id: object::id(pool),
    │                  ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:201:24
    │
201 │     assert!(pool_id == object::id(pool), EWrongPool);
    │                        ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:281:24
    │
281 │     assert!(pool_id == object::id(pool), EWrongPool);
    │                        ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:302:25
    │
302 │                     id: object::new(ctx),
    │                         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:315:13
    │
315 │         id: object::new(ctx),
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:329:35
    │
329 │     assert!(staked_sui.pool_id == object::id(pool), EWrongPool);
    │                                   ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:464:31
    │
464 │     pool.deactivation_epoch = option::some(deactivation_epoch);
    │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:527:13
    │
527 │         id: object::new(ctx),
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:555:13
    │
555 │         id: object::new(ctx),
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:569:5
    │
569 │     transfer::transfer(stake.split(split_amount, ctx), ctx.sender());
    │     ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:713:5
    │
713 │     bag::borrow(&pool.extra_fields, FungibleStakedSuiDataKey {})
    │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:748:13
    │
748 │         id: object::new(ctx),
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/staking_pool.move:749:18
    │
749 │         pool_id: object::id(self),
    │                  ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/storage_fund.move:6:5
  │
6 │ use sui::balance::{Self, Balance};
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./sources/storage_fund.move:7:5
  │
7 │ use sui::sui::SUI;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unexpected name in this position
   ┌─ ./sources/storage_fund.move:26:39
   │
26 │         total_object_storage_rebates: balance::zero(),
   │                                       ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
   ┌─ ./sources/sui_system.move:42:5
   │
42 │ use sui::balance::Balance;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./sources/sui_system.move:43:5
   │
43 │ use sui::coin::Coin;
   │     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'

error: unbound module
   ┌─ ./sources/sui_system.move:44:5
   │
44 │ use sui::dynamic_field;
   │     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::dynamic_field'

error: unbound module
   ┌─ ./sources/sui_system.move:45:5
   │
45 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./sources/sui_system.move:46:5
   │
46 │ use sui::table::Table;
   │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'

error: unbound module
   ┌─ ./sources/sui_system.move:47:5
   │
47 │ use sui::vec_map::VecMap;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'

error: unbound module
   ┌─ ./sources/sui_system.move:60:5
   │
60 │ use sui::balance;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./sources/sui_system.move:64:5
   │
64 │ use sui::vec_set::VecSet;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_set'

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:102:5
    │
102 │     dynamic_field::add(&mut self.id, version, system_state);
    │     ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:103:5
    │
103 │     transfer::share_object(self);
    │     ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:236:5
    │
236 │     transfer::public_transfer(staked_sui, ctx.sender());
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:254:19
    │
254 │     stake_amount: option::Option<u64>,
    │                   ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid type

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:262:5
    │
262 │     transfer::public_transfer(staked_sui, ctx.sender());
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:273:5
    │
273 │     transfer::public_transfer(withdrawn_stake.into_coin(ctx), ctx.sender());
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:634:41
    │
634 │     let inner: &SuiSystemStateInnerV2 = dynamic_field::borrow(
    │                                         ^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:644:39
    │
644 │         let v1: SuiSystemStateInner = dynamic_field::remove(&mut self.id, self.version);
    │                                       ^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:647:9
    │
647 │         dynamic_field::add(&mut self.id, self.version, v2);
    │         ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:650:45
    │
650 │     let inner: &mut SuiSystemStateInnerV2 = dynamic_field::borrow_mut(
    │                                             ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:892:26
    │
892 │     let storage_reward = balance::create_for_testing(storage_charge);
    │                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system.move:893:30
    │
893 │     let computation_reward = balance::create_for_testing(computation_charge);
    │                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/sui_system_state_inner.move:6:5
  │
6 │ use sui::bag::{Self, Bag};
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'

error: unbound module
  ┌─ ./sources/sui_system_state_inner.move:7:5
  │
7 │ use sui::balance::{Self, Balance};
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./sources/sui_system_state_inner.move:8:5
  │
8 │ use sui::coin::Coin;
  │     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'

error: unbound module
  ┌─ ./sources/sui_system_state_inner.move:9:5
  │
9 │ use sui::event;
  │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::event'

error: unbound module
   ┌─ ./sources/sui_system_state_inner.move:10:5
   │
10 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./sources/sui_system_state_inner.move:11:5
   │
11 │ use sui::table::Table;
   │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'

error: unbound module
   ┌─ ./sources/sui_system_state_inner.move:12:5
   │
12 │ use sui::vec_map::{Self, VecMap};
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'

error: unbound module
   ┌─ ./sources/sui_system_state_inner.move:13:5
   │
13 │ use sui::vec_set::{Self, VecSet};
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_set'

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:239:35
    │
239 │         validator_report_records: vec_map::empty(),
    │                                   ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:242:36
    │
242 │         safe_mode_storage_rewards: balance::zero(),
    │                                    ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:243:40
    │
243 │         safe_mode_computation_rewards: balance::zero(),
    │                                        ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:247:23
    │
247 │         extra_fields: bag::new(ctx),
    │                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:271:23
    │
271 │         extra_fields: bag::new(ctx),
    │                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:567:56
    │
567 │         validator_report_records.insert(reportee_addr, vec_set::singleton(reporter_address));
    │                                                        ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/sui_system_state_inner.move:907:29
    │
907 │     let mut stake_subsidy = balance::zero();
    │                             ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/sui_system_state_inner.move:1000:5
     │
1000 │     event::emit(SystemEpochInfoEvent {
     │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/sui_system_state_inner.move:1068:29
     │
1068 │     let mut voting_powers = vec_map::empty();
     │                             ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/sui_system_state_inner.move:1095:10
     │
1095 │     else vec_set::empty()
     │          ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/sui_system_state_inner.move:1140:13
     │
1140 │             transfer::public_transfer(total_balance.into_coin(ctx), ctx.sender());
     │             ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/sui_system_state_inner.move:1299:9
     │
1299 │         option::none(),
     │         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/validator.move:7:5
  │
7 │ use std::bcs;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'std::bcs'

error: unbound module
  ┌─ ./sources/validator.move:8:5
  │
8 │ use std::string::String;
  │     ^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::string'

error: unbound module
  ┌─ ./sources/validator.move:9:5
  │
9 │ use sui::bag::{Self, Bag};
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'

error: unbound module
   ┌─ ./sources/validator.move:10:5
   │
10 │ use sui::balance::Balance;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./sources/validator.move:11:5
   │
11 │ use sui::event;
   │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::event'

error: unbound module
   ┌─ ./sources/validator.move:12:5
   │
12 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./sources/validator.move:13:5
   │
13 │ use sui::url::{Self, Url};
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'

error: unexpected name in this position
    ┌─ ./sources/validator.move:199:43
    │
199 │         next_epoch_protocol_pubkey_bytes: option::none(),
    │                                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:200:42
    │
200 │         next_epoch_network_pubkey_bytes: option::none(),
    │                                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:201:41
    │
201 │         next_epoch_worker_pubkey_bytes: option::none(),
    │                                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:202:41
    │
202 │         next_epoch_proof_of_possession: option::none(),
    │                                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:203:33
    │
203 │         next_epoch_net_address: option::none(),
    │                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:204:33
    │
204 │         next_epoch_p2p_address: option::none(),
    │                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:205:37
    │
205 │         next_epoch_primary_address: option::none(),
    │                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:206:36
    │
206 │         next_epoch_worker_address: option::none(),
    │                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:251:9
    │
251 │         url::new_unsafe_from_bytes(image_url),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:252:9
    │
252 │         url::new_unsafe_from_bytes(project_url),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:257:9
    │
257 │         bag::new(ctx),
    │         ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:297:5
    │
297 │     event::emit(StakingRequestEvent {
    │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:316:5
    │
316 │     event::emit(ConvertingToFungibleStakedSuiEvent {
    │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:336:5
    │
336 │     event::emit(RedeemingFungibleStakedSuiEvent {
    │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:359:5
    │
359 │     transfer::public_transfer(staked_sui, staker_address);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:378:5
    │
378 │     event::emit(UnstakingRequestEvent {
    │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:591:5
    │
591 │     object::id(&self.staking_pool)
    │     ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:682:31
    │
682 │     self.metadata.image_url = url::new_unsafe_from_bytes(image_url);
    │                               ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:691:33
    │
691 │     self.metadata.project_url = url::new_unsafe_from_bytes(project_url);
    │                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:704:44
    │
704 │     self.metadata.next_epoch_net_address = option::some(net_address);
    │                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:730:44
    │
730 │     self.metadata.next_epoch_p2p_address = option::some(p2p_address);
    │                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:756:48
    │
756 │     self.metadata.next_epoch_primary_address = option::some(primary_address);
    │                                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:785:47
    │
785 │     self.metadata.next_epoch_worker_address = option::some(worker_address);
    │                                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:811:54
    │
811 │     self.metadata.next_epoch_protocol_pubkey_bytes = option::some(protocol_pubkey);
    │                                                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:812:52
    │
812 │     self.metadata.next_epoch_proof_of_possession = option::some(proof_of_possession);
    │                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:834:53
    │
834 │     self.metadata.next_epoch_network_pubkey_bytes = option::some(network_pubkey);
    │                                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:853:52
    │
853 │     self.metadata.next_epoch_worker_pubkey_bytes = option::some(worker_pubkey);
    │                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:907:27
    │
907 │     validate_metadata_bcs(bcs::to_bytes(metadata));
    │                           ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:943:23
    │
943 │         extra_fields: bag::new(ctx),
    │                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:982:13
    │
982 │             url::new_unsafe_from_bytes(image_url),
    │             ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:983:13
    │
983 │             url::new_unsafe_from_bytes(project_url),
    │             ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator.move:988:13
    │
988 │             bag::new(ctx),
    │             ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./sources/validator_cap.move:51:13
   │
51 │         id: object::new(ctx),
   │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./sources/validator_cap.move:54:28
   │
54 │     let operation_cap_id = object::id(&operation_cap);
   │                            ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./sources/validator_cap.move:55:5
   │
55 │     transfer::public_transfer(operation_cap, validator_address);
   │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/validator_set.move:6:5
  │
6 │ use sui::bag::{Self, Bag};
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'

error: unbound module
  ┌─ ./sources/validator_set.move:7:5
  │
7 │ use sui::balance::Balance;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./sources/validator_set.move:8:5
  │
8 │ use sui::event;
  │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::event'

error: unbound module
  ┌─ ./sources/validator_set.move:9:5
  │
9 │ use sui::priority_queue as pq;
  │     ^^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::priority_queue'

error: unbound module
   ┌─ ./sources/validator_set.move:10:5
   │
10 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./sources/validator_set.move:11:5
   │
11 │ use sui::table::{Self, Table};
   │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'

error: unbound module
   ┌─ ./sources/validator_set.move:12:5
   │
12 │ use sui::table_vec::{Self, TableVec};
   │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table_vec'

error: unbound module
   ┌─ ./sources/validator_set.move:13:5
   │
13 │ use sui::vec_map::{Self, VecMap};
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'

error: unbound module
   ┌─ ./sources/validator_set.move:14:5
   │
14 │ use sui::vec_set::VecSet;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_set'

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:146:37
    │
146 │     let mut staking_pool_mappings = table::new(ctx);
    │                                     ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:153:36
    │
153 │         pending_active_validators: table_vec::empty(ctx),
    │                                    ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:156:30
    │
156 │         inactive_validators: table::new(ctx),
    │                              ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:157:31
    │
157 │         validator_candidates: table::new(ctx),
    │                               ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:158:29
    │
158 │         at_risk_validators: vec_map::empty(),
    │                             ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:159:23
    │
159 │         extra_fields: bag::new(ctx),
    │                       ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./sources/validator_set.move:501:37
    │
501 │     let pending_active_validators = vector::tabulate!(
    │                                     ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:501:37
    │
501 │     let pending_active_validators = vector::tabulate!(
    │                                     ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:585:13
    │
585 │             event::emit(ValidatorJoinEvent {
    │             ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:611:23
    │
611 │         .map_ref!(|v| pq::new_entry(v.gas_price(), v.voting_power()));
    │                       ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:614:18
    │
614 │     let mut pq = pq::new(entries);
    │                  ^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:757:32
    │
757 │                 return 'search option::some(i)
    │                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:761:9
    │
761 │         option::none()
    │         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:913:46
    │
913 │     assert!(validator.operation_cap_id() == &object::id(cap), EInvalidCap);
    │                                              ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./sources/validator_set.move:961:5
    │
961 │     event::emit(ValidatorLeaveEvent {
    │     ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/validator_set.move:1019:53
     │
1019 │     let mut individual_staking_reward_adjustments = vec_map::empty();
     │                                                     ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/validator_set.move:1021:58
     │
1021 │     let mut individual_storage_fund_reward_adjustments = vec_map::empty();
     │                                                          ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/validator_set.move:1220:13
     │
1220 │             transfer::public_transfer(rewards_stake, validator_address);
     │             ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./sources/validator_set.move:1254:9
     │
1254 │         event::emit(ValidatorEpochInfoEventV2 {
     │         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./sources/validator_wrapper.move:6:5
  │
6 │ use sui::versioned::{Self, Versioned};
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::versioned'

error: unexpected name in this position
   ┌─ ./sources/validator_wrapper.move:18:16
   │
18 │         inner: versioned::create(1, validator, ctx),
   │                ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/builders/test_runner.move:9:5
  │
9 │ use sui::balance::{Self, Balance};
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./tests/builders/test_runner.move:10:5
   │
10 │ use sui::coin::{Self, Coin};
   │     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'

error: unbound module
   ┌─ ./tests/builders/test_runner.move:11:5
   │
11 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./tests/builders/test_runner.move:12:5
   │
12 │ use sui::test_scenario::{Self, Scenario};
   │     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:41:21
   │
41 │         validators: option::none(),
   │                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:42:27
   │
42 │         validators_count: option::none(),
   │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:43:28
   │
43 │         sui_supply_amount: option::none(),
   │                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:44:30
   │
44 │         storage_fund_amount: option::none(),
   │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:45:35
   │
45 │         validators_initial_stake: option::none(),
   │                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:46:27
   │
46 │         protocol_version: option::none(),
   │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:47:37
   │
47 │         stake_distribution_counter: option::none(),
   │                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:48:25
   │
48 │         epoch_duration: option::none(),
   │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:49:22
   │
49 │         start_epoch: option::none(),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:50:33
   │
50 │         low_stake_grace_period: option::none(),
   │                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:55:24
   │
55 │     let mut scenario = test_scenario::begin(@0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
   ┌─ ./tests/builders/test_runner.move:73:9
   │
73 │         vector::tabulate!(validators_count, |idx| {
   │         ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:73:9
   │
73 │         vector::tabulate!(validators_count, |idx| {
   │         ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/test_runner.move:99:9
   │
99 │         balance::create_for_testing<SUI>(sui_supply_amount.destroy_or!(1000) * MIST_PER_SUI), // sui_supply
   │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:116:9
    │
116 │         object::new(scenario.ctx()), // it doesn't matter what ID sui system state has in tests
    │         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:118:9
    │
118 │         balance::create_for_testing<SUI>(storage_fund_amount.destroy_or!(0) * MIST_PER_SUI), // storage_fund
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:152:30
    │
152 │     builder.epoch_duration = option::some(epoch_duration);
    │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:168:32
    │
168 │     builder.validators_count = option::some(validators_count);
    │                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:176:33
    │
176 │     builder.sui_supply_amount = option::some(sui_supply_amount);
    │                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:184:35
    │
184 │     builder.storage_fund_amount = option::some(storage_fund_amount);
    │                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:192:40
    │
192 │     builder.validators_initial_stake = option::some(validators_initial_stake);
    │                                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:197:27
    │
197 │     builder.start_epoch = option::some(start_epoch);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:205:32
    │
205 │     builder.protocol_version = option::some(protocol_version);
    │                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:213:42
    │
213 │     builder.stake_distribution_counter = option::some(stake_distribution_counter);
    │                                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:231:27
    │
231 │         protocol_version: option::none(),
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:232:25
    │
232 │         storage_charge: option::none(),
    │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:233:29
    │
233 │         computation_charge: option::none(),
    │                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:234:25
    │
234 │         storage_rebate: option::none(),
    │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:235:37
    │
235 │         non_refundable_storage_fee: option::none(),
    │                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:236:37
    │
236 │         storage_fund_reinvest_rate: option::none(),
    │                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:237:31
    │
237 │         reward_slashing_rate: option::none(),
    │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:238:27
    │
238 │         epoch_start_time: option::none(),
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:248:29
    │
248 │     opts.protocol_version = option::some(protocol_version);
    │                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:253:27
    │
253 │     opts.storage_charge = option::some(storage_charge);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:261:31
    │
261 │     opts.computation_charge = option::some(computation_charge);
    │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:266:27
    │
266 │     opts.storage_rebate = option::some(storage_rebate);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:274:39
    │
274 │     opts.non_refundable_storage_fee = option::some(non_refundable_storage_fee);
    │                                       ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:282:39
    │
282 │     opts.storage_fund_reinvest_rate = option::some(storage_fund_reinvest_rate);
    │                                       ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:290:33
    │
290 │     opts.reward_slashing_rate = option::some(reward_slashing_rate);
    │                                 ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:298:29
    │
298 │     opts.epoch_start_time = option::some(epoch_start_time);
    │                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:331:5
    │
331 │     transfer::public_transfer(object, runner.sender);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:339:5
    │
339 │     balance::create_for_testing(amount * MIST_PER_SUI)
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:381:5
    │
381 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:467:13
    │
467 │             coin::mint_for_testing(amount * MIST_PER_SUI, ctx),
    │             ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:486:17
    │
486 │                 coin::mint_for_testing(amount * MIST_PER_SUI, ctx),
    │                 ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:599:5
    │
599 │     test_scenario::return_shared(system);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/test_runner.move:615:26
    │
615 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
   ┌─ ./tests/builders/validator_builder.move:12:5
   │
12 │ use sui::bag;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::bag'

error: unbound module
   ┌─ ./tests/builders/validator_builder.move:13:5
   │
13 │ use sui::balance;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./tests/builders/validator_builder.move:14:5
   │
14 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./tests/builders/validator_builder.move:15:5
   │
15 │ use sui::url;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:44:22
   │
44 │         sui_address: option::none(),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:45:32
   │
45 │         protocol_pubkey_bytes: option::none(),
   │                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:46:31
   │
46 │         network_pubkey_bytes: option::none(),
   │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:47:30
   │
47 │         worker_pubkey_bytes: option::none(),
   │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:48:30
   │
48 │         proof_of_possession: option::none(),
   │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:49:15
   │
49 │         name: option::none(),
   │               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:50:22
   │
50 │         description: option::none(),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:51:20
   │
51 │         image_url: option::none(),
   │                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:52:22
   │
52 │         project_url: option::none(),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:53:22
   │
53 │         net_address: option::none(),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:54:22
   │
54 │         p2p_address: option::none(),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:55:26
   │
55 │         primary_address: option::none(),
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:56:25
   │
56 │         worker_address: option::none(),
   │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:57:20
   │
57 │         gas_price: option::none(),
   │                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:58:26
   │
58 │         commission_rate: option::none(),
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:60:24
   │
60 │         initial_stake: option::none(),
   │                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:66:22
   │
66 │         sui_address: option::some(preset.account_address()),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:67:32
   │
67 │         protocol_pubkey_bytes: option::some(preset.protocol_pubkey_bytes()),
   │                                ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:68:31
   │
68 │         network_pubkey_bytes: option::some(preset.network_pubkey_bytes()),
   │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:69:30
   │
69 │         worker_pubkey_bytes: option::some(preset.worker_pubkey_bytes()),
   │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:70:30
   │
70 │         proof_of_possession: option::some(preset.proof_of_possession()),
   │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:71:15
   │
71 │         name: option::some(preset.name()),
   │               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:72:22
   │
72 │         description: option::some(preset.description()),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:73:20
   │
73 │         image_url: option::some(preset.image_url()),
   │                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:74:22
   │
74 │         project_url: option::some(preset.project_url()),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:75:22
   │
75 │         net_address: option::some(preset.net_address()),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:76:22
   │
76 │         p2p_address: option::some(preset.p2p_address()),
   │                      ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:77:26
   │
77 │         primary_address: option::some(preset.primary_address()),
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:78:25
   │
78 │         worker_address: option::some(preset.worker_address()),
   │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:79:20
   │
79 │         gas_price: option::none(),
   │                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:80:26
   │
80 │         commission_rate: option::none(),
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/builders/validator_builder.move:82:24
   │
82 │         initial_stake: option::none(),
   │                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:127:37
    │
127 │         initial_stake.map!(|amount| balance::create_for_testing<SUI>(amount * 1_000_000_000)),
    │                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:165:9
    │
165 │         url::new_unsafe_from_bytes(image_url.destroy_or!(b"image_url")),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:166:9
    │
166 │         url::new_unsafe_from_bytes(project_url.destroy_or!(b"project_url")),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:171:9
    │
171 │         bag::new(ctx),
    │         ^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:178:27
    │
178 │     builder.sui_address = option::some(sui_address);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:186:37
    │
186 │     builder.protocol_pubkey_bytes = option::some(protocol_pubkey_bytes);
    │                                     ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:194:36
    │
194 │     builder.network_pubkey_bytes = option::some(network_pubkey_bytes);
    │                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:202:35
    │
202 │     builder.worker_pubkey_bytes = option::some(worker_pubkey_bytes);
    │                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:210:35
    │
210 │     builder.proof_of_possession = option::some(proof_of_possession);
    │                                   ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:215:20
    │
215 │     builder.name = option::some(name);
    │                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:220:27
    │
220 │     builder.description = option::some(description);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:225:25
    │
225 │     builder.image_url = option::some(image_url);
    │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:230:27
    │
230 │     builder.project_url = option::some(project_url);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:235:27
    │
235 │     builder.net_address = option::some(net_address);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:240:27
    │
240 │     builder.p2p_address = option::some(p2p_address);
    │                           ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:248:31
    │
248 │     builder.primary_address = option::some(primary_address);
    │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:256:30
    │
256 │     builder.worker_address = option::some(worker_address);
    │                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:261:25
    │
261 │     builder.gas_price = option::some(gas_price);
    │                         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:266:31
    │
266 │     builder.commission_rate = option::some(commission_rate);
    │                               ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/builders/validator_builder.move:272:29
    │
272 │     builder.initial_stake = option::some(initial_stake);
    │                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/delegation_tests.move:7:5
  │
7 │ use std::unit_test::assert_eq;
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/delegation_tests.move:8:5
  │
8 │ use sui::table::Table;
  │     ^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::table'

error: unexpected name in this position
   ┌─ ./tests/delegation_tests.move:72:26
   │
72 │     runner.advance_epoch(option::none()).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:143:26
    │
143 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:159:26
    │
159 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:198:26
    │
198 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:208:9
    │
208 │         option::some(runner.advance_epoch_opts().computation_charge(80))
    │         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:210:9
    │
210 │         option::none()
    │         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:258:26
    │
258 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:264:26
    │
264 │     runner.advance_epoch(option::some(options)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:307:26
    │
307 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:313:26
    │
313 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:343:26
    │
343 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:347:26
    │
347 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:410:26
    │
410 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:433:26
    │
433 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:449:26
    │
449 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:463:26
    │
463 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:499:26
    │
499 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:504:26
    │
504 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:510:26
    │
510 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:555:26
    │
555 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:591:26
    │
591 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:597:26
    │
597 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:598:26
    │
598 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:599:26
    │
599 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:660:26
    │
660 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:676:26
    │
676 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:716:26
    │
716 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/delegation_tests.move:721:26
    │
721 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/governance_test_utils.move:8:5
  │
8 │ use std::unit_test::{assert_eq, destroy};
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/governance_test_utils.move:9:5
  │
9 │ use sui::balance::{Self, Balance};
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./tests/governance_test_utils.move:10:5
   │
10 │ use sui::coin::{Self, Coin};
   │     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'

error: unbound module
   ┌─ ./tests/governance_test_utils.move:11:5
   │
11 │ use sui::sui::SUI;
   │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unbound module
   ┌─ ./tests/governance_test_utils.move:12:5
   │
12 │ use sui::test_scenario::{Self, Scenario};
   │     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'

error: unexpected name in this position
   ┌─ ./tests/governance_test_utils.move:40:9
   │
40 │         option::some(balance::create_for_testing<SUI>(init_stake_amount_in_sui * MIST_PER_SUI)),
   │         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/governance_test_utils.move:40:22
   │
40 │         option::some(balance::create_for_testing<SUI>(init_stake_amount_in_sui * MIST_PER_SUI)),
   │                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/governance_test_utils.move:67:9
   │
67 │         balance::create_for_testing<SUI>(sui_supply_amount * MIST_PER_SUI), // sui_supply
   │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/governance_test_utils.move:75:9
   │
75 │         object::new(ctx), // it doesn't matter what ID sui system state has in tests
   │         ^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/governance_test_utils.move:77:9
   │
77 │         balance::create_for_testing<SUI>(storage_fund_amount * MIST_PER_SUI), // storage_fund
   │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/governance_test_utils.move:87:24
   │
87 │     let mut scenario = test_scenario::begin(@0x0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:130:5
    │
130 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:175:5
    │
175 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:186:9
    │
186 │         coin::mint_for_testing(amount * MIST_PER_SUI, ctx),
    │         ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:190:5
    │
190 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:201:5
    │
201 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:236:9
    │
236 │         coin::mint_for_testing<SUI>(init_stake_amount * MIST_PER_SUI, ctx),
    │         ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:241:5
    │
241 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:274:5
    │
274 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:283:5
    │
283 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:292:5
    │
292 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:302:5
    │
302 │     test_scenario::return_shared(system_state);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:323:9
    │
323 │         test_scenario::return_shared(system_state);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:342:9
    │
342 │         test_scenario::return_shared(system_state);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/governance_test_utils.move:361:9
    │
361 │         test_scenario::return_shared(system_state);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/rewards_distribution_tests.move:7:5
  │
7 │ use std::unit_test::assert_eq;
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/rewards_distribution_tests.move:8:5
  │
8 │ use sui::address;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::address'

error: unexpected name in this position
   ┌─ ./tests/rewards_distribution_tests.move:39:26
   │
39 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/rewards_distribution_tests.move:52:26
   │
52 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/rewards_distribution_tests.move:78:26
   │
78 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:108:26
    │
108 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:127:26
    │
127 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:141:26
    │
141 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:157:26
    │
157 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:188:26
    │
188 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:193:26
    │
193 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:200:26
    │
200 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:227:24
    │
227 │         .advance_epoch(option::none())
    │                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:240:26
    │
240 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:266:24
    │
266 │         .advance_epoch(option::none())
    │                        ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:270:26
    │
270 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:321:26
    │
321 │     runner.advance_epoch(option::none()).destroy_zero();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:322:26
    │
322 │     runner.advance_epoch(option::none()).destroy_zero();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:348:26
    │
348 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:355:26
    │
355 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:393:26
    │
393 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:403:26
    │
403 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:446:26
    │
446 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:451:26
    │
451 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:465:26
    │
465 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:520:26
    │
520 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:554:26
    │
554 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:560:26
    │
560 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:567:26
    │
567 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:574:26
    │
574 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:603:26
    │
603 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/rewards_distribution_tests.move:616:22
    │
616 │     let validators = vector::tabulate!(num_validators, |i| {
    │                      ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:616:22
    │
616 │     let validators = vector::tabulate!(num_validators, |i| {
    │                      ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:619:26
    │
619 │             .sui_address(address::from_u256(i as u256))
    │                          ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:626:26
    │
626 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:631:24
    │
631 │             let addr = address::from_u256(i as u256);
    │                        ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:668:26
    │
668 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:683:26
    │
683 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:725:26
    │
725 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:740:26
    │
740 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:774:26
    │
774 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:839:26
    │
839 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:846:19
    │
846 │         pool_id = object::id(pool);
    │                   ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:859:26
    │
859 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:904:26
    │
904 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:932:26
    │
932 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:973:26
    │
973 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/rewards_distribution_tests.move:982:19
    │
982 │         pool_id = object::id(pool);
    │                   ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
     ┌─ ./tests/rewards_distribution_tests.move:1024:26
     │
1024 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
     │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/staking_pool_tests.move:7:5
  │
7 │ use std::unit_test::{assert_eq, destroy};
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/staking_pool_tests.move:8:5
  │
8 │ use sui::balance;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./tests/staking_pool_tests.move:9:5
  │
9 │ use sui::test_scenario::{Self, Scenario};
  │     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'

error: unexpected name in this position
   ┌─ ./tests/staking_pool_tests.move:14:24
   │
14 │     let mut scenario = test_scenario::begin(@0x0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/staking_pool_tests.move:38:24
   │
38 │     let mut scenario = test_scenario::begin(@0x0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/staking_pool_tests.move:58:24
   │
58 │     let mut scenario = test_scenario::begin(@0x0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/staking_pool_tests.move:80:24
   │
80 │     let mut scenario = test_scenario::begin(@0x0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/staking_pool_tests.move:95:24
   │
95 │     let mut scenario = test_scenario::begin(@0x0);
   │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/staking_pool_tests.move:98:15
   │
98 │     let sui = balance::create_for_testing(1_000_000_000);
   │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:114:24
    │
114 │     let mut scenario = test_scenario::begin(@0x0);
    │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:117:15
    │
117 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:136:24
    │
136 │     let mut scenario = test_scenario::begin(@0x0);
    │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:139:15
    │
139 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:159:24
    │
159 │     let mut scenario = test_scenario::begin(@0x0);
    │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:163:15
    │
163 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:180:24
    │
180 │     let mut scenario = test_scenario::begin(@0x0);
    │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:186:15
    │
186 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:199:15
    │
199 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:226:49
    │
226 │     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));
    │                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:237:49
    │
237 │     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));
    │                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:253:20
    │
253 │     let mut test = test_scenario::begin(@0x0);
    │                    ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:257:15
    │
257 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:278:24
    │
278 │     let mut scenario = test_scenario::begin(@0x0);
    │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:284:15
    │
284 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:297:15
    │
297 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:322:49
    │
322 │     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));
    │                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:333:49
    │
333 │     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));
    │                                                 ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:385:24
    │
385 │     let mut scenario = test_scenario::begin(@0x0);
    │                        ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:391:15
    │
391 │     let sui = balance::create_for_testing(1_000_000_000);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:404:15
    │
404 │     let sui = balance::create_for_testing(1_000_000_001);
    │               ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:429:47
    │
429 │     assert_eq!(fungible_staked_sui.pool_id(), object::id(&staking_pool));
    │                                               ^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
    ┌─ ./tests/staking_pool_tests.move:455:9
    │
455 │     use sui::tx_context::epoch;
    │         ^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::tx_context'

error: unbound module
    ┌─ ./tests/staking_pool_tests.move:456:9
    │
456 │     use sui::coin;
    │         ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'

error: unbound module
    ┌─ ./tests/staking_pool_tests.move:457:9
    │
457 │     use sui::sui::SUI;
    │         ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::sui'

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:459:19
    │
459 │     let rewards = coin::mint_for_testing<SUI>(reward_amount, scenario.ctx());
    │                   ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:460:34
    │
460 │     staking_pool.deposit_rewards(coin::into_balance(rewards));
    │                                  ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/staking_pool_tests.move:463:5
    │
463 │     test_scenario::next_epoch(scenario, @0x0);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
   ┌─ ./tests/sui_system_tests.move:11:5
   │
11 │ use std::unit_test::assert_eq;
   │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unexpected name in this position
   ┌─ ./tests/sui_system_tests.move:58:26
   │
58 │     runner.advance_epoch(option::none()).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/sui_system_tests.move:79:26
   │
79 │     runner.advance_epoch(option::none()).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/sui_system_tests.move:86:26
   │
86 │     runner.advance_epoch(option::none()).destroy_for_testing();
   │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:116:9
    │
116 │         transfer::public_transfer(cap, stakee);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:135:9
    │
135 │         transfer::public_transfer(cap, new_stakee);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:163:26
    │
163 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:171:26
    │
171 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:193:9
    │
193 │         transfer::public_transfer(cap, stakee);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:222:9
    │
222 │         transfer::public_transfer(cap, stakee);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:357:26
    │
357 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:381:26
    │
381 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:421:26
    │
421 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:431:26
    │
431 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:441:26
    │
441 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:512:26
    │
512 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/sui_system_tests.move:535:26
    │
535 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/validator_metadata_tests.move:7:5
  │
7 │ use std::unit_test;
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/validator_metadata_tests.move:8:5
  │
8 │ use sui::test_scenario::{Self, Scenario};
  │     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'

error: unbound module
  ┌─ ./tests/validator_metadata_tests.move:9:5
  │
9 │ use sui::url;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'

error: unexpected name in this position
   ┌─ ./tests/validator_metadata_tests.move:21:20
   │
21 │     let ctx = &mut tx_context::dummy();
   │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/validator_metadata_tests.move:27:5
   │
27 │     unit_test::destroy(vector[validator_0, validator_1, validator_2, validator_3]);
   │     ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:327:26
    │
327 │     runner.advance_epoch(option::none()).destroy_zero();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:520:9
    │
520 │         test_scenario::return_shared(system_state);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:551:26
    │
551 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:594:9
    │
594 │         test_scenario::return_shared(system_state);
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:600:26
    │
600 │     runner.advance_epoch(option::some(opts)).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:669:26
    │
669 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:675:26
    │
675 │     runner.advance_epoch(option::none()).destroy_for_testing();
    │                          ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:843:52
    │
843 │         validator.next_epoch_network_address() == &option::some(new_network_address.to_string()),
    │                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:845:52
    │
845 │     assert!(validator.next_epoch_p2p_address() == &option::some(new_p2p_address.to_string()));
    │                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:847:52
    │
847 │         validator.next_epoch_primary_address() == &option::some(new_primary_address.to_string()),
    │                                                    ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:849:55
    │
849 │     assert!(validator.next_epoch_worker_address() == &option::some(new_worker_address.to_string()));
    │                                                       ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:850:62
    │
850 │     assert!(validator.next_epoch_protocol_pubkey_bytes() == &option::some(new_protocol_pub_key), 0);
    │                                                              ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:851:60
    │
851 │     assert!(validator.next_epoch_proof_of_possession() == &option::some(new_pop), 0);
    │                                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:852:60
    │
852 │     assert!(validator.next_epoch_worker_pubkey_bytes() == &option::some(new_worker_pubkey), 0);
    │                                                            ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:853:61
    │
853 │     assert!(validator.next_epoch_network_pubkey_bytes() == &option::some(new_network_pubkey), 0);
    │                                                             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:871:39
    │
871 │     assert!(validator.image_url() == &url::new_unsafe_from_bytes(b"new_image_url"));
    │                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_metadata_tests.move:872:41
    │
872 │     assert!(validator.project_url() == &url::new_unsafe_from_bytes(b"new_project_url"));
    │                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/validator_set_tests.move:7:5
  │
7 │ use std::unit_test::{assert_eq, destroy};
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/validator_set_tests.move:8:5
  │
8 │ use sui::address;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::address'

error: unbound module
  ┌─ ./tests/validator_set_tests.move:9:5
  │
9 │ use sui::balance;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
   ┌─ ./tests/validator_set_tests.move:10:5
   │
10 │ use sui::coin;
   │     ^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::coin'

error: unbound module
   ┌─ ./tests/validator_set_tests.move:11:5
   │
11 │ use sui::test_scenario::{Self, Scenario};
   │     ^^^^^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::test_scenario'

error: unbound module
   ┌─ ./tests/validator_set_tests.move:12:5
   │
12 │ use sui::vec_map;
   │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::vec_map'

error: unexpected name in this position
   ┌─ ./tests/validator_set_tests.move:21:28
   │
21 │     let mut scenario_val = test_scenario::begin(@0x0);
   │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/validator_set_tests.move:49:28
   │
49 │     let mut scenario_val = test_scenario::begin(@0x1);
   │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/validator_set_tests.move:55:13
   │
55 │             coin::mint_for_testing(500 * MIST_PER_SUI, ctx1).into_balance(),
   │             ^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/validator_set_tests.move:58:9
   │
58 │         transfer::public_transfer(stake, @0x1);
   │         ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/validator_set_tests.move:94:28
   │
94 │     let mut scenario_val = test_scenario::begin(@0x0);
   │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:145:28
    │
145 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:154:28
    │
154 │     let mut scenario_val = test_scenario::begin(@0x1);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:160:9
    │
160 │         balance::create_for_testing(MIST_PER_SUI - 1), // 1 MIST lower than the threshold
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:163:5
    │
163 │     transfer::public_transfer(stake, @0x1);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:170:28
    │
170 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:179:28
    │
179 │     let mut scenario_val = test_scenario::begin(@0x1);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:184:9
    │
184 │         balance::create_for_testing(MIST_PER_SUI), // min possible stake
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:187:5
    │
187 │     transfer::public_transfer(stake, @0x1);
    │     ^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:198:28
    │
198 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:250:28
    │
250 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:283:28
    │
283 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:298:28
    │
298 │     let mut scenario_val = test_scenario::begin(@0x1);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:320:28
    │
320 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:335:9
    │
335 │         balance::create_for_testing(3 * MIST_PER_SUI),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:348:28
    │
348 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:361:9
    │
361 │         balance::create_for_testing(4 * MIST_PER_SUI),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:385:28
    │
385 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:399:9
    │
399 │         balance::create_for_testing(4 * MIST_PER_SUI),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:426:28
    │
426 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:440:9
    │
440 │         balance::create_for_testing(4 * MIST_PER_SUI),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:485:28
    │
485 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:499:9
    │
499 │         balance::create_for_testing(4 * MIST_PER_SUI),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:531:28
    │
531 │     let mut scenario_val = test_scenario::begin(@0x0);
    │                            ^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:544:9
    │
544 │         balance::create_for_testing(1000 * MIST_PER_SUI),
    │         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:556:13
    │
556 │             address::from_u256((i + 1 as u256)),
    │             ^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:557:13
    │
557 │             balance::create_for_testing(to_add),
    │             ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:600:9
    │
600 │         option::some(balance::create_for_testing(stake_value)),
    │         ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:600:22
    │
600 │         option::some(balance::create_for_testing(stake_value)),
    │                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:632:13
    │
632 │             option::some(balance::create_for_testing(initial_stake * MIST_PER_SUI))
    │             ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:632:26
    │
632 │             option::some(balance::create_for_testing(initial_stake * MIST_PER_SUI))
    │                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:633:18
    │
633 │         } else { option::none() },
    │                  ^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:653:40
    │
653 │     let mut dummy_computation_reward = balance::zero();
    │                                        ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:654:41
    │
654 │     let mut dummy_storage_fund_reward = balance::zero();
    │                                         ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_set_tests.move:659:14
    │
659 │         &mut vec_map::empty(),
    │              ^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/validator_tests.move:7:5
  │
7 │ use std::unit_test::{assert_eq, destroy};
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: unbound module
  ┌─ ./tests/validator_tests.move:8:5
  │
8 │ use sui::balance;
  │     ^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'sui::balance'

error: unbound module
  ┌─ ./tests/validator_tests.move:9:5
  │
9 │ use sui::url;
  │     ^^^^^^^^ Invalid 'use'. Unbound module: 'sui::url'

error: unexpected name in this position
   ┌─ ./tests/validator_tests.move:84:41
   │
84 │         validator.deposit_stake_rewards(balance::zero());
   │                                         ^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/validator_tests.move:99:20
   │
99 │     let ctx = &mut tx_context::dummy();
   │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:107:20
    │
107 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:119:20
    │
119 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:131:20
    │
131 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:143:20
    │
143 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:153:20
    │
153 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:163:20
    │
163 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:173:20
    │
173 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:183:20
    │
183 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:194:20
    │
194 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:205:20
    │
205 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:228:20
    │
228 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:248:40
    │
248 │     assert_eq!(*validator.image_url(), url::new_unsafe_from_bytes(b"new_image_url"));
    │                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:249:42
    │
249 │     assert_eq!(*validator.project_url(), url::new_unsafe_from_bytes(b"new_proj_url"));
    │                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:298:20
    │
298 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:311:20
    │
311 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:321:20
    │
321 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:331:20
    │
331 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:341:20
    │
341 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:351:20
    │
351 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:361:20
    │
361 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:376:20
    │
376 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:379:49
    │
379 │     validator.update_next_epoch_primary_address(vector::tabulate!(257, |_| 0));
    │                                                 ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:379:49
    │
379 │     validator.update_next_epoch_primary_address(vector::tabulate!(257, |_| 0));
    │                                                 ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:390:20
    │
390 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:393:49
    │
393 │     validator.update_next_epoch_network_address(vector::tabulate!(257, |_| 0));
    │                                                 ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:393:49
    │
393 │     validator.update_next_epoch_network_address(vector::tabulate!(257, |_| 0));
    │                                                 ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:405:20
    │
405 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:408:48
    │
408 │     validator.update_next_epoch_worker_address(vector::tabulate!(257, |_| 0));
    │                                                ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:408:48
    │
408 │     validator.update_next_epoch_worker_address(vector::tabulate!(257, |_| 0));
    │                                                ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:420:20
    │
420 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:423:45
    │
423 │     validator.update_next_epoch_p2p_address(vector::tabulate!(257, |_| 0));
    │                                             ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:423:45
    │
423 │     validator.update_next_epoch_p2p_address(vector::tabulate!(257, |_| 0));
    │                                             ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:435:20
    │
435 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:438:27
    │
438 │     validator.update_name(vector::tabulate!(257, |_| 0));
    │                           ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:438:27
    │
438 │     validator.update_name(vector::tabulate!(257, |_| 0));
    │                           ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:450:20
    │
450 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:453:34
    │
453 │     validator.update_description(vector::tabulate!(257, |_| 0));
    │                                  ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:453:34
    │
453 │     validator.update_description(vector::tabulate!(257, |_| 0));
    │                                  ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:465:20
    │
465 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:468:34
    │
468 │     validator.update_project_url(vector::tabulate!(257, |_| 0));
    │                                  ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:468:34
    │
468 │     validator.update_project_url(vector::tabulate!(257, |_| 0));
    │                                  ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:480:20
    │
480 │     let ctx = &mut tx_context::dummy();
    │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/validator_tests.move:483:32
    │
483 │     validator.update_image_url(vector::tabulate!(257, |_| 0));
    │                                ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/validator_tests.move:483:32
    │
483 │     validator.update_image_url(vector::tabulate!(257, |_| 0));
    │                                ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound module
  ┌─ ./tests/voting_power_tests.move:7:5
  │
7 │ use std::unit_test::{assert_eq, destroy};
  │     ^^^^^^^^^^^^^^ Invalid 'use'. Unbound module: 'std::unit_test'

error: invalid use of reserved name
   ┌─ ./tests/voting_power_tests.move:21:25
   │
21 │     let voting_powers = vector::tabulate!(
   │                         ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
   ┌─ ./tests/voting_power_tests.move:21:25
   │
21 │     let voting_powers = vector::tabulate!(
   │                         ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/voting_power_tests.move:32:20
   │
32 │     let ctx = &mut tx_context::dummy();
   │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/voting_power_tests.move:72:20
   │
72 │     let ctx = &mut tx_context::dummy();
   │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unexpected name in this position
   ┌─ ./tests/voting_power_tests.move:98:20
   │
98 │     let ctx = &mut tx_context::dummy();
   │                    ^^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: invalid use of reserved name
    ┌─ ./tests/voting_power_tests.move:118:5
    │
118 │     vector::tabulate!(stakes.length(), |i| {
    │     ^^^^^^ Invalid address name 'vector'. 'vector' is restricted and cannot be used to name an address

error: unexpected name in this position
    ┌─ ./tests/voting_power_tests.move:118:5
    │
118 │     vector::tabulate!(stakes.length(), |i| {
    │     ^^^^^^^^^^^^^^^^ Unexpected module identifier. A module identifier is not a valid expression

error: unbound type
    ┌─ ./tests/delegation_tests.move:736:36
    │
736 │ use fun assert_exchange_rate_eq as Table.assert_exchange_rate_eq;
    │                                    ^^^^^ Unbound type 'Table' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:603:5
    │
603 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:415:9
    │
415 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:416:9
    │
416 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:417:9
    │
417 │         assert_eq!(
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:438:9
    │
438 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:439:9
    │
439 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:440:9
    │
440 │         assert_eq!(system.validator_stake_amount(NEW_VALIDATOR_ADDR), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:441:9
    │
441 │         assert_eq!(
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:456:5
    │
456 │     assert_eq!(runner.sui_balance(), 110002000000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:459:5
    │
459 │     assert_eq!(runner.sui_balance(), 110002000000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:468:5
    │
468 │     assert_eq!(runner.sui_balance(), 78862939078);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:514:5
    │
514 │     assert_eq!(runner.sui_balance(), 130006000000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:351:5
    │
351 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:138:9
    │
138 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:139:9
    │
139 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:149:13
    │
149 │             assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 160 * MIST_PER_SUI);
    │             ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:150:13
    │
150 │             assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 100 * MIST_PER_SUI);
    │             ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:154:13
    │
154 │             assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 160 * MIST_PER_SUI);
    │             ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:161:9
    │
161 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:545:9
    │
545 │         assert_eq!(validator.total_stake(), 200 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:546:9
    │
546 │         assert_eq!(validator.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:547:9
    │
547 │         assert_eq!(validator.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:552:5
    │
552 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:559:5
    │
559 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:565:9
    │
565 │         assert_eq!(validator.total_stake(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:566:9
    │
566 │         assert_eq!(validator.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:567:9
    │
567 │         assert_eq!(validator.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
    ┌─ ./tests/delegation_tests.move:739:13
    │
739 │     rates: &Table<u64, PoolTokenExchangeRate>,
    │             ^^^^^ Unbound type 'Table' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:745:5
    │
745 │     assert_eq!(rate.sui_amount(), sui_amount * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:746:5
    │
746 │     assert_eq!(rate.pool_token_amount(), pool_token_amount * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:273:9
    │
273 │         assert_eq!(stake.amount(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:281:5
    │
281 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:289:5
    │
289 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt + validator_reward_amt);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:634:5
    │
634 │     assert_eq!(runner.set_sender(validator_address).sui_balance(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:202:9
    │
202 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 200 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:203:9
    │
203 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:224:9
    │
224 │         assert_eq!(stake.amount(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:233:5
    │
233 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:239:5
    │
239 │     assert_eq!(runner.sui_balance(), 100 * MIST_PER_SUI + reward_amt + validator_reward_amt);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/delegation_tests.move:45:9
   │
45 │         assert_eq!(ids.length(), 2);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/delegation_tests.move:50:9
   │
50 │         assert_eq!(stake_1.amount(), 20 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/delegation_tests.move:51:9
   │
51 │         assert_eq!(stake_2.amount(), 40 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/delegation_tests.move:58:9
   │
58 │         assert_eq!(stake.amount(), 60 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/delegation_tests.move:726:9
    │
726 │         assert_eq!(rates.length(), 3);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound module
    ┌─ ./tests/delegation_tests.move:5:1
    │  
  5 │ ╭ module sui_system::delegation_tests;
  6 │ │ 
  7 │ │ use std::unit_test::assert_eq;
  8 │ │ use sui::table::Table;
    · │
746 │ │     assert_eq!(rate.pool_token_amount(), pool_token_amount * MIST_PER_SUI);
747 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
   ┌─ ./sources/genesis.move:57:28
   │
57 │     staked_with_validator: Option<address>,
   │                            ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/genesis.move:179:21
    │
179 │     mut sui_supply: Balance<SUI>,
    │                     ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/genesis.move:182:15
    │
182 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
   ┌─ ./sources/genesis.move:73:26
   │
73 │     sui_system_state_id: UID,
   │                          ^^^ Unbound type 'UID' in current scope

error: unbound type
   ┌─ ./sources/genesis.move:74:21
   │
74 │     mut sui_supply: Balance<SUI>,
   │                     ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/genesis.move:78:15
   │
78 │     ctx: &mut TxContext,
   │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./sources/genesis.move:4:1
    │  
  4 │ ╭ module sui_system::genesis;
  5 │ │ 
  6 │ │ use sui::balance::{Self, Balance};
  7 │ │ use sui::sui::SUI;
    · │
206 │ │     sui_supply.destroy_zero();
207 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./tests/governance_test_utils.move:286:61
    │
286 │ public fun add_validator(validator: address, scenario: &mut Scenario) {
    │                                                             ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:250:20
    │
250 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:211:20
    │
211 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:101:41
    │
101 │ public fun advance_epoch(scenario: &mut Scenario) {
    │                                         ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:138:20
    │
138 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
    ┌─ ./tests/governance_test_utils.move:147:5
    │
147 │     destroy(storage_rebate)
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:154:20
    │
154 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
    ┌─ ./tests/governance_test_utils.move:174:5
    │
174 │     destroy(storage_rebate);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:110:20
    │
110 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:111:4
    │
111 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:350:20
    │
350 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
    ┌─ ./tests/governance_test_utils.move:360:9
    │
360 │         assert_eq!(non_self_stake_amount, amount);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:308:20
    │
308 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
    ┌─ ./tests/governance_test_utils.move:322:9
    │
322 │         assert_eq!(stake_plus_rewards, amount);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:331:20
    │
331 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
   ┌─ ./tests/governance_test_utils.move:53:15
   │
53 │     ctx: &mut TxContext,
   │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
   ┌─ ./tests/governance_test_utils.move:24:15
   │
24 │     ctx: &mut TxContext,
   │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:295:64
    │
295 │ public fun remove_validator(validator: address, scenario: &mut Scenario) {
    │                                                                ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:277:74
    │
277 │ public fun remove_validator_candidate(validator: address, scenario: &mut Scenario) {
    │                                                                          ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:380:20
    │
380 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:370:20
    │
370 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:179:88
    │
179 │ public fun stake_with(staker: address, validator: address, amount: u64, scenario: &mut Scenario) {
    │                                                                                        ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:397:60
    │
397 │ public fun total_sui_balance(addr: address, scenario: &mut Scenario): u64 {
    │                                                            ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:400:44
    │
400 │     let coin_ids = scenario.ids_for_sender<Coin<SUI>>();
    │                                            ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./tests/governance_test_utils.move:403:52
    │
403 │         let coin = scenario.take_from_sender_by_id<Coin<SUI>>(coin_ids[i]);
    │                                                    ^^^^ Unbound type 'Coin' in current scope

error: unbound module
    ┌─ ./tests/governance_test_utils.move:6:1
    │  
  6 │ ╭ module sui_system::governance_test_utils;
  7 │ │ 
  8 │ │ use std::unit_test::{assert_eq, destroy};
  9 │ │ use sui::balance::{Self, Balance};
    · │
408 │ │     sum
409 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./tests/governance_test_utils.move:193:73
    │
193 │ public fun unstake(staker: address, staked_sui_idx: u64, scenario: &mut Scenario) {
    │                                                                         ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1057:9
     │
1057 │         assert_eq!(sum_rewards, expected_amount);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:312:9
    │
312 │         assert_eq!(validator.commission_rate(), 100);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:316:9
    │
316 │         assert_eq!(validator.commission_rate(), 101);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:326:9
    │
326 │         assert_eq!(validator.commission_rate(), 101);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
     ┌─ ./tests/rewards_distribution_tests.move:1043:77
     │
1043 │ fun check_distribution_counter_invariant(system: &mut SuiSystemState, ctx: &TxContext) {
     │                                                                             ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1044:5
     │
1044 │     assert_eq!(ctx.epoch(), system.epoch());
     │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1046:5
     │
1046 │     assert_eq!(system.get_stake_subsidy_distribution_counter() + 20, ctx.epoch());
     │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:425:5
    │
425 │     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), (550 + 150) * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:426:5
    │
426 │     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:531:9
    │
531 │         assert_eq!(system.get_storage_fund_total_balance(), 4000 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:533:9
    │
533 │         assert_eq!(system.get_storage_fund_object_rebates(), 1000 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:578:9
    │
578 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 140 * 23 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:591:5
    │
591 │     assert_eq!(
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:596:5
    │
596 │     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), (480 + 40 * 2) * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:599:5
    │
599 │     assert_eq!(runner.set_sender(STAKER_ADDR_3).sui_balance(), (390 + 280 + 30) * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:601:5
    │
601 │     assert_eq!(runner.set_sender(STAKER_ADDR_4).sui_balance(), 1400 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:607:9
    │
607 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 140 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:805:9
    │
805 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:806:9
    │
806 │         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:816:9
    │
816 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:817:9
    │
817 │         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:818:9
    │
818 │         assert_eq!(pool.pending_stake_amount(), 101 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:830:9
    │
830 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:831:9
    │
831 │         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:832:9
    │
832 │         assert_eq!(pool.pending_stake_amount(), 101 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:833:9
    │
833 │         assert_eq!(pool.pending_stake_withdraw_amount(), 101 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:834:9
    │
834 │         assert_eq!(pool.pending_pool_token_withdraw_amount(), 101 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
    ┌─ ./tests/rewards_distribution_tests.move:842:18
    │
842 │     let pool_id: ID;
    │                  ^^ Unbound type 'ID' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:848:9
    │
848 │         assert_eq!(pool.pending_stake_amount(), 0 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:849:9
    │
849 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:850:9
    │
850 │         assert_eq!(pool.pending_pool_token_withdraw_amount(), 0 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:851:9
    │
851 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:852:9
    │
852 │         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:868:9
    │
868 │         assert_eq!(pool.sui_balance(), 0 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:869:9
    │
869 │         assert_eq!(pool.pool_token_balance(), 0 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:893:9
    │
893 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:894:9
    │
894 │         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:906:9
    │
906 │         assert_eq!(system.epoch(), 1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:909:9
    │
909 │         assert_eq!(pool.sui_balance(), 125 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:910:9
    │
910 │         assert_eq!(pool.pool_token_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:937:9
    │
937 │         assert_eq!(exchange_rate.sui_amount(), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:941:9
    │
941 │         assert_eq!(exchange_rate.pool_token_amount(), 166666666666);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:942:9
    │
942 │         assert_eq!(pool.sui_balance(), 250 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:943:9
    │
943 │         assert_eq!(pool.pool_token_balance(), 166666666666);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:948:9
    │
948 │         assert_eq!(stake.stake_activation_epoch(), 3);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:960:13
    │
960 │             assert_eq!(pool.pending_pool_token_withdraw_amount(), 80 * MIST_PER_SUI);
    │             ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:963:13
    │
963 │             assert_eq!(pool.pending_stake_withdraw_amount(), 120 * MIST_PER_SUI); // 100 principal + 20 rewards
    │             ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
    ┌─ ./tests/rewards_distribution_tests.move:978:18
    │
978 │     let pool_id: ID;
    │                  ^^ Unbound type 'ID' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:984:9
    │
984 │         assert_eq!(pool.sui_balance(), 155 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:985:9
    │
985 │         assert_eq!(pool.pending_stake_withdraw_amount(), 155 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:994:9
    │
994 │         assert_eq!(pool.pending_stake_withdraw_amount(), 155 * MIST_PER_SUI); // 100 principal + 55 rewards
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:997:9
    │
997 │         assert_eq!(exchange_rate_epoch_0.sui_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:998:9
    │
998 │         assert_eq!(exchange_rate_epoch_0.pool_token_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1000:9
     │
1000 │         assert_eq!(exchange_rate_epoch_1.sui_amount(), 125000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1001:9
     │
1001 │         assert_eq!(exchange_rate_epoch_1.pool_token_amount(), 100000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1006:9
     │
1006 │         assert_eq!(exchange_rate_epoch_5.sui_amount(), 250000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1007:9
     │
1007 │         assert_eq!(exchange_rate_epoch_5.pool_token_amount(), 166666666666);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1009:9
     │
1009 │         assert_eq!(exchange_rate_epoch_6.sui_amount(), 155000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1010:9
     │
1010 │         assert_eq!(exchange_rate_epoch_6.pool_token_amount(), 86666666666);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1013:9
     │
1013 │         assert_eq!(pool.sui_balance(), 155000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1014:9
     │
1014 │         assert_eq!(pool.pending_stake_withdraw_amount(), 155000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1015:9
     │
1015 │         assert_eq!(pool.pool_token_balance(), 86666666666);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1016:9
     │
1016 │         assert_eq!(pool.pending_pool_token_withdraw_amount(), 100000000000);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1032:9
     │
1032 │         assert_eq!(validator.pending_stake_amount(), 0);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1033:9
     │
1033 │         assert_eq!(validator.pending_stake_withdraw_amount(), 0);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1034:9
     │
1034 │         assert_eq!(validator.get_staking_pool_ref().pending_pool_token_withdraw_amount(), 0);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1035:9
     │
1035 │         assert_eq!(validator.get_staking_pool_ref().sui_balance(), 0);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
     ┌─ ./tests/rewards_distribution_tests.move:1036:9
     │
1036 │         assert_eq!(validator.get_staking_pool_ref().pool_token_balance(), 0);
     │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:371:5
    │
371 │     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), 565 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:372:5
    │
372 │     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 370 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:483:5
    │
483 │     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), (100 + 80) * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:485:5
    │
485 │     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), (100 + 48) * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:113:9
    │
113 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 300 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:114:9
    │
114 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 300 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:115:9
    │
115 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 300 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:116:9
    │
116 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 400 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:150:5
    │
150 │     assert_eq!(runner.set_sender(STAKER_ADDR_1).sui_balance(), 220 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:154:5
    │
154 │     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 120 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:168:5
    │
168 │     assert_eq!(runner.set_sender(STAKER_ADDR_2).sui_balance(), 728108108107);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:81:9
   │
81 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 100_000_025 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:82:9
   │
82 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 200_000_025 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:83:9
   │
83 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 300_000_025 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:84:9
   │
84 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 400_000_025 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:661:9
    │
661 │         assert_eq!(ctx.epoch(), 562);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:662:9
    │
662 │         assert_eq!(system.epoch(), 562);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:663:9
    │
663 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:672:9
    │
672 │         assert_eq!(ctx.epoch(), 563);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:673:9
    │
673 │         assert_eq!(system.epoch(), 563);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:674:9
    │
674 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 3);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:687:9
    │
687 │         assert_eq!(ctx.epoch(), 564);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:688:9
    │
688 │         assert_eq!(system.epoch(), 564);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:689:9
    │
689 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:718:9
    │
718 │         assert_eq!(ctx.epoch(), 563);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:719:9
    │
719 │         assert_eq!(system.epoch(), 563);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:720:9
    │
720 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:729:9
    │
729 │         assert_eq!(ctx.epoch(), 564);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:730:9
    │
730 │         assert_eq!(system.epoch(), 564);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:731:9
    │
731 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:744:9
    │
744 │         assert_eq!(ctx.epoch(), 565);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:745:9
    │
745 │         assert_eq!(system.epoch(), 565);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:746:9
    │
746 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), start_distribution_counter + 5);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:767:9
    │
767 │         assert_eq!(ctx.epoch(), 540);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:768:9
    │
768 │         assert_eq!(system.epoch(), 540);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:769:9
    │
769 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), 540);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:778:9
    │
778 │         assert_eq!(ctx.epoch(), 541);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:779:9
    │
779 │         assert_eq!(system.epoch(), 541);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:780:9
    │
780 │         assert_eq!(system.get_stake_subsidy_distribution_counter(), 541);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:632:13
    │
632 │             assert_eq!(system.validator_stake_amount(addr), (962 + i * 4) * MIST_PER_SUI);
    │             ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound module
     ┌─ ./tests/rewards_distribution_tests.move:5:1
     │  
   5 │ ╭ module sui_system::rewards_distribution_tests;
   6 │ │ 
   7 │ │ use std::unit_test::assert_eq;
   8 │ │ use sui::address;
     · │
1058 │ │     });
1059 │ │ }
     │ ╰─^ Unbound module 'std::unit_test'

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:233:9
    │
233 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 200 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:234:9
    │
234 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 300 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:235:9
    │
235 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 300 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:236:9
    │
236 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 400 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:244:9
    │
244 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 230 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:245:9
    │
245 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 330 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:246:9
    │
246 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 330 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:247:9
    │
247 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 430 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:272:9
    │
272 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 290 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:273:9
    │
273 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 390 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:274:9
    │
274 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 390 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/rewards_distribution_tests.move:275:9
    │
275 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 490 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:43:9
   │
43 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 125 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:44:9
   │
44 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 225 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:45:9
   │
45 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 325 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:46:9
   │
46 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 425 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:56:9
   │
56 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_1), 150 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:57:9
   │
57 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_2), 970 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:58:9
   │
58 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_3), 350 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/rewards_distribution_tests.move:59:9
   │
59 │         assert_eq!(system.validator_stake_amount(VALIDATOR_ADDR_4), 450 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
   ┌─ ./sources/stake_subsidy.move:16:14
   │
16 │     balance: Balance<SUI>,
   │              ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/stake_subsidy.move:28:19
   │
28 │     extra_fields: Bag,
   │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
   ┌─ ./sources/stake_subsidy.move:55:61
   │
55 │ public(package) fun advance_epoch(self: &mut StakeSubsidy): Balance<SUI> {
   │                                                             ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/stake_subsidy.move:32:14
   │
32 │     balance: Balance<SUI>,
   │              ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/stake_subsidy.move:36:15
   │
36 │     ctx: &mut TxContext,
   │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
   ┌─ ./sources/stake_subsidy.move:4:1
   │  
 4 │ ╭ module sui_system::stake_subsidy;
 5 │ │ 
 6 │ │ use sui::bag::{Self, Bag};
 7 │ │ use sui::balance::Balance;
   · │
90 │ │     self.distribution_counter = distribution_counter;
91 │ │ }
   │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
   ┌─ ./sources/staking_pool.move:90:9
   │
90 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:92:14
   │
92 │     pool_id: ID,
   │              ^^ Unbound type 'ID' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:99:9
   │
99 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:103:16
    │
103 │     principal: Balance<SUI>,
    │                ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:76:9
   │
76 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:78:14
   │
78 │     pool_id: ID,
   │              ^^ Unbound type 'ID' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:82:16
   │
82 │     principal: Balance<SUI>,
   │                ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:42:23
   │
42 │     activation_epoch: Option<u64>,
   │                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:45:25
   │
45 │     deactivation_epoch: Option<u64>,
   │                         ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:56:21
   │
56 │     exchange_rates: Table<u64, PoolTokenExchangeRate>,
   │                     ^^^^^ Unbound type 'Table' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:65:19
   │
65 │     extra_fields: Bag,
   │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:39:9
   │
39 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
   ┌─ ./sources/staking_pool.move:50:19
   │
50 │     rewards_pool: Balance<SUI>,
   │                   ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:500:59
    │
500 │ public(package) fun activation_epoch(pool: &StakingPool): Option<u64> {
    │                                                           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:277:15
    │
277 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:745:15
    │
745 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:350:70
    │
350 │ public(package) fun deposit_rewards(pool: &mut StakingPool, rewards: Balance<SUI>) {
    │                                                                      ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:630:58
    │
630 │ public(package) fun exchange_rates(pool: &StakingPool): &Table<u64, PoolTokenExchangeRate> {
    │                                                          ^^^^^ Unbound type 'Table' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:475:82
    │
475 │ public fun fungible_staked_sui_pool_id(fungible_staked_sui: &FungibleStakedSui): ID {
    │                                                                                  ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:116:35
    │
116 │ public(package) fun new(ctx: &mut TxContext): StakingPool {
    │                                   ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:471:45
    │
471 │ public fun pool_id(staked_sui: &StakedSui): ID { staked_sui.pool_id }
    │                                             ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:355:88
    │
355 │ public(package) fun process_pending_stakes_and_withdraws(pool: &mut StakingPool, ctx: &TxContext) {
    │                                                                                        ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:198:11
    │
198 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:199:4
    │
199 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:137:12
    │
137 │     stake: Balance<SUI>,
    │            ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:139:15
    │
139 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:160:11
    │
160 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:161:4
    │
161 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:547:69
    │
547 │ public fun split(self: &mut StakedSui, split_amount: u64, ctx: &mut TxContext): StakedSui {
    │                                                                     ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:520:15
    │
520 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:568:87
    │
568 │ public entry fun split_staked_sui(stake: &mut StakedSui, split_amount: u64, ctx: &mut TxContext) {
    │                                                                                       ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./sources/staking_pool.move:5:1
    │  
  5 │ ╭ module sui_system::staking_pool;
  6 │ │ 
  7 │ │ use sui::bag::{Self, Bag};
  8 │ │ use sui::balance::{Self, Balance};
    · │
836 │ │     assert!(principal_amount + rewards_amount >= min_out, 0);
837 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/staking_pool.move:341:47
    │
341 │ fun unwrap_staked_sui(staked_sui: StakedSui): Balance<SUI> {
    │                                               ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:327:10
    │
327 │ ): (u64, Balance<SUI>) {
    │          ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/staking_pool.move:429:4
    │
429 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:193:5
    │
193 │     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut scenario, 0), 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:196:5
    │
196 │     assert_eq!(latest_exchange_rate.sui_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:197:5
    │
197 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:206:5
    │
206 │     assert_eq!(
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:216:5
    │
216 │     assert_eq!(latest_exchange_rate.sui_amount(), 3_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:217:5
    │
217 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:225:5
    │
225 │     assert_eq!(fungible_staked_sui_1.value(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:226:5
    │
226 │     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:229:5
    │
229 │     assert_eq!(fungible_staked_sui_data.total_supply(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:230:5
    │
230 │     assert_eq!(fungible_staked_sui_data.principal_value(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:236:5
    │
236 │     assert_eq!(fungible_staked_sui_2.value(), 500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:237:5
    │
237 │     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:240:5
    │
240 │     assert_eq!(fungible_staked_sui_data.total_supply(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:241:5
    │
241 │     assert_eq!(fungible_staked_sui_data.principal_value(), 2_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:243:5
    │
243 │     destroy(staking_pool);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:245:5
    │
245 │     destroy(fungible_staked_sui_1);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:246:5
    │
246 │     destroy(fungible_staked_sui_2);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
    ┌─ ./tests/staking_pool_tests.move:452:20
    │
452 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:28:5
   │
28 │     assert_eq!(fungible_staked_sui_1.value(), 300_000_000_000);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:30:5
   │
30 │     destroy(staking_pool);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:31:5
   │
31 │     destroy(fungible_staked_sui_1);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:291:5
    │
291 │     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut scenario, 0), 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:294:5
    │
294 │     assert_eq!(latest_exchange_rate.sui_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:295:5
    │
295 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:304:5
    │
304 │     assert_eq!(
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:314:5
    │
314 │     assert_eq!(latest_exchange_rate.sui_amount(), 3_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:315:5
    │
315 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:321:5
    │
321 │     assert_eq!(fungible_staked_sui_1.value(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:322:5
    │
322 │     assert_eq!(fungible_staked_sui_1.pool_id(), object::id(&staking_pool));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:325:5
    │
325 │     assert_eq!(fungible_staked_sui_data.total_supply(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:326:5
    │
326 │     assert_eq!(fungible_staked_sui_data.principal_value(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:332:5
    │
332 │     assert_eq!(fungible_staked_sui_2.value(), 500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:333:5
    │
333 │     assert_eq!(fungible_staked_sui_2.pool_id(), object::id(&staking_pool));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:336:5
    │
336 │     assert_eq!(fungible_staked_sui_data.total_supply(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:337:5
    │
337 │     assert_eq!(fungible_staked_sui_data.principal_value(), 2_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:340:5
    │
340 │     assert_eq!(
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:350:5
    │
350 │     assert_eq!(latest_exchange_rate.sui_amount(), 6_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:351:5
    │
351 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:353:5
    │
353 │     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:354:5
    │
354 │     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:357:5
    │
357 │     assert_eq!(sui_1.value(), 4_000_000_000 - 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:360:5
    │
360 │     assert_eq!(fungible_staked_sui_data.total_supply(), 500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:361:5
    │
361 │     assert_eq!(fungible_staked_sui_data.principal_value(), 2_000_000_000 / 3 + 1); // round against user
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:363:5
    │
363 │     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 4_000_000_000 - 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:364:5
    │
364 │     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:367:5
    │
367 │     assert_eq!(sui_2.value(), 2_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:370:5
    │
370 │     assert_eq!(fungible_staked_sui_data.total_supply(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:371:5
    │
371 │     assert_eq!(fungible_staked_sui_data.principal_value(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:373:5
    │
373 │     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 6_000_000_000 - 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:374:5
    │
374 │     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:376:5
    │
376 │     destroy(staking_pool);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:377:5
    │
377 │     destroy(sui_1);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:378:5
    │
378 │     destroy(sui_2);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:398:5
    │
398 │     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut scenario, 0), 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:401:5
    │
401 │     assert_eq!(latest_exchange_rate.sui_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:402:5
    │
402 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:411:5
    │
411 │     assert_eq!(
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:421:5
    │
421 │     assert_eq!(latest_exchange_rate.sui_amount(), 3_000_000_001);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:422:5
    │
422 │     assert_eq!(latest_exchange_rate.pool_token_amount(), 1_500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:428:5
    │
428 │     assert_eq!(fungible_staked_sui.value(), 500_000_000); // rounding!
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:429:5
    │
429 │     assert_eq!(fungible_staked_sui.pool_id(), object::id(&staking_pool));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:432:5
    │
432 │     assert_eq!(fungible_staked_sui_data.total_supply(), 500_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:433:5
    │
433 │     assert_eq!(fungible_staked_sui_data.principal_value(), 1_000_000_001);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:437:5
    │
437 │     assert_eq!(sui.value(), 1_000_000_000);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:440:5
    │
440 │     assert_eq!(fungible_staked_sui_data.total_supply(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:441:5
    │
441 │     assert_eq!(fungible_staked_sui_data.principal_value(), 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:443:5
    │
443 │     destroy(staking_pool);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:444:5
    │
444 │     destroy(staked_sui_1);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:445:5
    │
445 │     destroy(sui);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:68:5
   │
68 │     assert_eq!(fungible_staked_sui_1.value(), 25_000_000_000);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:69:5
   │
69 │     assert_eq!(fungible_staked_sui_2.value(), 75_000_000_000);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:71:5
   │
71 │     destroy(staking_pool);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:72:5
   │
72 │     destroy(fungible_staked_sui_1);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
   ┌─ ./tests/staking_pool_tests.move:73:5
   │
73 │     destroy(fungible_staked_sui_2);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:259:5
    │
259 │     assert_eq!(distribute_rewards_and_advance_epoch(&mut staking_pool, &mut test, 0), 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:266:5
    │
266 │     assert_eq!(staking_pool.sui_balance(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:267:5
    │
267 │     assert_eq!(staking_pool.pending_stake_withdraw_amount(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:268:5
    │
268 │     assert_eq!(staking_pool.pool_token_balance(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:269:5
    │
269 │     assert_eq!(staking_pool.pending_pool_token_withdraw_amount(), 0);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:271:5
    │
271 │     destroy(staking_pool);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/staking_pool_tests.move:272:5
    │
272 │     destroy(staked_sui_1);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound module
    ┌─ ./tests/staking_pool_tests.move:5:1
    │  
  5 │ ╭ module sui_system::staking_pool_tests;
  6 │ │ 
  7 │ │ use std::unit_test::{assert_eq, destroy};
  8 │ │ use sui::balance;
    · │
465 │ │     scenario.ctx().epoch()
466 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
   ┌─ ./sources/storage_fund.move:19:29
   │
19 │     non_refundable_balance: Balance<SUI>,
   │                             ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/storage_fund.move:18:35
   │
18 │     total_object_storage_rebates: Balance<SUI>,
   │                                   ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/storage_fund.move:34:22
   │
34 │     storage_charges: Balance<SUI>,
   │                      ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/storage_fund.move:35:32
   │
35 │     storage_fund_reinvestment: Balance<SUI>,
   │                                ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/storage_fund.move:36:31
   │
36 │     leftover_staking_rewards: Balance<SUI>,
   │                               ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/storage_fund.move:39:4
   │
39 │ ): Balance<SUI> {
   │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/storage_fund.move:23:39
   │
23 │ public(package) fun new(initial_fund: Balance<SUI>): StorageFund {
   │                                       ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound module
   ┌─ ./sources/storage_fund.move:4:1
   │  
 4 │ ╭ module sui_system::storage_fund;
 5 │ │ 
 6 │ │ use sui::balance::{Self, Balance};
 7 │ │ use sui::sui::SUI;
   · │
70 │ │     self.total_object_storage_rebates.value() + self.non_refundable_balance.value()
71 │ │ }
   │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
   ┌─ ./sources/sui_system.move:67:9
   │
67 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:556:70
    │
556 │ public fun active_validator_voting_powers(wrapper: &SuiSystemState): VecMap<address, u64> {
    │                                                                      ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:590:21
    │
590 │     storage_reward: Balance<SUI>,
    │                     ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:591:25
    │
591 │     computation_reward: Balance<SUI>,
    │                         ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:601:15
    │
601 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:602:4
    │
602 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:890:15
    │
890 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:891:4
    │
891 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:570:11
    │
570 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:280:15
    │
280 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
   ┌─ ./sources/sui_system.move:79:9
   │
79 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
   ┌─ ./sources/sui_system.move:81:19
   │
81 │     storage_fund: Balance<SUI>,
   │                   ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
   ┌─ ./sources/sui_system.move:86:15
   │
86 │     ctx: &mut TxContext,
   │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:756:75
    │
756 │ public fun get_reporters_of(wrapper: &mut SuiSystemState, addr: address): VecSet<address> {
    │                                                                           ^^^^^^ Unbound type 'VecSet' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:540:15
    │
540 │     pool_id: &ID,
    │               ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:541:5
    │
541 │ ): &Table<u64, PoolTokenExchangeRate> {
    │     ^^^^^ Unbound type 'Table' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:289:11
    │
289 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:290:4
    │
290 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:231:12
    │
231 │     stake: Coin<SUI>,
    │            ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:233:15
    │
233 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:253:20
    │
253 │     stakes: vector<Coin<SUI>>,
    │                    ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:256:15
    │
256 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:242:12
    │
242 │     stake: Coin<SUI>,
    │            ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:244:15
    │
244 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:169:80
    │
169 │ public entry fun request_add_validator(wrapper: &mut SuiSystemState, ctx: &mut TxContext) {
    │                                                                                ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:131:15
    │
131 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:854:15
    │
854 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:805:82
    │
805 │ public fun request_add_validator_for_testing(wrapper: &mut SuiSystemState, ctx: &TxContext) {
    │                                                                                  ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:179:83
    │
179 │ public entry fun request_remove_validator(wrapper: &mut SuiSystemState, ctx: &mut TxContext) {
    │                                                                                   ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:159:15
    │
159 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:210:15
    │
210 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:270:15
    │
270 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:298:15
    │
298 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:299:4
    │
299 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:336:76
    │
336 │ public entry fun rotate_operation_cap(self: &mut SuiSystemState, ctx: &mut TxContext) {
    │                                                                            ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:220:15
    │
220 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./sources/sui_system.move:40:1
    │  
 40 │ ╭ module sui_system::sui_system;
 41 │ │ 
 42 │ │ use sui::balance::Balance;
 43 │ │ use sui::coin::Coin;
    · │
907 │ │     storage_rebate
908 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/sui_system.move:396:11
    │
396 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:528:11
    │
528 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:417:11
    │
417 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:438:11
    │
438 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:484:11
    │
484 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:459:11
    │
459 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:507:11
    │
507 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:355:11
    │
355 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:365:11
    │
365 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:345:11
    │
345 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:386:11
    │
386 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:518:11
    │
518 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:407:11
    │
407 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:428:11
    │
428 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:471:11
    │
471 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:449:11
    │
449 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:497:11
    │
497 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:375:11
    │
375 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:533:81
    │
533 │ public fun validator_address_by_pool_id(wrapper: &mut SuiSystemState, pool_id: &ID): address {
    │                                                                                 ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:744:94
    │
744 │ public fun validator_staking_pool_id(wrapper: &mut SuiSystemState, validator_addr: address): ID {
    │                                                                                              ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:750:76
    │
750 │ public fun validator_staking_pool_mappings(wrapper: &mut SuiSystemState): &Table<ID, address> {
    │                                                                            ^^^^^ Unbound type 'Table' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:660:60
    │
660 │ fun validator_voting_powers(wrapper: &mut SuiSystemState): VecMap<address, u64> {
    │                                                            ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:717:79
    │
717 │ public fun validator_voting_powers_for_testing(wrapper: &mut SuiSystemState): VecMap<address, u64> {
    │                                                                               ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/sui_system.move:703:11
    │
703 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:149:19
    │
149 │     extra_fields: Bag,
    │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:143:36
    │
143 │     safe_mode_computation_rewards: Balance<SUI>,
    │                                    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:142:32
    │
142 │     safe_mode_storage_rewards: Balance<SUI>,
    │                                ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:132:31
    │
132 │     validator_report_records: VecMap<address, VecSet<address>>,
    │                               ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:195:19
    │
195 │     extra_fields: Bag,
    │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:189:36
    │
189 │     safe_mode_computation_rewards: Balance<SUI>,
    │                                    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:188:32
    │
188 │     safe_mode_storage_rewards: Balance<SUI>,
    │                                ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:178:31
    │
178 │     validator_report_records: VecMap<address, VecSet<address>>,
    │                               ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
   ┌─ ./sources/sui_system_state_inner.move:73:19
   │
73 │     extra_fields: Bag,
   │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:103:19
    │
103 │     extra_fields: Bag,
    │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1066:4
     │
1066 │ ): VecMap<address, u64> {
     │    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:862:25
    │
862 │     mut storage_reward: Balance<SUI>,
    │                         ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:863:29
    │
863 │     mut computation_reward: Balance<SUI>,
    │                             ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:872:15
    │
872 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:873:4
    │
873 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:513:15
    │
513 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:221:27
    │
221 │     initial_storage_fund: Balance<SUI>,
    │                           ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:226:15
    │
226 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:261:15
    │
261 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1206:66
     │
1206 │ public(package) fun extra_fields(self: &SuiSystemStateInnerV2): &Bag {
     │                                                                  ^^^ Unbound type 'Bag' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1210:78
     │
1210 │ public(package) fun extra_fields_mut(self: &mut SuiSystemStateInnerV2): &mut Bag {
     │                                                                              ^^^ Unbound type 'Bag' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1127:23
     │
1127 │     mut coins: vector<Coin<SUI>>,
     │                       ^^^^ Unbound type 'Coin' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1128:13
     │
1128 │     amount: Option<u64>,
     │             ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1129:15
     │
1129 │     ctx: &mut TxContext,
     │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1130:4
     │
1130 │ ): Balance<SUI> {
     │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1093:84
     │
1093 │ public(package) fun get_reporters_of(self: &SuiSystemStateInnerV2, addr: address): VecSet<address> {
     │                                                                                    ^^^^^^ Unbound type 'VecSet' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1115:14
     │
1115 │     pool_id: ID,
     │              ^^ Unbound type 'ID' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1116:5
     │
1116 │ ): &Table<u64, PoolTokenExchangeRate> {
     │     ^^^^^ Unbound type 'Table' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:521:11
    │
521 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:522:4
    │
522 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:562:36
    │
562 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:480:12
    │
480 │     stake: Coin<SUI>,
    │            ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:482:15
    │
482 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:490:20
    │
490 │     stakes: vector<Coin<SUI>>,
    │                    ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:491:19
    │
491 │     stake_amount: Option<u64>,
    │                   ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:493:15
    │
493 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:395:83
    │
395 │ public(package) fun request_add_validator(self: &mut SuiSystemStateInnerV2, ctx: &TxContext) {
    │                                                                                   ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:358:15
    │
358 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1283:15
     │
1283 │     ctx: &mut TxContext,
     │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:409:86
    │
409 │ public(package) fun request_remove_validator(self: &mut SuiSystemStateInnerV2, ctx: &TxContext) {
    │                                                                                      ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:386:15
    │
386 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:457:11
    │
457 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:503:11
    │
503 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:504:4
    │
504 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:597:86
    │
597 │ public(package) fun rotate_operation_cap(self: &mut SuiSystemStateInnerV2, ctx: &mut TxContext) {
    │                                                                                      ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:469:11
    │
469 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:579:36
    │
579 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound module
     ┌─ ./sources/sui_system_state_inner.move:4:1
     │  
   4 │ ╭ module sui_system::sui_system_state_inner;
   5 │ │ 
   6 │ │ use sui::bag::{Self, Bag};
   7 │ │ use sui::balance::{Self, Balance};
     · │
1310 │ │     (($a as u128) * ($b as u128) / ($c as u128)) as u64
1311 │ │ }
     │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:676:11
    │
676 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:843:11
    │
843 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:704:11
    │
704 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:729:11
    │
729 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:787:11
    │
787 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:757:11
    │
757 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:815:11
    │
815 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:629:11
    │
629 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:639:11
    │
639 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:613:11
    │
613 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:663:11
    │
663 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:830:11
    │
830 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:691:11
    │
691 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:716:11
    │
716 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:773:11
    │
773 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:744:11
    │
744 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:802:11
    │
802 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/sui_system_state_inner.move:649:11
    │
649 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1108:15
     │
1108 │     pool_id: &ID,
     │               ^^ Unbound type 'ID' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1081:4
     │
1081 │ ): ID {
     │    ^^ Unbound type 'ID' in current scope

error: unbound type
     ┌─ ./sources/sui_system_state_inner.move:1088:5
     │
1088 │ ): &Table<ID, address> {
     │     ^^^^^ Unbound type 'Table' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:502:9
    │
502 │         assert_eq!(pool.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:503:9
    │
503 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:504:9
    │
504 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:509:5
    │
509 │     assert_eq!(staked_sui.amount(), stake_amount * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:515:9
    │
515 │         assert_eq!(pool.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:516:9
    │
516 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:517:9
    │
517 │         assert_eq!(pool.sui_balance(), (100 + stake_amount) * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:526:5
    │
526 │     assert_eq!(fungible_staked_sui.value(), stake_amount * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:533:5
    │
533 │     assert_eq!(sui.destroy_for_testing(), stake_amount * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:538:9
    │
538 │         assert_eq!(pool.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:539:9
    │
539 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:540:9
    │
540 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:37:9
   │
37 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:43:9
   │
43 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1, @3])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:49:9
   │
49 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1, @3])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:55:9
   │
55 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:62:9
   │
62 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:68:9
   │
68 │         assert_eq!(system.get_reporters_of(@1).into_keys(), vector[@2])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:74:9
   │
74 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1, @3])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/sui_system_tests.move:81:9
   │
81 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1])
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:122:9
    │
122 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1]);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:141:9
    │
141 │         assert_eq!(system.get_reporters_of(@2).into_keys(), vector[@1]);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:159:9
    │
159 │         assert_eq!(system.active_validator_by_address(@1).next_epoch_gas_price(), 666);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:160:9
    │
160 │         assert_eq!(system.pending_validator_by_address(new_validator).next_epoch_gas_price(), 777);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:167:9
    │
167 │         assert_eq!(system.active_validator_by_address(@1).gas_price(), 666);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:168:9
    │
168 │         assert_eq!(system.active_validator_by_address(new_validator).gas_price(), 1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:175:9
    │
175 │         assert_eq!(system.active_validator_by_address(new_validator).gas_price(), 777);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:424:9
    │
424 │         assert_eq!(counter, 1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:434:9
    │
434 │         assert_eq!(counter, 1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:444:9
    │
444 │         assert_eq!(counter, 2);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:344:9
    │
344 │         assert_eq!(pool_mappings.length(), 4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:345:9
    │
345 │         assert_eq!(pool_mappings[pool_id_1], @1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:346:9
    │
346 │         assert_eq!(pool_mappings[pool_id_2], @2);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:347:9
    │
347 │         assert_eq!(pool_mappings[pool_id_3], @3);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:348:9
    │
348 │         assert_eq!(pool_mappings[pool_id_4], @4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:371:9
    │
371 │         assert_eq!(pool_mappings.length(), 5);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:372:9
    │
372 │         assert_eq!(pool_mappings[pool_id_1], @1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:373:9
    │
373 │         assert_eq!(pool_mappings[pool_id_2], @2);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:374:9
    │
374 │         assert_eq!(pool_mappings[pool_id_3], @3);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:375:9
    │
375 │         assert_eq!(pool_mappings[pool_id_4], @4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:376:9
    │
376 │         assert_eq!(pool_mappings[pool_id_5], new_validator);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:392:9
    │
392 │         assert_eq!(pool_mappings.length(), 4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:393:9
    │
393 │         assert_eq!(pool_mappings[pool_id_2], @2);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:394:9
    │
394 │         assert_eq!(pool_mappings[pool_id_3], @3);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:395:9
    │
395 │         assert_eq!(pool_mappings[pool_id_4], @4);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:396:9
    │
396 │         assert_eq!(pool_mappings[pool_id_5], new_validator);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound module
    ┌─ ./tests/sui_system_tests.move:9:1
    │  
  9 │ ╭ module sui_system::sui_system_tests;
 10 │ │ 
 11 │ │ use std::unit_test::assert_eq;
 12 │ │ use sui_system::test_runner;
    · │
543 │ │     runner.finish();
544 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:319:9
    │
319 │         assert_eq!(system.validator_address_by_pool_id(&pool_id), @1);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:460:9
    │
460 │         assert_eq!(pool.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:461:9
    │
461 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:462:9
    │
462 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:471:9
    │
471 │         assert_eq!(pool.pending_stake_amount(), stake_amount * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:472:9
    │
472 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:473:9
    │
473 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:482:9
    │
482 │         assert_eq!(pool.pending_stake_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:483:9
    │
483 │         assert_eq!(pool.pending_stake_withdraw_amount(), 0);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/sui_system_tests.move:484:9
    │
484 │         assert_eq!(pool.sui_balance(), 100 * MIST_PER_SUI);
    │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:221:25
    │
221 │     computation_charge: Option<u64>,
    │                         ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:226:23
    │
226 │     epoch_start_time: Option<u64>,
    │                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:223:33
    │
223 │     non_refundable_storage_fee: Option<u64>,
    │                                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:219:23
    │
219 │     protocol_version: Option<u64>,
    │                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:225:27
    │
225 │     reward_slashing_rate: Option<u64>,
    │                           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:220:21
    │
220 │     storage_charge: Option<u64>,
    │                     ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:224:33
    │
224 │     storage_fund_reinvest_rate: Option<u64>,
    │                                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:222:21
    │
222 │     storage_rebate: Option<u64>,
    │                     ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:306:15
    │
306 │     scenario: Scenario,
    │               ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:35:21
   │
35 │     epoch_duration: Option<u64>,
   │                     ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:36:29
   │
36 │     low_stake_grace_period: Option<u64>,
   │                             ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:32:23
   │
32 │     protocol_version: Option<u64>,
   │                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:33:33
   │
33 │     stake_distribution_counter: Option<u64>,
   │                                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:34:18
   │
34 │     start_epoch: Option<u64>,
   │                  ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:28:26
   │
28 │     storage_fund_amount: Option<u64>,
   │                          ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:27:24
   │
27 │     sui_supply_amount: Option<u64>,
   │                        ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:26:17
   │
26 │     validators: Option<vector<ValidatorBuilder>>,
   │                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:30:23
   │
30 │     validators_count: Option<u64>,
   │                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/test_runner.move:31:31
   │
31 │     validators_initial_stake: Option<u64>,
   │                               ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:409:14
    │
409 │     options: Option<AdvanceEpochOptions>,
    │              ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:410:4
    │
410 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:319:47
    │
319 │ public fun ctx(runner: &mut TestRunner): &mut TxContext { runner.scenario.ctx() }
    │                                               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./tests/builders/test_runner.move:344:5
    │
344 │     std::unit_test::destroy(v);
    │     ^^^^^^^^^^^^^^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./tests/builders/test_runner.move:338:31
    │
338 │ public fun mint(amount: u64): Balance<SUI> {
    │                               ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:356:66
    │
356 │ public macro fun scenario_fn($runner: &mut TestRunner, $f: |&mut Scenario|) {
    │                                                                  ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:322:56
    │
322 │ public fun scenario_mut(runner: &mut TestRunner): &mut Scenario { &mut runner.scenario }
    │                                                        ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:575:29
    │
575 │     scenario.ids_for_sender<Coin<SUI>>().fold!(0, |mut sum, coin_id| {
    │                             ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:576:52
    │
576 │         let coin = scenario.take_from_sender_by_id<Coin<SUI>>(coin_id);
    │                                                    ^^^^ Unbound type 'Coin' in current scope

error: unbound type
    ┌─ ./tests/builders/test_runner.move:374:36
    │
374 │     $f: |&mut SuiSystemState, &mut TxContext|,
    │                                    ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./tests/builders/test_runner.move:7:1
    │  
  7 │ ╭ module sui_system::test_runner;
  8 │ │ 
  9 │ │ use sui::balance::{Self, Balance};
 10 │ │ use sui::coin::{Self, Coin};
    · │
617 │ │     runner.finish();
618 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/validator.move:156:14
    │
156 │     pool_id: ID,
    │              ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator.move:164:14
    │
164 │     pool_id: ID,
    │              ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator.move:136:14
    │
136 │     pool_id: ID,
    │              ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator.move:145:14
    │
145 │     pool_id: ID,
    │              ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator.move:131:19
    │
131 │     extra_fields: Bag,
    │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
    ┌─ ./sources/validator.move:117:23
    │
117 │     operation_cap_id: ID,
    │                       ^^ Unbound type 'ID' in current scope

error: unbound type
   ┌─ ./sources/validator.move:85:18
   │
85 │     description: String,
   │                  ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:107:19
    │
107 │     extra_fields: Bag,
    │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
   ┌─ ./sources/validator.move:86:16
   │
86 │     image_url: Url,
   │                ^^^ Unbound type 'Url' in current scope

error: unbound type
   ┌─ ./sources/validator.move:84:11
   │
84 │     name: String,
   │           ^^^^^^ Unbound type 'String' in current scope

error: unbound type
   ┌─ ./sources/validator.move:89:18
   │
89 │     net_address: String,
   │                  ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:102:29
    │
102 │     next_epoch_net_address: Option<String>,
    │                             ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:100:38
    │
100 │     next_epoch_network_pubkey_bytes: Option<vector<u8>>,
    │                                      ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:103:29
    │
103 │     next_epoch_p2p_address: Option<String>,
    │                             ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:104:33
    │
104 │     next_epoch_primary_address: Option<String>,
    │                                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./sources/validator.move:99:37
   │
99 │     next_epoch_proof_of_possession: Option<vector<u8>>,
   │                                     ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./sources/validator.move:98:39
   │
98 │     next_epoch_protocol_pubkey_bytes: Option<vector<u8>>,
   │                                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:105:32
    │
105 │     next_epoch_worker_address: Option<String>,
    │                                ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:101:37
    │
101 │     next_epoch_worker_pubkey_bytes: Option<vector<u8>>,
    │                                     ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./sources/validator.move:91:18
   │
91 │     p2p_address: String,
   │                  ^^^^^^ Unbound type 'String' in current scope

error: unbound type
   ┌─ ./sources/validator.move:93:22
   │
93 │     primary_address: String,
   │                      ^^^^^^ Unbound type 'String' in current scope

error: unbound type
   ┌─ ./sources/validator.move:87:18
   │
87 │     project_url: Url,
   │                  ^^^ Unbound type 'Url' in current scope

error: unbound type
   ┌─ ./sources/validator.move:95:21
   │
95 │     worker_address: String,
   │                     ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:642:39
    │
642 │ macro fun both_some_and_equal<$T>($a: Option<$T>, $b: Option<$T>): bool {
    │                                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:642:55
    │
642 │ macro fun both_some_and_equal<$T>($a: Option<$T>, $b: Option<$T>): bool {
    │                                                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:310:15
    │
310 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:432:73
    │
432 │ public(package) fun deposit_stake_rewards(self: &mut Validator, reward: Balance<SUI>) {
    │                                                                         ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator.move:461:44
    │
461 │ public fun description(self: &Validator): &String {
    │                                            ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:896:35
    │
896 │ macro fun do_extract<$T>($o: &mut Option<$T>, $f: |$T|) {
    │                                   ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:465:42
    │
465 │ public fun image_url(self: &Validator): &Url {
    │                                          ^^^ Unbound type 'Url' in current scope

error: unbound type
    ┌─ ./sources/validator.move:457:37
    │
457 │ public fun name(self: &Validator): &String {
    │                                     ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:473:48
    │
473 │ public fun network_address(self: &Validator): &String {
    │                                                ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:227:15
    │
227 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:967:27
    │
967 │     initial_stake_option: Option<Balance<SUI>>,
    │                           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:971:15
    │
971 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:921:15
    │
921 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:175:11
    │
175 │     name: String,
    │           ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:176:18
    │
176 │     description: String,
    │                  ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:177:16
    │
177 │     image_url: Url,
    │                ^^^ Unbound type 'Url' in current scope

error: unbound type
    ┌─ ./sources/validator.move:178:18
    │
178 │     project_url: Url,
    │                  ^^^ Unbound type 'Url' in current scope

error: unbound type
    ┌─ ./sources/validator.move:179:18
    │
179 │     net_address: String,
    │                  ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:180:18
    │
180 │     p2p_address: String,
    │                  ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:181:22
    │
181 │     primary_address: String,
    │                      ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:182:21
    │
182 │     worker_address: String,
    │                     ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:183:19
    │
183 │     extra_fields: Bag,
    │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
    ┌─ ./sources/validator.move:653:15
    │
653 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:505:59
    │
505 │ public fun next_epoch_network_address(self: &Validator): &Option<String> {
    │                                                           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:529:64
    │
529 │ public fun next_epoch_network_pubkey_bytes(self: &Validator): &Option<vector<u8>> {
    │                                                                ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:509:55
    │
509 │ public fun next_epoch_p2p_address(self: &Validator): &Option<String> {
    │                                                       ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:513:59
    │
513 │ public fun next_epoch_primary_address(self: &Validator): &Option<String> {
    │                                                           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:525:63
    │
525 │ public fun next_epoch_proof_of_possession(self: &Validator): &Option<vector<u8>> {
    │                                                               ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:521:65
    │
521 │ public fun next_epoch_protocol_pubkey_bytes(self: &Validator): &Option<vector<u8>> {
    │                                                                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:517:58
    │
517 │ public fun next_epoch_worker_address(self: &Validator): &Option<String> {
    │                                                          ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:533:63
    │
533 │ public fun next_epoch_worker_pubkey_bytes(self: &Validator): &Option<vector<u8>> {
    │                                                               ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator.move:537:49
    │
537 │ public fun operation_cap_id(self: &Validator): &ID {
    │                                                 ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator.move:477:44
    │
477 │ public fun p2p_address(self: &Validator): &String {
    │                                            ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:481:48
    │
481 │ public fun primary_address(self: &Validator): &String {
    │                                                ^^^^^^ Unbound type 'String' in current scope

error: unbound type
    ┌─ ./sources/validator.move:438:86
    │
438 │ public(package) fun process_pending_stakes_and_withdraws(self: &mut Validator, ctx: &TxContext) {
    │                                                                                      ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:469:44
    │
469 │ public fun project_url(self: &Validator): &Url {
    │                                            ^^^ Unbound type 'Url' in current scope

error: unbound type
    ┌─ ./sources/validator.move:329:11
    │
329 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:330:4
    │
330 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator.move:284:12
    │
284 │     stake: Balance<SUI>,
    │            ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator.move:286:15
    │
286 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:348:12
    │
348 │     stake: Balance<SUI>,
    │            ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator.move:350:15
    │
350 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:370:11
    │
370 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator.move:371:4
    │
371 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator.move:590:47
    │
590 │ public fun staking_pool_id(self: &Validator): ID {
    │                                               ^^ Unbound type 'ID' in current scope

error: unbound module
     ┌─ ./sources/validator.move:5:1
     │  
   5 │ ╭ module sui_system::validator;
   6 │ │ 
   7 │ │ use std::bcs;
   8 │ │ use std::string::String;
     · │
1009 │ │     validator
1010 │ │ }
     │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/validator.move:485:47
    │
485 │ public fun worker_address(self: &Validator): &String {
    │                                               ^^^^^^ Unbound type 'String' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:36:22
   │
36 │     commission_rate: Option<u64>,
   │                      ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:28:18
   │
28 │     description: Option<vector<u8>>,
   │                  ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:35:16
   │
35 │     gas_price: Option<u64>,
   │                ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:29:16
   │
29 │     image_url: Option<vector<u8>>,
   │                ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:38:20
   │
38 │     initial_stake: Option<u64>,
   │                    ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:27:11
   │
27 │     name: Option<vector<u8>>,
   │           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:31:18
   │
31 │     net_address: Option<vector<u8>>,
   │                  ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:24:27
   │
24 │     network_pubkey_bytes: Option<vector<u8>>,
   │                           ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:32:18
   │
32 │     p2p_address: Option<vector<u8>>,
   │                  ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:33:22
   │
33 │     primary_address: Option<vector<u8>>,
   │                      ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:30:18
   │
30 │     project_url: Option<vector<u8>>,
   │                  ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:26:26
   │
26 │     proof_of_possession: Option<vector<u8>>,
   │                          ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:23:28
   │
23 │     protocol_pubkey_bytes: Option<vector<u8>>,
   │                            ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:22:18
   │
22 │     sui_address: Option<address>,
   │                  ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:34:21
   │
34 │     worker_address: Option<vector<u8>>,
   │                     ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:25:26
   │
25 │     worker_pubkey_bytes: Option<vector<u8>>,
   │                          ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
   ┌─ ./tests/builders/validator_builder.move:92:55
   │
92 │ public fun build(builder: ValidatorBuilder, ctx: &mut TxContext): Validator {
   │                                                       ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./tests/builders/validator_builder.move:135:64
    │
135 │ public fun build_metadata(builder: ValidatorBuilder, ctx: &mut TxContext): ValidatorMetadata {
    │                                                                ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./tests/builders/validator_builder.move:10:1
    │  
 10 │ ╭ module sui_system::validator_builder;
 11 │ │ 
 12 │ │ use sui::bag;
 13 │ │ use sui::balance;
    · │
340 │ │     preset.protocol_pubkey_bytes()
341 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
   ┌─ ./sources/validator_cap.move:18:9
   │
18 │     id: UID,
   │         ^^^ Unbound type 'UID' in current scope

error: unbound type
   ┌─ ./sources/validator_cap.move:42:15
   │
42 │     ctx: &mut TxContext,
   │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
   ┌─ ./sources/validator_cap.move:43:4
   │
43 │ ): ID {
   │    ^^ Unbound type 'ID' in current scope

error: unbound module
   ┌─ ./sources/validator_cap.move:4:1
   │  
 4 │ ╭ module sui_system::validator_cap;
 5 │ │ 
 6 │ │ /// The capability object is created when creating a new `Validator` or when the
 7 │ │ /// validator explicitly creates a new capability object for rotation/revocation.
   · │
62 │ │     ValidatorOperationCap { authorizer_validator_address: cap.authorizer_validator_address }
63 │ │ }
   │ ╰─^ Unbound module 'std::unit_test'

error: unbound module
    ┌─ ./tests/validator_metadata_tests.move:5:1
    │  
  5 │ ╭ module sui_system::validator_metadata_tests;
  6 │ │ 
  7 │ │ use std::unit_test;
  8 │ │ use sui::test_scenario::{Self, Scenario};
    · │
917 │ │     assert!(validator.next_epoch_network_pubkey_bytes().is_none());
918 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./tests/validator_metadata_tests.move:777:20
    │
777 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound module
    ┌─ ./tests/builders/validator_preset.move:186:26
    │
186 │         account_address: sui::address::from_bytes(preset[1]),
    │                          ^^^^^^^^^^^^ Unbound module 'sui::address'

error: unbound module
    ┌─ ./tests/builders/validator_preset.move:5:1
    │  
  5 │ ╭ module sui_system::validator_preset;
  6 │ │ 
  7 │ │ const VALID_NET_PUBKEY: vector<u8> = vector[
  8 │ │     171, 2, 39, 3, 139, 105, 166, 171, 153, 151, 102, 197, 151, 186, 140, 116, 114, 90, 213, 225, 20,
    · │
250 │ │     preset.project_url
251 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/validator_set.move:123:22
    │
123 │     staking_pool_id: ID,
    │                      ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:131:22
    │
131 │     staking_pool_id: ID,
    │                      ^^ Unbound type 'ID' in current scope

error: unbound type
   ┌─ ./sources/validator_set.move:82:25
   │
82 │     at_risk_validators: VecMap<address, u64>,
   │                         ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
   ┌─ ./sources/validator_set.move:84:19
   │
84 │     extra_fields: Bag,
   │                   ^^^ Unbound type 'Bag' in current scope

error: unbound type
   ┌─ ./sources/validator_set.move:74:26
   │
74 │     inactive_validators: Table<ID, ValidatorWrapper>,
   │                          ^^^^^ Unbound type 'Table' in current scope

error: unbound type
   ┌─ ./sources/validator_set.move:65:32
   │
65 │     pending_active_validators: TableVec<Validator>,
   │                                ^^^^^^^^ Unbound type 'TableVec' in current scope

error: unbound type
   ┌─ ./sources/validator_set.move:70:28
   │
70 │     staking_pool_mappings: Table<ID, address>,
   │                            ^^^^^ Unbound type 'Table' in current scope

error: unbound type
   ┌─ ./sources/validator_set.move:80:27
   │
80 │     validator_candidates: Table<address, ValidatorWrapper>,
   │                           ^^^^^ Unbound type 'Table' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:376:30
    │
376 │     computation_reward: &mut Balance<SUI>,
    │                              ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:377:31
    │
377 │     storage_fund_reward: &mut Balance<SUI>,
    │                               ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:378:36
    │
378 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:381:15
    │
381 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:233:53
    │
233 │ fun can_join(self: &ValidatorSet, stake: u64, ctx: &TxContext): bool {
    │                                                     ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:976:36
    │
976 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1125:44
     │
1125 │     individual_staking_reward_adjustments: VecMap<u64, u64>,
     │                                            ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1127:49
     │
1127 │     individual_storage_fund_reward_adjustments: VecMap<u64, u64>,
     │                                                 ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1014:5
     │
1014 │     VecMap<u64, u64>, // mapping of individual validator's staking reward adjustment from index -> amount
     │     ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1016:5
     │
1016 │     VecMap<u64, u64>, // mapping of individual validator's storage fund reward adjustment from index -> amount
     │     ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1066:35
     │
1066 │     mut validator_report_records: VecMap<address, VecSet<address>>,
     │                                   ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:329:15
    │
329 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1188:27
     │
1188 │     staking_rewards: &mut Balance<SUI>,
     │                           ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1189:31
     │
1189 │     storage_fund_reward: &mut Balance<SUI>,
     │                               ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1190:15
     │
1190 │     ctx: &mut TxContext,
     │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1237:22
     │
1237 │     report_records: &VecMap<address, VecSet<address>>,
     │                      ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:743:81
    │
743 │ fun find_validator(validators: &vector<Validator>, validator_address: address): Option<u64> {
    │                                                                                 ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:751:18
    │
751 │     validators: &TableVec<Validator>,
    │                  ^^^^^^^^ Unbound type 'TableVec' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:753:4
    │
753 │ ): Option<u64> {
    │    ^^^^^^ Unbound type 'Option' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:247:60
    │
247 │ fun get_voting_power_thresholds(self: &ValidatorSet, ctx: &TxContext): (u64, u64, u64) {
    │                                                            ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1317:81
     │
1317 │ public(package) fun inactive_validator_by_pool_id(self: &ValidatorSet, pool_id: ID): &Validator {
     │                                                                                 ^^ Unbound type 'ID' in current scope

error: unbound type
     ┌─ ./sources/validator_set.move:1296:72
     │
1296 │ public fun is_inactive_validator(self: &ValidatorSet, staking_pool_id: ID): bool {
     │                                                                        ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:143:15
    │
143 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:664:14
    │
664 │     pool_id: ID,
    │              ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:665:5
    │
665 │ ): &Table<u64, PoolTokenExchangeRate> {
    │     ^^^^^ Unbound type 'Table' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:921:36
    │
921 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:922:15
    │
922 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:945:36
    │
945 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:947:15
    │
947 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:348:11
    │
348 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:349:4
    │
349 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:292:12
    │
292 │     stake: Balance<SUI>,
    │            ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:293:15
    │
293 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:221:74
    │
221 │ public(package) fun request_add_validator(self: &mut ValidatorSet, ctx: &TxContext) {
    │                                                                          ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:171:15
    │
171 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:273:77
    │
273 │ public(package) fun request_remove_validator(self: &mut ValidatorSet, ctx: &TxContext) {
    │                                                                             ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:198:15
    │
198 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:311:11
    │
311 │     ctx: &TxContext,
    │           ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:312:4
    │
312 │ ): Balance<SUI> {
    │    ^^^^^^^ Unbound type 'Balance' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:648:57
    │
648 │ public fun staking_pool_mappings(self: &ValidatorSet): &Table<ID, address> {
    │                                                         ^^^^^ Unbound type 'Table' in current scope

error: unbound module
     ┌─ ./sources/validator_set.move:4:1
     │  
   4 │ ╭ module sui_system::validator_set;
   5 │ │ 
   6 │ │ use sui::bag::{Self, Bag};
   7 │ │ use sui::balance::Balance;
     · │
1369 │ │     abort
1370 │ │ }
     │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
    ┌─ ./sources/validator_set.move:497:36
    │
497 │     validator_report_records: &mut VecMap<address, VecSet<address>>,
    │                                    ^^^^^^ Unbound type 'VecMap' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:498:15
    │
498 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:652:76
    │
652 │ public fun validator_address_by_pool_id(self: &mut ValidatorSet, pool_id: &ID): address {
    │                                                                            ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:678:76
    │
678 │ public(package) fun validator_by_pool_id(self: &mut ValidatorSet, pool_id: ID): &Validator {
    │                                                                            ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./sources/validator_set.move:644:88
    │
644 │ public fun validator_staking_pool_id(self: &ValidatorSet, validator_address: address): ID {
    │                                                                                        ^^ Unbound type 'ID' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:676:20
    │
676 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:295:5
    │
295 │     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:304:5
    │
304 │     assert_eq!(validator_set.validator_address_by_pool_id(&pool_id_2), @0x2);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:311:5
    │
311 │     assert_eq!(validator_set.validator_address_by_pool_id(&pool_id_2), @0x2);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:313:5
    │
313 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:547:5
    │
547 │     destroy(stake);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:561:9
    │
561 │         destroy(stake);
    │         ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:571:5
    │
571 │     assert_eq!(effects.num_user_events(), num_validators); // epoch changes hould not emit ValidatorJoinEvent or ValidatorLeaveEvent
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:573:5
    │
573 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:220:5
    │
220 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:275:5
    │
275 │     assert_eq!(effects.num_user_events(), num_validators + 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:277:5
    │
277 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:669:87
    │
669 │ fun advance_epoch_with_dummy_rewards(validator_set: &mut ValidatorSet, scenario: &mut Scenario) {
    │                                                                                       ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:650:20
    │
650 │     scenario: &mut Scenario,
    │                    ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:582:15
    │
582 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:614:15
    │
614 │     ctx: &mut TxContext,
    │               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:225:33
    │
225 │ fun get_10_validators(ctx: &mut TxContext): vector<Validator> {
    │                                 ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:456:5
    │
456 │     assert_eq!(validator_set.find_for_testing(@0xB).voting_power(), 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:474:5
    │
474 │     assert_eq!(effects.num_user_events(), num_validators + 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:476:5
    │
476 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:477:5
    │
477 │     destroy(bal);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:478:5
    │
478 │     destroy(stake);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:523:5
    │
523 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:524:5
    │
524 │     destroy(stake1);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:525:5
    │
525 │     destroy(stake2);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:105:5
    │
105 │     assert_eq!(validator_set.derive_reference_gas_price(), 45);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:110:5
    │
110 │     assert_eq!(validator_set.derive_reference_gas_price(), 45);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:119:5
    │
119 │     assert_eq!(validator_set.derive_reference_gas_price(), 42);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:128:5
    │
128 │     assert_eq!(validator_set.derive_reference_gas_price(), 42);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:137:5
    │
137 │     assert_eq!(validator_set.derive_reference_gas_price(), 43);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:139:5
    │
139 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:341:5
    │
341 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:342:5
    │
342 │     destroy(bal);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
    ┌─ ./tests/validator_set_tests.move:241:58
    │
241 │ fun skip_to_min_stake_v2_final_thresholds(scenario: &mut Scenario) {
    │                                                          ^^^^^^^^ Unbound type 'Scenario' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:151:5
    │
151 │     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:164:5
    │
164 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:176:5
    │
176 │     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:192:5
    │
192 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound module
    ┌─ ./tests/validator_set_tests.move:5:1
    │  
  5 │ ╭ module sui_system::validator_set_tests;
  6 │ │ 
  7 │ │ use std::unit_test::{assert_eq, destroy};
  8 │ │ use sui::address;
    · │
681 │ │     validator_set.request_add_validator(ctx);
682 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:31:5
   │
31 │     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:40:5
   │
40 │     assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:61:9
   │
61 │         assert_eq!(validator_set.total_stake(), 100 * MIST_PER_SUI);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:72:5
   │
72 │     assert_eq!(validator_set.total_stake(), 1500 * MIST_PER_SUI);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:82:5
   │
82 │     assert_eq!(validator_set.total_stake(), 1500 * MIST_PER_SUI);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:85:5
   │
85 │     assert_eq!(validator_set.total_stake(), 900 * MIST_PER_SUI);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_set_tests.move:87:5
   │
87 │     destroy(validator_set);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:416:5
    │
416 │     assert_eq!(effects.num_user_events(), num_validators + 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:418:5
    │
418 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:419:5
    │
419 │     destroy(bal);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:420:5
    │
420 │     destroy(stake);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:376:5
    │
376 │     assert_eq!(effects.num_user_events(), num_validators + 1);
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:378:5
    │
378 │     destroy(validator_set);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_set_tests.move:379:5
    │
379 │     destroy(bal);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:102:5
    │
102 │     destroy(metadata);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:69:5
   │
69 │     assert_eq!(validator.total_stake(), initial_stake);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:70:5
   │
70 │     assert_eq!(validator.pending_stake_amount(), added_stake);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:78:9
   │
78 │         assert_eq!(withdrawn_balance, initial_stake);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:79:9
   │
79 │         assert_eq!(validator.total_stake(), initial_stake);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:80:9
   │
80 │         assert_eq!(validator.pending_stake_amount(), added_stake);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:81:9
   │
81 │         assert_eq!(validator.pending_stake_withdraw_amount(), initial_stake);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:87:9
   │
87 │         assert_eq!(validator.total_stake(), added_stake);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:88:9
   │
88 │         assert_eq!(validator.pending_stake_amount(), 0);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:89:9
   │
89 │         assert_eq!(validator.pending_stake_withdraw_amount(), 0);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound module
    ┌─ ./tests/validator_tests.move:5:1
    │  
  5 │ ╭ module sui_system::validator_tests;
  6 │ │ 
  7 │ │ use std::unit_test::{assert_eq, destroy};
  8 │ │ use sui::balance;
    · │
485 │ │     abort
486 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:33:5
   │
33 │     assert_eq!(validator.total_stake(), initial_stake);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:34:5
   │
34 │     assert_eq!(validator.sui_address(), @2);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:38:9
   │
38 │         assert_eq!(stake.amount(), initial_stake);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:39:9
   │
39 │         assert_eq!(stake.pool_id(), pool_id);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/validator_tests.move:40:9
   │
40 │         assert_eq!(stake.stake_activation_epoch(), 0);
   │         ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:246:5
    │
246 │     assert_eq!(*validator.name(), b"new_name".to_string());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:247:5
    │
247 │     assert_eq!(*validator.description(), b"new_desc".to_string());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:248:5
    │
248 │     assert_eq!(*validator.image_url(), url::new_unsafe_from_bytes(b"new_image_url"));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:249:5
    │
249 │     assert_eq!(*validator.project_url(), url::new_unsafe_from_bytes(b"new_proj_url"));
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:250:5
    │
250 │     assert_eq!(*validator.network_address(), validator_builder::valid_net_addr().to_string());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:251:5
    │
251 │     assert_eq!(*validator.p2p_address(), validator_builder::valid_p2p_addr().to_string());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:252:5
    │
252 │     assert_eq!(*validator.primary_address(), validator_builder::valid_consensus_addr().to_string());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:253:5
    │
253 │     assert_eq!(*validator.worker_address(), validator_builder::valid_worker_addr().to_string());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:254:5
    │
254 │     assert_eq!(*validator.protocol_pubkey_bytes(), validator_builder::valid_pubkey());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:255:5
    │
255 │     assert_eq!(*validator.proof_of_possession(), validator_builder::valid_proof_of_possession());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:256:5
    │
256 │     assert_eq!(*validator.network_pubkey_bytes(), validator_builder::valid_net_pubkey());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:257:5
    │
257 │     assert_eq!(*validator.worker_pubkey_bytes(), validator_builder::valid_worker_pubkey());
    │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
    ┌─ ./tests/validator_tests.move:293:5
    │
293 │     destroy(validator);
    │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
   ┌─ ./sources/validator_wrapper.move:12:12
   │
12 │     inner: Versioned,
   │            ^^^^^^^^^ Unbound type 'Versioned' in current scope

error: unbound type
   ┌─ ./sources/validator_wrapper.move:16:63
   │
16 │ public(package) fun create_v1(validator: Validator, ctx: &mut TxContext): ValidatorWrapper {
   │                                                               ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
   ┌─ ./sources/validator_wrapper.move:4:1
   │  
 4 │ ╭ module sui_system::validator_wrapper;
 5 │ │ 
 6 │ │ use sui::versioned::{Self, Versioned};
 7 │ │ use sui_system::validator::Validator;
   · │
49 │ │     self.inner.version()
50 │ │ }
   │ ╰─^ Unbound module 'std::unit_test'

error: unbound module
    ┌─ ./sources/voting_power.move:4:1
    │  
  4 │ ╭ module sui_system::voting_power;
  5 │ │ 
  6 │ │ use sui_system::validator::Validator;
  7 │ │ 
    · │
165 │ │     QUORUM_THRESHOLD
166 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: unbound type
   ┌─ ./tests/voting_power_tests.move:15:69
   │
15 │ fun check(stakes: vector<u64>, voting_power: vector<u64>, ctx: &mut TxContext) {
   │                                                                     ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound unscoped name
   ┌─ ./tests/voting_power_tests.move:26:5
   │
26 │     assert_eq!(voting_powers, voting_power);
   │     ^^^^^^^^^ Unbound function 'assert_eq' in current scope

error: unbound unscoped name
   ┌─ ./tests/voting_power_tests.move:27:5
   │
27 │     destroy(validators);
   │     ^^^^^^^ Unbound function 'destroy' in current scope

error: unbound type
    ┌─ ./tests/voting_power_tests.move:117:66
    │
117 │ fun create_validators_with_stakes(stakes: vector<u64>, ctx: &mut TxContext): vector<Validator> {
    │                                                                  ^^^^^^^^^ Unbound type 'TxContext' in current scope

error: unbound module
    ┌─ ./tests/voting_power_tests.move:5:1
    │  
  5 │ ╭ module sui_system::voting_power_tests;
  6 │ │ 
  7 │ │ use std::unit_test::{assert_eq, destroy};
  8 │ │ use sui_system::validator::{Self, Validator};
    · │
123 │ │     })
124 │ │ }
    │ ╰─^ Unbound module 'std::unit_test'

error: invalid method call
    ┌─ ./tests/rewards_distribution_tests.move:630:9
    │  
615 │       let num_validators = 20;
    │           -------------- Unable to infer type for method call. Try annotating this type
    ·  
630 │ ╭         num_validators.do!(|i| {
631 │ │             let addr = address::from_u256(i as u256);
632 │ │             assert_eq!(system.validator_stake_amount(addr), (962 + i * 4) * MIST_PER_SUI);
633 │ │         });
    │ ╰──────────^ Invalid method call

error: invalid method call
     ┌─ ./sources/validator_set.move:1079:14
     │
1070 │         let (validator_address, reporters) = validator_report_records.pop();
     │                                 --------- Unable to infer type for method call. Try annotating this type
     ·
1079 │             &reporters.into_keys(),
     │              ^^^^^^^^^^^^^^^^^^^^^ Invalid method call

error: cannot infer type
     ┌─ ./sources/validator_set.move:1070:33
     │
1070 │         let (validator_address, reporters) = validator_report_records.pop();
     │                                 ^^^^^^^^^ Could not infer this type. Try adding an annotation

error: cannot infer type
     ┌─ ./sources/validator_set.move:1247:13
     │
1247 │             vector[]
     │             ^^^^^^^^ Could not infer this type. Try adding an annotation

error: invalid object declaration
   ┌─ ./sources/staking_pool.move:90:5
   │
89 │ public struct FungibleStakedSui has key, store {
   │                                     --- The 'key' ability is used to declare objects in Sui
90 │     id: UID,
   │     ^^  --- But found type: '_'
   │     │    
   │     Invalid object 'FungibleStakedSui'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field

error: invalid object declaration
   ┌─ ./sources/staking_pool.move:99:5
   │
98 │ public struct FungibleStakedSuiData has key, store {
   │                                         --- The 'key' ability is used to declare objects in Sui
99 │     id: UID,
   │     ^^  --- But found type: '_'
   │     │    
   │     Invalid object 'FungibleStakedSuiData'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field

error: invalid object declaration
   ┌─ ./sources/staking_pool.move:76:5
   │
75 │ public struct StakedSui has key, store {
   │                             --- The 'key' ability is used to declare objects in Sui
76 │     id: UID,
   │     ^^  --- But found type: '_'
   │     │    
   │     Invalid object 'StakedSui'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field

error: invalid object declaration
   ┌─ ./sources/staking_pool.move:39:5
   │
38 │ public struct StakingPool has key, store {
   │                               --- The 'key' ability is used to declare objects in Sui
39 │     id: UID,
   │     ^^  --- But found type: '_'
   │     │    
   │     Invalid object 'StakingPool'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field

error: invalid object declaration
   ┌─ ./sources/sui_system.move:67:5
   │
66 │ public struct SuiSystemState has key {
   │                                  --- The 'key' ability is used to declare objects in Sui
67 │     id: UID,
   │     ^^  --- But found type: '_'
   │     │    
   │     Invalid object 'SuiSystemState'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field

error: invalid object declaration
   ┌─ ./sources/validator_cap.move:18:5
   │
17 │ public struct UnverifiedValidatorOperationCap has key, store {
   │                                                   --- The 'key' ability is used to declare objects in Sui
18 │     id: UID,
   │     ^^  --- But found type: '_'
   │     │    
   │     Invalid object 'UnverifiedValidatorOperationCap'. Structs with the 'key' ability must have 'id: sui::object::UID' as their first field



```

---

## Assistant

The project doesn't compile standalone (missing sui framework deps), so `all_accesses` can't be computed via the binary. But that's fine — the function is a trivial 0-dep getter. I have everything needed from the source.

### Code `toolu_01XUqr8cPMCcVuMEemTvvwJj`

```python

spec_plan = {
    "function": "staking_pool_specs::stake_activation_epoch_spec",
    "complexity": "low",
    "summary": "Pure accessor returning the epoch at which a StakedSui receipt becomes active, reading directly from the stake_activation_epoch field of the StakedSui struct.",
    "role": "Read-only getter exposing the stake_activation_epoch field of StakedSui; consumed by callers (e.g. request_withdraw_stake, convert_to_fungible_staked_sui, pool_token_exchange_rate_at_epoch lookups) that need to know when a given stake began earning rewards. Spec function mirrors the public staking_pool::stake_activation_epoch accessor.",
    "aborts": [],
    "requires": [],
    "ensures": [
        {
            "condition": "result == staked_sui.stake_activation_epoch",
            "reason": "Function is a pure field projection; return value must exactly equal the stored field value."
        }
    ],
    "observations": [
        "The function takes an immutable reference, so no state is mutated and no abort is possible.",
        "stake_activation_epoch is set once at request_add_stake time and never mutated thereafter; the returned value is therefore stable across the lifetime of the StakedSui object.",
        "The field is used as an epoch index into the pool's exchange_rates table; correctness of downstream exchange-rate lookups depends on this value being accurately reported.",
        "Splitting a StakedSui via staking_pool::split preserves stake_activation_epoch in both halves, so the getter behaves identically on split fragments."
    ]
}

final_result(spec_plan)

```

> 110140 in, 513 out, 15801 cache read, 12807 cache create

### Result `toolu_01XUqr8cPMCcVuMEemTvvwJj`

<!-- {"executed":true,"return_value":{"function":"staking_pool_specs::stake_activation_epoch_spec","complexity":"low","summary":"Pure accessor returning the epoch at which a StakedSui receipt becomes active, reading directly from the stake_activation_epoch field of the StakedSui struct.","role":"Read-only getter exposing the stake_activation_epoch field of StakedSui; consumed by callers (e.g. request_withdraw_stake, convert_to_fungible_staked_sui, pool_token_exchange_rate_at_epoch lookups) that need to know when a given stake began earning rewards. Spec function mirrors the public staking_pool::stake_activation_epoch accessor.","aborts":[],"requires":[],"ensures":[{"condition":"result == staked_sui.stake_activation_epoch","reason":"Function is a pure field projection; return value must exactly equal the stored field value."}],"observations":["The function takes an immutable reference, so no state is mutated and no abort is possible.","stake_activation_epoch is set once at request_add_stake time and never mutated thereafter; the returned value is therefore stable across the lifetime of the StakedSui object.","The field is used as an epoch index into the pool's exchange_rates table; correctness of downstream exchange-rate lookups depends on this value being accurately reported.","Splitting a StakedSui via staking_pool::split preserves stake_activation_epoch in both halves, so the getter behaves identically on split fragments."]}} -->
```
✓ ok
```

---

# Session Complete

| | |
|---|---|
| status | success |
| stop_reason | tool_use |
| model | claude-sonnet-4-6 |
| prompt | f1c104bef746c3804787ccae6440fa36ddc4e9c2522c964a6bc1ed76a5231f2b |
| turns | 2 |
| tool_calls | 2 |
| time | 18.42s |
| ended_at | 2026-06-16T00:03:05.008705 |
| cost | $0.4359 |
| tokens | 122790 in / 670 out |
| cache | 31602 read / 12807 create |


## Knowledge Output

- field: spec/writeup/staking_pool_specs::stake_activation_epoch_spec
- data: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/stake_activation_epoch_spec/spec.writeup.yaml
- meta: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/stake_activation_epoch_spec/writeup.meta.yaml
