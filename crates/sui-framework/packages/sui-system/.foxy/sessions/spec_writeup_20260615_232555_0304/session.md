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
name: null
description: null
cwd: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system
started_at: 2026-06-15T23:25:55.030604
foxy_commit: 38346c7c25594d3c381dff95b53fe33dba150411
prompt_part_hashes: {"base": "5fdb6c5e65d5df8a", "core": "35fc153c53e2c232", "file_ops": "b76d200c47b2271e", "function_knowledge": "ec5c60d9b1e6f113", "project_env": "21a3de2d42771978", "spec_writeup": "61585c0ab1a10352", "_global": "13968af88bcc8317b84d34d5a904ff01121fe1f3940cac1fd5cf9ad4645fb391"}
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

Produce a spec plan for `staking_pool_specs::is_equal_staking_metadata_spec` — a YAML document capturing what should be formally verified about this function. Analyze the function source, its callees, dependency graph, and role in the protocol. The analysis guidelines and output schema are in your system prompt.

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


## Target Function: `staking_pool_specs::is_equal_staking_metadata_spec`

Visibility: `public` | Entry: `False` | Estimated complexity: **low** (0 transitive deps)

```move
/// Returns true if all the staking parameters of the staked sui except the principal are identical
public fun is_equal_staking_metadata(self: &StakedSui, other: &StakedSui): bool {
    (self.pool_id == other.pool_id) &&
    (self.stake_activation_epoch == other.stake_activation_epoch)
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

## Entry Points Reaching `staking_pool_specs::is_equal_staking_metadata_spec`

These are the entry and public functions that can reach the target through the call graph — use this to understand the target's role in the protocol.

- **`staking_pool::is_equal_staking_metadata`**: `staking_pool::is_equal_staking_metadata`

## Relevant Structs

No relevant structs found.

---

Produce the spec plan for `staking_pool_specs::is_equal_staking_metadata_spec`. Return it via `final_result(spec_plan)` as a Python dict matching the YAML schema in your system prompt.

## Assistant

### Code `toolu_01AuPgJwBtShPRh2tX3B1P1Y`

```python

from foxy.skills.function_knowledge.api import get_knowledge

project = get_project_root()
k = get_knowledge(project)
fn = k.item("staking_pool_specs::is_equal_staking_metadata_spec")
foxy_inspect(fn.source)
foxy_inspect(fn.signature)
foxy_inspect(fn.all_accesses)

```

> 12643 in, 146 out, 15801 cache read

