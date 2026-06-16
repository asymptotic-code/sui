---
schema_version: 1
agent: spec_review
model: claude-opus-4-8
max_tokens: 16384
interactive: false
effort: null
timeout: 360
token_budget: null
harness: foxy
parent_session: null
name: null
description: null
cwd: /Users/cos/asymptotic/agent/clients/mysten/sui
started_at: 2026-06-16T00:13:03.534521
foxy_commit: 38346c7c25594d3c381dff95b53fe33dba150411
prompt_part_hashes: {"base": "5fdb6c5e65d5df8a", "core": "35fc153c53e2c232", "file_ops": "b76d200c47b2271e", "function_knowledge": "ec5c60d9b1e6f113", "project_env": "21a3de2d42771978", "spec_loop": "26d59a7f8c0f21db", "spec_postcondition": "c9935e5df9cbd57c", "spec_precondition": "74781a107ed639cf", "spec_review": "e31ecea77dacc494", "spec_scenario": "d1ce03efba7186ff", "sui_prover_guide": "9b7aaa77fb185386", "_global": "6504d9e79699e64eaa159774cd5eeb95ea20d0d3940fe8a11ef8cf8ac9d6daa3"}
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

# Sui Prover Specification Guide

## Quick Reference: Core Functions

Import with `use prover::prover::*`:

| Function | Description |
|----------|-------------|
| `requires(condition)` | Precondition that must hold before execution |
| `ensures(condition)` | Postcondition that must hold after execution |
| `asserts(condition)` | Assert condition is true, or function aborts |
| `clone!(&var)` | Capture variable's value at this point for later comparison |
| `forall!<T>(\|x\| predicate(x))` | Universal quantification |
| `exists!<T>(\|x\| predicate(x))` | Existential quantification |
| `boogie_split_here()` | Emit `assume {:split_here} true;` — cut the VC at this point |
| `boogie_focus()` | Emit `assume {:focus} true;` — split into "paths-through-here" vs "not-through-here" VCs |
| `boogie_allow_path_isolation()` | Mark the next `if` with `{:allow_path_isolation}`; pair with `boogie_opt = b"{:isolate_paths}"` |

Import with `use prover::vector_iter::*`:

| Function | Returns | Description |
|----------|---------|-------------|
| `all!<T>(&vec, \|x\| pred(x))` | `bool` | All elements satisfy predicate |
| `any!<T>(&vec, \|x\| pred(x))` | `bool` | Any element satisfies predicate |
| `count!<T>(&vec, \|x\| pred(x))` | `u64` | Count elements satisfying predicate |
| `map!<T, U>(&vec, \|x\| f(x))` | `&vector<U>` | Transform vector elements |
| `filter!<T>(&vec, \|x\| pred(x))` | `&vector<T>` | Filter vector elements |
| `find!<T>(&vec, \|x\| pred(x))` | `&Option<T>` | Find first matching element |
| `find_index!<T>(&vec, \|x\| pred(x))` | `Option<u64>` | Find index of first match |
| `find_indices!<T>(&vec, \|x\| pred(x))` | `vector<u64>` | Find all matching indices |
| `sum_map!<T, U>(&vec, \|x\| f(x))` | `Integer` | Sum of f(x) over elements |

All above have `_range!` variants: `all_range!(&vec, start, end, |x| ...)`.

Native functions (not macros):

| Function | Returns | Description |
|----------|---------|-------------|
| `sum<T>(&vec)` | `Integer` | Sum of vector elements |
| `sum_range<T>(&vec, start, end)` | `Integer` | Sum of elements in range |

Index-range functions (no vector input):

| Function | Returns | Description |
|----------|---------|-------------|
| `range_map!<T>(start, end, \|i\| f(i))` | `&vector<T>` | Generate vector from index range |
| `range_count!(start, end, \|i\| pred(i))` | `u64` | Count indices satisfying predicate |
| `range_sum_map!(start, end, \|i\| f(i))` | `Integer` | Sum of f(i) over index range |

Import with `use prover::ghost::*`:

| Function | Description |
|----------|-------------|
| `ghost::declare_global_mut<Key, T>()` | Declare ghost variable |
| `ghost::global<Key, T>()` | Read current value |
| `ghost::borrow_mut<Key, T>()` | Get mutable reference |

## Quick Reference: Mathematical Types (Spec-Only)

### `std::integer::Integer`
Arbitrary-precision mathematical integer. The bridge between bounded uN bit-patterns and unbounded math. **Stay in `Integer`-space inside every `ensures` and `asserts`**; cast in at the boundary with `.to_int()` / `.to_signed_int()` and cast out with `.to_uN()`.

- **In from uN (unsigned)**: `42u64.to_int()` (value-preserving, lifts u64 → mathematical integer).
- **In from uN (two's-complement)**: `bits.to_signed_int()` (interprets a uN bit-pattern as signed int in `[-(2^(W-1)), 2^(W-1) - 1]`). Use this for `bits: uN` fields in signed-int structs — see "Two's-complement signed integers in the int encoding" below.
- **Out to uN**: `n.to_u32()` / `.to_u64()` / `.to_u128()`. **Implicitly does `mod 2^W`** — i.e. truncates a wider integer to its low W bits without aborting on overflow. This is exactly the postcondition shape for `wrapping_*` / `overflowing_*`: `result == n1.to_int().add(n2.to_int()).to_u64()`.
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `mod`, `neg`, `abs`, `div_trunc`, `mod_trunc`, `div_round_up`.
  - `div` / `mod` are the mathematical (floor for positives, language-defined for negatives) variants.
  - `div_trunc` / `mod_trunc` are truncated-toward-zero, matching the Move source semantics for signed integers. **Use these for signed `div` / `mod` specs.**
  - `div_round_up` matches the `checked_div_round(.., true)` rounding helper pattern.
- **Math**: `sqrt`, `pow`.
- **Compare**: `lt`, `gt`, `lte`, `gte`.
- **Bitwise**: `bit_or`, `bit_and`, `bit_xor`, `bit_not`, `shl`, `shr`.
- **Predicates**: `is_pos`, `is_neg`.
- **Range predicates (signed)**: `n.is_i32()`, `n.is_i64()`, `n.is_i128()`. Each is `true` iff `n` is in `[-(2^(W-1)), 2^(W-1) - 1]`. **The canonical signed-overflow guard** — use directly as the abort precondition for `add` / `sub` / `mul` / `div` / `abs` / `neg_from` on `IN`:
  ```move
  asserts(num1.to_int().add(num2.to_int()).is_i32());     // signed-add doesn't overflow
  asserts(v.to_int().abs().is_i32());                     // |v| representable (rules out INT_MIN)
  asserts(num1.to_int().mul(num2.to_int()).is_i32());     // signed-mul fits
  ```
- **Type max constants**: `std::u32::max_value!()`, `std::u64::max_value!()`, `std::u128::max_value!()`. These are `Integer`-valued in spec context (no `to_int()` needed) — pair with `.lte` for the unsigned overflow guard:
  ```move
  asserts(a.to_int().add(b.to_int()).lte(std::u64::max_value!().to_int()));
  ```

### `std::real::Real`
- Convert: `42u64.to_real()`
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
- Math: `sqrt`, `exp`
- Compare: `lt`, `gt`, `lte`, `gte`
- Convert back: `to_integer`

### `std::q32::Q32`, `std::q64::Q64`, `std::q128::Q128`
Signed fixed-point types with 32/64/128 fractional bits.

- Convert: `42u64.to_q64()`, `my_integer.to_q64()`, `my_real.to_q64()`
- From fraction: `std::q32::quot(num.to_int(), den.to_int())`
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqrt`, `pow`
- Compare: `lt`, `gt`, `lte`, `gte`
- Rounding: `floor`, `ceil`, `round` (return Integer)
- Predicates: `is_pos`, `is_neg`, `is_int`
- From unsigned fixed-point: `my_uq64_64.to_q64()`, `my_fp32.to_q32()`
- `std::q32::quot(num, den)` — NOT `Q32::quot(...)` (Q32 is a struct, not enum)
- `use fun` aliases (e.g., `PriceFeed.to_q32()`) must be re-imported in each spec module

### `std::q_wad::Q_wad`
Signed fixed-point type with 18 decimal fractional digits (WAD, scale = 10^18).

- Convert: `42u64.to_q_wad()`, `my_integer.to_q_wad()`, `my_real.to_q_wad()`
- From fraction: `std::q_wad::quot(num.to_int(), den.to_int())`
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqrt`, `pow`
- Compare: `lt`, `gt`, `lte`, `gte`, `min`, `max`
- Rounding: `floor`, `ceil`, `round`, `to_int` (return Integer)
- Predicates: `is_pos`, `is_neg`, `is_int`
- Range checks: `in_range_u128`, `in_range_u256`
- Raw access: `raw` (get underlying Integer), `from_raw`

### `std::q_ray::Q_ray`
Signed fixed-point type with 27 decimal fractional digits (RAY, scale = 10^27).
Same API as `Q_wad` but with higher precision.

**Q type usage notes:**
- `use fun` aliases (e.g., `.to_q32()`, `.to_q_wad()`) must be re-imported in each spec module
- For exact ops (add/sub): use `==`
- For truncating ops (mul/div/create_from_rational): use within-epsilon checks
- Epsilon for Q32 = 1 ULP = `std::q32::quot(1.to_int(), 4_294_967_296u64.to_int())`

**Custom fixed-point types (e.g., project-specific Decimal):**
Builtin `.to_q32()` only works for `FixedPoint32`. For custom types, define a local conversion:
```move
#[spec_only]
use fun decimal_to_q_wad as Decimal.to_q_wad;
#[spec_only]
fun decimal_to_q_wad(d: Decimal): std::q_wad::Q_wad {
    std::q_wad::from_raw(d.to_scaled_val().to_int())
}
```
Then use in specs: `ensures(result.to_q_wad() == a.to_q_wad().add(b.to_q_wad()))`

## Quick Reference: Attributes

### `#[spec(...)]`
| Parameter | Description |
|-----------|-------------|
| `prove` | Verify this specification. **Required** — without it, spec silently passes without checking |
| `target = <PATH>` | Target external function |
| `include = <PATH>` | Include another spec module's behavior (makes its `prove`-flagged specs opaque) |
| `ignore_abort` | Don't check abort conditions. Prefer fixing root causes over using this |
| `no_opaque` | Inline called function bodies in proofs |
| `skip` | Skip verification (axiom-assumed). Optional reason: `skip = b"reason"` |
| `focus` | Verify only focused specs (useful for debugging) |
| `uninterpreted = <PATH>` | Strip body definition from a pure function. Requires `#[ext(pure)]` on target |
| `interpreted = <PATH>` | Per-spec override: re-inline a function marked `#[ext(uninterpreted)]` globally |
| `extra_bpl = b"path.bpl"` | Load additional Boogie axiom file. Repeatable: `extra_bpl = b"a.bpl", extra_bpl = b"b.bpl"` to load multiple files |
| `boogie_opt = b"options"` | Pass options directly to Boogie (e.g., `b"/timeLimit:300 /errorLimit:1"`) |
| `run_on = "local"` | Per-spec override: run locally even when `--cloud` is configured |

### `#[ext(...)]`
| Parameter | Description |
|-----------|-------------|
| `pure` | Pure function, usable in spec expressions |
| `no_abort` | Function never aborts |
| `axiom` | Axiomatically defined function |
| `uninterpreted` | Globally strip body definition from this pure function (per-spec override with `interpreted =`) |

### `#[spec_only(...)]`
| Parameter | Description |
|-----------|-------------|
| (none) | Spec-only item |
| `(axiom)` | Axiom definition |
| `(inv_target = <TYPE>)` | Datatype invariant |
| `(loop_inv(target = <FUNC>))` | External loop invariant |
| `(loop_inv(target = <FUNC>, label = N))` | Loop invariant for Nth loop (0-indexed) |
| `(include = specs::foo_specs)` | On a module: makes foo_specs's `prove`-flagged specs opaque in _Check mode |

---

# Project Setup & Running the Prover

## Directory structure
Specs live in a separate package: `specs/sources/` with its own `Move.toml` that depends on the source package.

## ALWAYS run from specs/
```bash
cd /path/to/project/specs && sui-prover
```
Running from the project root silently fails — specs not found, "Function does not exist" errors. The `specs/Move.toml` is a separate package. Default `--path` is `.`, so no need to specify it.

## Use `--trace` when debugging
Always include `--trace` when iterating on specs. Shows the call trace to the location of the condition failure.
```
🔄 test::double_spec_Check
    ⊢  $42_test_double$verify
    📌 split 1/2: prover::ensures does not hold (at ./sources/test.move:11)
       → double (at ./sources/test.move:20)
       → add_spec (at ./sources/test.move:7)
    📌 split 2/2: prover::ensures does not hold (at ./sources/test.move:21)
✅ test::double_spec_Check
```
The `→` arrows show the call chain leading to each assert — useful when one spec calls another (e.g., `double_spec` calls `add` whose spec is `add_spec`).

## Use `-v` / `--verbose` to see spec composition
`--verbose` shows the spec dependency tree — which callee specs the target relies on:
```
🔄 test::quad_spec_Check
✅ test::quad_spec_Check
└── 🎯test::quad
   └── test::double_spec
```
Here `quad_spec` verifies `quad` (🎯), whose body calls `double` — used opaquely via `double_spec`. The tree stops at `double_spec` because the prover uses its postconditions, not its implementation (won't descend to `add_spec`).

## Run prover invocations sequentially
Don't run multiple sui-prover instances in parallel — resource-intensive.

## CLI reference
```bash
sui-prover                              # Run from specs directory (default --path is .)
sui-prover --trace                      # Show call trace to condition failure location; also lists timeouts at the end (always use when debugging)
sui-prover --verbose                    # Detailed output
sui-prover --timeout 60                 # Set timeout in seconds (default 45)
sui-prover --force-timeout              # Kill Boogie if its internal timeout breaks (always pair with --timeout)
sui-prover --functions <pattern>        # Filter functions; pattern is <function>, <module>::<function>, or <package>::<module>::<function> (repeatable)
sui-prover --modules <pattern>          # Filter modules; pattern is <module> or <package>::<module> (repeatable)
sui-prover --keep-temp                  # Keep generated .bpl files for inspection
sui-prover --generate-only              # Generate Boogie only (no solving)
sui-prover --dump-bytecode              # Dump bytecode
sui-prover --skip-spec-no-abort         # Skip SpecNoAbortCheck
sui-prover --skip-fun-no-abort          # Skip fun no-abort checking
sui-prover --split-paths <N>            # Split verification paths
sui-prover --no-bv-int-encoding         # Encode integers as bitvectors (for specs with heavy shifts/bitwise carry; see Bitvector encoding section)
sui-prover --cloud                      # Use cloud backend (requires prior config)
sui-prover --stats                      # Coverage statistics
```

## Debugging workflow

1. Add `#[spec(prove, focus)]` to the spec you're working on
2. Run `sui-prover`
3. Read the error:
   - **"code aborts"** → missing asserts. Add asserts for all abort paths in the function and nested calls
   - **"ensures does not hold"** → postcondition is wrong or too strong. Check the logic, look at the counterexample
   - **"assert does not hold"** → the asserts conditions are too strong. Check the logic, look at the counterexample
   - **Timeout** → complexity issue. Try `uninterpreted`, `boogie_opt`, or simplifying. If the function is dominated by bitwise / shift ops (carry-chain adders, arithmetic right shift, bit-mask packing), also consider moving the spec to a sibling `specs-bv` package proved with `--no-bv-int-encoding` (see Bitvector encoding section)
4. Once passing with `focus`, remove `focus` and verify full suite

---

# The Three Verification Checks

Every spec generates three `.bpl` files — one per `AssertsMode`. All three must pass.

## 1. `_Check` — Main verification

- `asserts` → `requires` (assumed as preconditions)
- `ensures` → asserted
- Abort points → `assert !$abort_flag`
- Callee bodies **inlined**, except `prove`-flagged specs from `include`d modules which are opaque (havoced + postconditions assumed via their Opaque procedures)

## 2. `_Assume` — Asserts are necessary for abort-freedom

For specs **without** `asserts` calls (and without `ignore_abort`): passes vacuously (no verification procedure emitted).

For specs **with** `asserts` calls: the $verify procedure runs the implementation with abort points assumed away (`assume !$abort_flag`), `asserts` dropped from the body, `ensures` dropped, then **calls the `$asserts` inline procedure** which asserts the `asserts` conditions.

This checks: **if the code doesn't abort, do the `asserts` conditions hold?** i.e., `¬abort → P`.

Combined with _Check (which checks `P → ¬abort ∧ Q`), together they prove:
- `P ⟺ ¬abort` — the asserts conditions are exactly when the function doesn't abort
- `P → Q` — under non-abort conditions, the postconditions hold

For specs **with** `ignore_abort`: same as above, but `ensures` are also **asserted** (not dropped). This additionally checks that postconditions hold on all non-aborting paths.

## 3. `_SpecNoAbortCheck` — Can spec code abort?

- `ensures` and `asserts` → `requires` (assumed, not checked)
- Target call havoced, postconditions assumed
- Abort points → `assert !$abort_flag`
- Checks that the spec body code itself doesn't abort

---

# Writing Specs: requires / ensures / asserts

## `asserts` — proves abort on violation
Use when the code actually checks the condition (e.g., `assert!(condition)`, overflow checks, permission checks). Stronger than `requires` — proves the function **will abort** if violated.

## `requires` — assumes condition holds
Use for invariants callers must maintain but aren't runtime-checked. With `ignore_abort`, the distinction with `asserts` is moot and `requires` better documents intent.

## `ensures` — postcondition
Must hold after execution. For opaque specs, **every** property the caller needs must be explicitly ensured — nothing is assumed preserved automatically.

## `ignore_abort` — last resort
Never add `ignore_abort` unless explicitly asked. Specs should prove abort-freedom via `asserts`. Using `ignore_abort` skips the abort-freedom proof and weakens the verification.

## `clone!` for before/after comparisons
```move
let old_vault = clone!(&vault);
// ... mutations happen ...
ensures(vault.balance() >= old_vault.balance());
```
`clone!(&mut_ref)` captures a snapshot; returns `&T`. Import from `prover::prover`.

## Overflow/underflow asserts with `.to_int()`
Use arbitrary-precision `Integer` arithmetic in asserts to check bounds:
```move
// addition overflow
asserts(a.to_int().add(b.to_int()).lte(std::u64::max_value!().to_int()));

// subtraction underflow
asserts(a.to_int().gte(b.to_int()));

// multiplication overflow
asserts(a.to_int().mul(b.to_int()).lte(std::u64::max_value!().to_int()));

// division by zero
asserts(divisor != 0);
```

## `df::exists_with_type` and `bag::contains_with_type`
For dynamic field and bag access, use the typed variants — the untyped `df::exists_` / `bag::contains` don't connect with `borrow` in the prover:
```move
// WRONG: prover can't connect exists_ to borrow
asserts(df::exists_<Key>(&self.id, key));
let val = df::borrow<Key, Value>(&self.id, key);

// CORRECT: typed variant connects
asserts(df::exists_with_type<Key, Value>(&self.id, key));
let val = df::borrow<Key, Value>(&self.id, key);
```
Same for bags: use `bag::contains_with_type<K, V>` not `bag::contains<K>`.

## Early return guards
When the implementation has `if (x == 0) { return }`, asserts for code after the guard must be conditional:
```move
fun process_spec(a: u64, b: u64) {
    if (a != 0) {
        asserts(b.to_int().div(a.to_int()).lte(std::u64::max_value!().to_int()));
    };
    module::process(a, b);
}
```

## Conditional ensures
Use `if (condition) { ensures(...) }` for branch-specific postconditions:
```move
// Idempotency: when already up-to-date, nothing changes
if (forall!<TypeName>(|t| borrow_index_matches(*t, old_debts, market))) {
    ensures(forall!<TypeName>(|t| debt_unchanged(*t, old_debts, new_debts)));
};
```

---

# Pure Functions

## Basics
Mark with `#[ext(pure)]`. Makes functions usable in spec expressions.

## `test_only` vs `spec_only` for helpers
- `#[test_only, ext(pure)]` — add to **implementation modules** for private field accessors. Visible in tests AND specs. This is the standard way to expose private struct fields to specs.
- `#[spec_only, ext(pure)]` — add to **spec modules** for spec-only helpers (predicates, computations). Visible only to the prover. Cannot be placed in implementation modules.

## Accessor specs make `$pure` opaque
When a getter function (e.g., `get_field()`) has a spec (even `skip`), its `$pure` Boogie function becomes **uninterpreted** (no `{:inline}` body). Without a spec, `$pure` gets `{:inline}` with direct field access. This means the prover can't connect `get_field(result)` to `result.field_name`.

**Fix**: Move getter specs into a separate spec module file. Don't include that module from specs that need the getters' `$pure` to be inline.

## Functions that are NOT pure
- `wit_table::borrow` — use `old_bs.get_table().borrow(t)` instead (`get_table` is `#[test_only, ext(pure)]`, `table::borrow` is built-in pure)
- `obligation.collateral_types()` / `obligation.debt_types()` — call `wit_table::keys` which has branching. Pass vector as parameter instead.
- Functions with loops cannot be `ext(pure)` (branching is fine)
- An `#[ext(pure)]` function cannot call a function that has a loop invariant attached (even if that callee is itself marked `ext(pure)`). The loop-invariant function must be called from non-pure code; expose the post-loop result to specs via a different route.

---

# Include & Opacity Mechanism

## What `include` does
`include` tells the prover to pull in specs from another module and use them as **opaque abstractions** for their target functions.

Two levels:
- **Module-level**: `#[spec_only(include = specs::foo_specs)]` on a module — all specs in `foo_specs` are included for every spec in this module.
- **Function-level**: `#[spec(prove, include = specs::foo_specs::bar_spec)]` on a spec — only `bar_spec` is included for this spec.

## Why it matters
Without `include`, a spec from another module is **not known** to the prover. The target function's body is inlined directly — the prover sees the full implementation. This can be slow (Z3 expands everything) or necessary (if you need implementation details).

With `include`, the included spec's target function becomes **opaque**: the prover havoces the call's return values and assumes the spec's `ensures` postconditions. The implementation body is not seen. This is faster (postconditions are simpler than full implementations) but requires the postconditions to be strong enough.

## The opacity rule
A callee is opaque when ALL of these hold:
1. The callee has a spec (with or without `prove`)
2. That spec is `include`d in the verifying module/function
3. The spec does NOT have `no_opaque`

Otherwise the callee's implementation is **inlined** (transparent).

## Performance impact
Critical for complex math. Example: without include, Z3 tries to connect bitvector arithmetic with Q32 pure functions (times out). With include, Z3 uses Q32 postconditions directly (2s vs timeout).

---

# Loop Invariants

## Bool pattern
```move
#[spec_only(loop_inv(target = my_spec))]
#[ext(no_abort)]
fun loop_inv(i: u64, n: u64, sum: u128): bool {
    i <= n && sum == (i as u128) * ((i as u128) + 1) / 2
}
```

## Void pattern (with ensures)
```move
#[spec_only(loop_inv(target = my_spec))]
#[ext(no_abort)]
fun loop_inv(i: u64, n: u64, s: u128) {
    ensures(i <= n);
    ensures(s == (i as u128) * ((i as u128) + 1) / 2);
}
```

Void loop invariants use `ensures` for each condition instead of returning a single `bool` with conditions joined by `&&`. This is more flexible — each condition is a separate assertion, so failure messages pinpoint which condition broke. Cannot mix: a bool-returning loop invariant must NOT use `ensures` (this is a compile error).

- Parameters must match loop variables exactly
- For multiple loops in one function, use `label = N` (0-indexed)

## Old state with `__old_` prefix
For values captured before the loop, prefix the parameter name with `__old_`:
```move
#[spec_only(loop_inv(target = my_spec))]
#[ext(no_abort)]
fun loop_inv(market: &Market, __old_market: &Market, i: u64): bool {
    // market is current value, __old_market is value at loop entry
    market.total_debt() >= __old_market.total_debt()
}
```
- Use `&T` not `&mut T` for mutable references in loop invariant parameters

## The complete loop invariant chain
For loop specs with opaque callees, every link must be present:

1. **`requires`** on the outer spec (precondition assumed)
2. **Loop invariant holds on entry** (prover checks)
3. **Callee's `requires` are satisfied** by loop invariant (prover checks)
4. **Callee's `ensures` provide properties back** (opaque spec assumed)
5. **Loop invariant preserved after body** (prover checks)
6. **Outer `ensures` follow** from loop invariant after loop exit (prover checks)

If any link is missing, verification fails.

## Frame conditions for callees
Opaque callee specs must explicitly `ensures` every property the loop invariant depends on. Nothing is assumed preserved:
```move
ensures(vault.get_vault_paused() == old_vault.get_vault_paused());
ensures(vault.get_vault_rate() == old_rate);
ensures(vault.get_withdrawal_queue().len() == old_vault.get_withdrawal_queue().len() - 1);
```

## Include and loop invariants interaction
When including callee specs with `requires` preconditions about unconstrained state, `_Assume` mode can't establish those `requires` after havoc. Fix: remove the `include` (callee gets inlined in `_Check`, and its `requires` aren't demanded in `_Assume`).

## `all!`/`sum_map!` in loop invariants
Using `all!`/`sum_map!` wrappers in loop invariants causes `_Assume` regression (Z3 timeout). Wrappers work fine in spec `requires`/`ensures` — only keep inline expressions in loop invariants.

## Loop havoc
Boogie havoces all mutable variables modified in the loop body. After havoc, the prover only knows what the loop invariant states — all other relationships are lost.

**Havoced:** all local mutable variables modified in the loop.
**Not havoced:** immutable references (`&T`), values not modified in the loop, `__old_` snapshots.

If an abort guard inside the loop depends on a relationship between a havoced variable and an unhavoced one, the prover can't verify it without a loop invariant that re-establishes the relationship.

Example — semantically safe but unprovable without invariant:
```move
let keys = vec_map::keys(&map);
while (keys.length() > 0) {
    let key = keys.pop_back();       // keys is havoced
    let val = vec_map::get(&map, &key);  // aborts if key not in map
    // prover lost the knowledge that keys came from vec_map::keys(&map)
}
```

## Safe-up-to-i invariant (asserts-completeness across a loop)
When a function with a loop has a function-level `asserts`, the prover checks two directions:
- **`_Check`**: under `asserts`, the body never aborts.
- **`_Assume`**: when `!asserts`, the function MUST abort.

`_Assume` is the subtle one — the loop is summarized by one symbolic body iteration, so the invariant must encode enough to derive a contradiction at loop exit when `!asserts`. Diagnostic: `prover::ensures does not hold` pointing at the `asserts` line in a function with a loop.

The fix is the **safe-up-to-i invariant**: a clause that is **vacuous at iteration 0** and **collapses to the asserts at loop exit**. As the loop runs, it carries "the per-iteration safety condition has held for all completed iterations".

For a forall asserts, use a `j >= i` guard:
```move
asserts(forall!(|j| safe(*j, v)));

invariant!(|| {
    ensures(i <= v.length());
    ensures(forall!(|j| j >= i || safe(*j, v)));   // safe-up-to-i
});
```
At `i = 0` the forall is vacuous; at `i = v.length()` it equals the asserts and contradicts `!asserts`, so the prover concludes the loop must have aborted.

For a scalar asserts (e.g. `asserts(n <= 100)` with body `assert!(i < 100)`), the safe-up-to-i analog is just a scalar bound:
```move
ensures(i <= 100);   // vacuous at i=0; equals asserts at i=n
```

For mutating quantified state, apply the guard to a cloned snapshot (`old_v`), and add a separate `element_state` invariant relating current `v` to `old_v`:
```move
let old_v = clone!(v);
invariant!(|| {
    ensures(forall!(|j| element_state(*j, i, v, old_v)));
    ensures(forall!(|j| j >= i || safe(*j, old_v)));   // safe-up-to-i on the stable witness
});
```

For nested loops, express the outer asserts as a forall over the outer iteration index and accumulate the safe-up-to-i form in the outer invariant.

Pitfall: writing `forall!(|j| safe(*j, v))` (no guard) in the invariant. That's exactly the asserts; under `!asserts` it cannot hold on loop entry because nothing abort-able runs before the loop. The `j >= i` guard is what splits the obligation across iterations.

---

# Quantifiers & Lambdas

## Lambda syntax
The lambda MUST be a single `#[ext(pure)]` function call — `|t| f(*t, ...)`. Complex block expressions (`|t| { ... }`) are rejected:
> "Invalid quantifier macro pattern: expected a lambda function, but found an inlined expression"

## `forall!` / `exists!` quantifier variable type
The Boogie quantifier variable type is the **first parameter** of the helper function:
```move
// x has type TypeName in the Boogie quantifier
fun f(t: TypeName, old_bs: &WitTable, new_bs: &WitTable): bool { ... }
forall!<TypeName>(|t| f(*t, old_bs, new_bs))
```
For `all!`/`any!` (vector-based), the quantifier variable is always `int` (an index into the vector).

## Lambda parameter must not be used as a function argument
The lambda parameter cannot be passed directly to external function calls:
```move
// BROKEN: "lambda parameter is used externally"
exists!<u64>(|j| helper(*j, i, u[*j], v[i]))
```

**Fix**: Wrap in an `#[ext(pure)]` helper function that does the external calls:
```move
#[spec_only, ext(pure)]
fun request_owner_has_account(r: &Request, vault: &Vault): bool {
    vault.has_account(r.owner())
}
all!(&requests, |r| request_owner_has_account(r, vault))
```

Note: `#[ext(pure)]` functions can call non-pure functions (e.g., `vector::borrow`, `table::borrow`) as long as:
- No mutable references are involved
- Aborts are guarded (e.g., `borrow` is preceded by a `contains` check)

---

# Speed & Optimization

## `uninterpreted` functions — the biggest speedup
```move
#[spec(prove, target = ..., uninterpreted = module::pure_function)]
```
Strips body definition so Z3 doesn't expand pure functions. Typical speedup: **300s+ to 5s**.

Requires the function to be `#[ext(pure)]`.

**Caveat**: If the struct is mutated between the `requires` and the point where the uninterpreted function is evaluated (e.g., loop entry after a field mutation), the prover can't tell the function would return the same value. Don't mark a function uninterpreted if its input struct changes between pre/post states.

### Layered proof strategy

| Layer | What | How |
|-------|------|-----|
| 0 | Leaf functions | Prove with full expansion |
| 1 | Helper wrappers | Prove with leaf uninterpreted |
| 2 | Mid-level functions | Prove with wrappers uninterpreted |
| 3 | Top-level evaluators | Prove with everything below uninterpreted |

## `extra_bpl` for Boogie-level axioms
Provide Boogie-level axioms the int encoding cannot derive itself. The file can be wired per-spec (`extra_bpl = b"file.bpl"` in the `#[spec]` attribute) or **project-wide** as `specs/prelude_extra.bpl` next to `Move.toml` — sui-prover auto-loads `prelude_extra.bpl` from the package root.

This is **the workhorse for bitwise-heavy code that should NOT escape to `specs-bv/`** — masking, XOR-as-complement, OR with the sign bit, sign-bit extraction via shift, and the u8 truth table feeding the signed-overflow detector. Reach for these axioms FIRST; the BV package is the fallback for the genuinely solver-hostile cases (ripple-carry adder, arithmetic right shift).

### Canonical `prelude_extra.bpl` for two's-complement / bitmask code

The following set covers `(x ^ MAX_uN)`, `(x | 2^(W-1))`, `(x & LO_MASK)`, `(x & HI_MASK)`, `(x >> (W-1)) as u8`, and the `u8` AND truth table needed by `(sign(a) & sign(b) & u8_neg(sign(sum)))`-style signed-overflow detectors. Copy verbatim into `specs/prelude_extra.bpl`; trim widths you don't use.

```boogie
// ----- XOR with the type's all-ones mask = arithmetic complement
axiom (forall x: int :: {$xorInt'u8'(x, $MAX_U8)}    $xorInt'u8'(x, $MAX_U8)    == $MAX_U8    - x);
axiom (forall x: int :: {$xorInt'u32'(x, $MAX_U32)}  $xorInt'u32'(x, $MAX_U32)  == $MAX_U32   - x);
axiom (forall x: int :: {$xorInt'u64'(x, $MAX_U64)}  $xorInt'u64'(x, $MAX_U64)  == $MAX_U64   - x);
axiom (forall x: int :: {$xorInt'u128'(x, $MAX_U128)} $xorInt'u128'(x, $MAX_U128) == $MAX_U128 - x);

// ----- OR with the sign-bit constant = addition when the input lacks that bit
const $POW_TWO_31:  int;  axiom $POW_TWO_31  == 2147483648;                               // 2^31
const $POW_TWO_63:  int;  axiom $POW_TWO_63  == 9223372036854775808;                      // 2^63
const $TWO_POW_127: int;  axiom $TWO_POW_127 == 170141183460469231731687303715884105728;  // 2^127

axiom (forall x: int :: {$orInt'u32'(x, $POW_TWO_31)}
    $orInt'u32'(x, $POW_TWO_31)  == if x < $POW_TWO_31  then x + $POW_TWO_31  else x);
axiom (forall x: int :: {$orInt'u64'(x, $POW_TWO_63)}
    $orInt'u64'(x, $POW_TWO_63)  == if x < $POW_TWO_63  then x + $POW_TWO_63  else x);
axiom (forall x: int :: {$orInt'u128'(x, $TWO_POW_127)}
    $orInt'u128'(x, $TWO_POW_127) == if x < $TWO_POW_127 then x + $TWO_POW_127 else x);

// ----- AND with the LO/HI half-masks = mod / divmod by the width's half-power
const $LO_64_MASK: int;  axiom $LO_64_MASK == 18446744073709551615;                       // 2^64 - 1
const $HI_64_MASK: int;  axiom $HI_64_MASK == 340282366920938463444927863358058659840;    // 2^128 - 2^64
const $TWO_POW_64: int;  axiom $TWO_POW_64 == 18446744073709551616;                       // 2^64

axiom (forall x: int :: {$andInt'u128'(x, $LO_64_MASK)} $andInt'u128'(x, $LO_64_MASK) == x mod $TWO_POW_64);
axiom (forall x: int :: {$andInt'u128'(x, $HI_64_MASK)} $andInt'u128'(x, $HI_64_MASK) == (x div $TWO_POW_64) * $TWO_POW_64);

const $LO_128_MASK: int; axiom $LO_128_MASK == 340282366920938463463374607431768211455;                                          // 2^128 - 1
const $HI_128_MASK: int; axiom $HI_128_MASK == 115792089237316195423570985008687907852929702298719625575994209400481361428480;  // 2^256 - 2^128
const $TWO_POW_128: int; axiom $TWO_POW_128 == 340282366920938463463374607431768211456;                                          // 2^128

axiom (forall x: int :: {$andInt'u256'(x, $LO_128_MASK)} $andInt'u256'(x, $LO_128_MASK) == x mod $TWO_POW_128);
axiom (forall x: int :: {$andInt'u256'(x, $HI_128_MASK)} $andInt'u256'(x, $HI_128_MASK) == (x div $TWO_POW_128) * $TWO_POW_128);

// ----- Right shift by (W-1) = div by 2^(W-1) (sign extraction `(v.bits >> 127) as u8`)
axiom (forall x: int :: {$shr(x, 127)} $shr(x, 127) == x div $TWO_POW_127);
axiom (forall x: int :: {$shr(x, 128)} $shr(x, 128) == x div $TWO_POW_128);

// ----- u8 AND truth table for {0, 1, 254, 255}
// Covers (sign(a) & sign(b) & u8_neg(sign(sum))) + (u8_neg(sign(a)) & u8_neg(sign(b)) & sign(sum))
// where sign() returns 0 or 1 and u8_neg() returns 254 or 255.
axiom ($andInt'u8'(0,   0) == 0);     axiom ($andInt'u8'(0,   1) == 0);
axiom ($andInt'u8'(0, 254) == 0);     axiom ($andInt'u8'(0, 255) == 0);
axiom ($andInt'u8'(1,   0) == 0);     axiom ($andInt'u8'(1,   1) == 1);
axiom ($andInt'u8'(1, 254) == 0);     axiom ($andInt'u8'(1, 255) == 1);
axiom ($andInt'u8'(254,   0) == 0);   axiom ($andInt'u8'(254,   1) == 0);
axiom ($andInt'u8'(254, 254) == 254); axiom ($andInt'u8'(254, 255) == 254);
axiom ($andInt'u8'(255,   0) == 0);   axiom ($andInt'u8'(255,   1) == 1);
axiom ($andInt'u8'(255, 254) == 254); axiom ($andInt'u8'(255, 255) == 255);
```

**Mangled-name pattern.** Boogie's bitwise primitives are mangled by type: `$xorInt'uN'`, `$orInt'uN'`, `$andInt'uN'`, `$shl(x, k)`, `$shr(x, k)`. The `'uN'` tag matches the Move operand type at the call site — `(n & LO_64_MASK)` on a `u128` becomes `$andInt'u128'(n, LO_64_MASK)` in Boogie, so the matching axiom must target `$andInt'u128'`.

**Triggers matter.** The `{...}` annotation on each `forall` is the trigger pattern; Z3 instantiates the axiom every time it sees a matching term. Without triggers, Z3 won't find the axiom at the right place. With triggers, you may need `EAGER_THRESHOLD=100` on hot specs (see `boogie_opt` recipes below).

**Coverage.** Once this prelude is in `specs/prelude_extra.bpl`, the int encoding can discharge:
- `(sum & LO_64_MASK) as u64 == sum mod 2^64` (wrap-around in `math_uN::overflowing_*` / `carry_add`)
- `(n & HI_64_MASK) >> 64 == n / 2^64` (hi/lo half extraction)
- `(v.bits >> (W-1)) as u8 ∈ {0, 1}` (sign extraction)
- `u32_neg(v) == MAX_U32 - v`, `u8_neg(v) == 255 - v` (XOR complement helpers)
- `(u32_neg(v) + 1) | (1 << 31)` (the `neg_from` two's-complement constructor)
- `(sign(a) & sign(b) & u8_neg(sign(sum))) + ...` (the branchless signed-overflow detector used by `add` / `sub`)

What it does NOT cover: ripple-carry adder loops (no loop invariant in axioms), arithmetic right shift on signed (needs `native fun ashr` + Boogie procedure binding), and nonlinear bv-wide multiplication. Those go to `specs-bv/`.

### Per-spec extra_bpl

For domain-specific axioms (e.g. a CLMM math helper), use the attribute form:
```move
#[spec(prove, target = ..., extra_bpl = b"clmm_math.bpl")]
```
The `.bpl` file lives alongside spec sources and is loaded only for this spec — useful for axioms you don't want global.

## `boogie_opt` reference

| Option | What it does |
|--------|-------------|
| `vcsSplitOnEveryAssert` | Split verification conditions on every assert |
| `vcsMaxKeepGoingSplits:N` | Enable path splitting (N >= 2 to activate; no splitting without it) |
| `{:isolate_paths}` | Function-level path isolation: every `ensures`/`asserts` in the spec gets `{:isolate "paths"}`. Pair with `boogie_allow_path_isolation()` inside the body to mark which `if` branches to isolate |
| `useArrayAxioms` | Use array axioms (helps with table access and loops) |
| `vcsCores:1` | Single-core for deterministic behavior |
| `prune:1` | Drop unreachable functions (per-struct/per-table `$IsEqual` etc.) before SMT emission. May help by deleting unused functions; may also hurt if a proof relied on a now-pruned function. A/B before keeping. |
| `proverOpt:O:smt.MBQI=false` | Disable model-based quantifier instantiation |
| `proverOpt:O:smt.QI.EAGER_THRESHOLD=N` | Tune eager quantifier instantiation threshold |
| `proverOpt:O:smt.QI.LAZY_THRESHOLD=N` | Tune lazy quantifier instantiation threshold |
| `proverOpt:O:smt.random_seed=N` | Try different seeds when proof is seed-sensitive |

Combine multiple: `boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms"`

**Recommended combo for `forall!`/iterator specs:**
```move
boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false"
```
`smt.MBQI=false` is essential for `forall!` to work reliably.

### Recipes for the patterns the prelude axioms drive

When `specs/prelude_extra.bpl` is loaded, certain spec shapes need Z3 nudged to actually use the axioms. The hot recipes:

| Recipe | Where to use it |
|---|---|
| `boogie_opt = b"vcsSplitOnEveryAssert vcsFinalAssertTimeout:300"` | Chain-arithmetic with both an abort precondition and a value postcondition under one spec — `iN::mul`, similar wide multiplications. The split decouples the abort dispatch from the postcondition dispatch. |
| `boogie_opt = b"vcsSplitOnEveryAssert proverOpt:O:smt.QI.EAGER_THRESHOLD=100"` | `iN::shl` and other shift specs whose ensures invoke the `Integer.shl(...).to_uN().to_signed_int()` chain. Eager instantiation forces the axiom triggers to fire before Z3 gives up. |
| `boogie_opt = b"proverOpt:O:smt.QI.EAGER_THRESHOLD=100"` | Mask / hi-lo extractors (`math_u128::lo`, `from_lo_hi`). The `$andInt'u128'(x, LO_64_MASK) == x mod TWO_POW_64` trigger needs eager firing because the spec body is short and Z3 otherwise defers. |
| `boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false"` | `forall!` / iterator specs (unchanged from above). |

**Diagnosis cue.** If `_Check` times out on a spec that uses the prelude axioms, raise `EAGER_THRESHOLD` first (`100`, then `200`). If the spec has multiple assertions, add `vcsSplitOnEveryAssert`. Combine only when both fail individually.

CLI alternative: `--split-paths <N>` adds `vcsSplitOnEveryAssert` + `verifySeparately` + `vcsMaxKeepGoingSplits:N` globally.

## Bitvector encoding (`--no-bv-int-encoding`)

The default integer encoding loses precision on bitwise ops (`&`, `|`, `^`, `<<`, `>>`) — Z3 treats them as uninterpreted and times out. The `--no-bv-int-encoding` flag switches the encoding to SMT bit-vectors so those ops are native. **Despite the name, it turns BV encoding ON.** It is invocation-wide and slower for plain arithmetic, so apply it only to the few specs that need it via a **sibling package**.

### Reach for it when

**Before reaching for BV, try `specs/prelude_extra.bpl` first.** A 60-line Boogie axiom file (XOR-as-complement, OR-with-sign-bit, AND-with-mask, shr-by-(W-1), u8 AND truth table — see the canonical set under `extra_bpl`) makes the int encoding handle ~80% of bitwise constructs: bitmask wrap-around `(x & MASK) as uN`, sign extraction `(v.bits >> (W-1)) as u8`, `neg_from`'s OR-with-sign-bit, the branchless signed-overflow detector in `add` / `sub`, and the hi/lo half extractors. **These all stay in `specs/`** — BV is overkill for them.

The genuine BV cases are narrow:

- The function is a **ripple-carry adder** — `while (carry != 0) { sum ^= carry; carry = (a & b) << 1; }`. The loop invariant `(num1 + num2) mod 2^w == (sum + carry) mod 2^w` (cast both sides up by one width) is a per-step relation, not an algebraic fact you can pre-axiomatize. **No prelude axiom rescues this — go BV.**
- The function is **arithmetic right shift** on a two's-complement value (`shr` on `IN`) — int encoding has no `ashr` semantics. Even with axioms, the OR-with-mask in the signed branch can't be cleanly tied to a mathematical `Integer.shr` without a native binding. **Go BV with `native fun ashr` backed by Boogie's `$AShr'BvN'`.**
- A spec is dominated by bitwise mixing and the int-encoded run still times out **after the prelude axioms are in place**.

If the property is algebraic (overflow conditions, comparisons, conversions, mul/div), **stay in the default `specs/` package** — those are easier to write, compose with the rest of the suite, and prove faster under int encoding.

### Set it up without asking

When you are running under "prove this project" (or any spec-improvement loop) and one of the symptoms above fires, set up `specs-bv/` and port the symptomatic spec **without prompting the user**. The decision to use bitvector encoding for genuinely bitwise code is a tool selection inside an authorized task, not a separate scope question. Follow the playbook below end to end and report results. Stop to ask only when (a) a `prelude_extra.bpl` Boogie glue (`$AShr'BvN'`, custom intrinsic) is required, or (b) the BV port itself fails after a real attempt with a correctly placed loop invariant.

When ports run into Z3 limits — chiefly **nonlinear multiplication on wide bit-vectors** (e.g. `bv128 = bv64 * bv64`) and **long signed-op chains that compound multiple bitwise primitives** — prove the load-bearing core (`wrapping_add`, `sign`, `or`, `and`, `shl`, masked extractors) and leave the dependent operations as `ignore_abort` in `specs/`, with a comment naming the limit. That is the documented limitation, not a gap.

### Setup playbook

1. **Create a sibling package** `specs-bv/` next to `specs/`:
   ```
   project/
     sources/                     <- main package
     specs/                       <- int-encoded specs (most things)
     specs-bv/
       Move.toml                  <- standalone, no dep on the main package
       sources/<module>.move      <- one Move file per source module
       prelude_extra.bpl          <- optional, for native fn glue (e.g. ashr)
   ```
   `Move.toml`:
   ```toml
   [package]
   name = "<Project>SpecsBV"
   edition = "2024.beta"
   [addresses]
   <project>_specs_bv = "0x0"
   ```

2. **In the BV package, redeclare locally** the struct and the source function you're proving. The BV package does **not** depend on the main package — it gives the bitvector encoding a clean view of just what's needed. Strip the type to its `bits` field:
   ```move
   public struct I32 has copy, drop, store { bits: u32 }
   ```

3. **In the main `specs/` package, declare the BV-bound spec without `prove`** so the int-encoded run skips it, with a comment pointing at the sibling:
   ```move
   /*
    ⚠️ Proved in a separate package as it requires a custom prover configuration.
   */
   #[spec(target = wrapping_add)]   // no `prove`
   public fun wrapping_add_spec(num1: I32, num2: I32): I32 { ... }
   ```

4. **Run both packages, with different flags:**
   ```bash
   cd specs    && sui-prover
   cd specs-bv && sui-prover --no-bv-int-encoding
   ```

### Writing rules for BV specs

- Phrase everything on `.bits`, not on `to_int()` / `Integer`. The whole point is to stay in bit-vector land.
- Wrap-arithmetic post-condition pattern: cast up by one width, add, mod by `2^w`, cast back.
  ```move
  ensures(result.bits == (((num1.bits as u64) + (num2.bits as u64)) % (1 << 32)) as u32);
  ```
- For a ripple-carry adder loop, attach an `invariant!` that equates `(num1 + num2) mod 2^w` with `(sum + carry) mod 2^w` (cast both sides up by one width to express the addition without overflow):
  ```move
  invariant!(|| {
      ensures(
          ((num1.bits as u64) + (num2.bits as u64)) % (1 << 32)
              == ((sum    as u64) + (carry    as u64)) % (1 << 32),
      );
  });
  ```
- For arithmetic right shift, declare a native intrinsic in Move and back it with Boogie's `$AShr'BvN'` in `prelude_extra.bpl`. The Boogie procedure name is `$<address>_<module>_<function>`; for `0x0` it's `$0_<module>_<function>`:
  ```move
  public native fun ashr(x: u32, y: u32): u32;

  #[spec(prove, target = shr)]
  public fun shr_spec(v: I32, shift: u8): I32 {
      asserts(shift < 32);
      let result = shr(v, shift);
      ensures(result.bits == ashr(v.bits, shift as u32));
      result
  }
  ```
  ```boogie
  procedure {:inline 1} $0_i32_ashr($t0: bv32, $t1: bv32) returns ($ret0: bv32) {
    $ret0 := $AShr'Bv32'($t0, $t1);
  }
  ```

### What stays in `specs/`

Everything else: overflow predicates (`add`, `sub`, `mul`, `div`), comparisons, type conversions, `cmp`/`eq`, `or`/`and` over short widths, abort conditions stated against `Integer`. Int encoding is the right model whenever the proof can be discharged algebraically.


## User-controlled path splitting (`{:isolate_paths}` + `boogie_allow_path_isolation()`)
A key lever for handling timeouts. Splits the VC on **paths** — orthogonal to `vcsSplitOnEveryAssert`, which splits on **boogie assert checks** — and the two compose: using `vcsSplitOnEveryAssert` together with `{:isolate_paths}` yields one VC per (assert × path) pair, each much simpler than the unsplit VC.

Two-part pattern:

1. On the spec, enable function-level path isolation:
   ```move
   #[spec(prove, boogie_opt = b"{:isolate_paths}")]
   ```
   Every `ensures`/`asserts` in the spec body gets `{:isolate "paths"}` on its Boogie assert.

2. Inside the function body (spec or implementation), mark the `if`s you want split by calling `boogie_allow_path_isolation()` immediately before them:
   ```move
   boogie_allow_path_isolation();
   if (cond) { ... } else { ... };
   ```
   Each marked `if` is annotated `{:allow_path_isolation}`, so Boogie forks the VC on its branches.

Combine with `vcsSplitOnEveryAssert` when one alone isn't enough:
```move
#[spec(prove, boogie_opt = b"vcsSplitOnEveryAssert {:isolate_paths}")]
```

When to reach for it:
- A spec times out and `vcsSplitOnEveryAssert` alone doesn't help.
- The timeout concentrates around a specific branching structure (match/if-else ladder, conditional update block).
- You want targeted splits at specific `if`s rather than the global `--split-paths` hammer.

`boogie_split_here()` / `boogie_focus()` are statement-level cousins — cut the VC at a specific program point (`split_here`) or split "paths through here" vs "not" (`focus`). Reach for them when the problem isn't a branch but a specific statement you want to isolate.

## Z3 non-determinism
Z3 is non-deterministic — the same spec can fail in 7s or pass in 300s across runs. Use generous timeouts (300s+) for complex specs with loops.

---

# Advanced Patterns

## Lemma functions for injecting proof hints
Create empty-bodied functions with specs to inject facts at specific program points:
```move
fun lemma_div_defined<T,R>(vault: &Vault<T,R>, shares: u64) {}
```
With a spec that has `requires` and `ensures`. Because the body is empty, the spec is trivially satisfiable. Call the lemma right before the point where the prover needs the fact.

- **Source code lemma** (in main module): Can be called mid-function to inject hints at exact program points.
- **Spec-only lemma**: Keeps source clean but can only be called from spec code.

## Ghost state for tracking side effects
For functions using `transfer::public_transfer` or other side-effecting operations:
```move
ghost::declare_global_mut<SpecTransferAddress, address>();
ghost::declare_global_mut<SpecTransferAddressExists, bool>();
```

## Wrapping arithmetic in specs
For operations that intentionally wrap (e.g., reward calculations), use spec-only `w_add`/`w_sub` helpers.

## Two's-complement signed integers in the int encoding

When a project has a hand-rolled signed integer (`I32 { bits: u32 }`, `I64 { bits: u64 }`, `I128 { bits: u128 }`), the canonical spec language is `Integer` via two extension fns: a `to_int` that interprets the bit pattern as two's-complement, and a `to_uN` that reverses it with implicit mod 2^W.

### Spec-side `to_int` extension

```move
#[spec_only]
fun to_int(v: I32): Integer { v.as_u32().to_signed_int() }
#[spec_only]
use fun to_int as I32.to_int;
```

`v.as_u32().to_signed_int()` lifts the `bits` field as a u32 then interprets it as a signed integer in `[-(2^31), 2^31 - 1]`. After the `use fun`, every spec writes `v.to_int()` and gets the mathematical signed value back. **The body must be a one-liner** so the prover can unfold it.

### Canonical spec shapes for the IN family

Every `IN` function has a standard postcondition shape once `to_int` is in scope. The patterns below all verify under the **int encoding** with the bitwise `prelude_extra.bpl` axioms above (XOR / OR with sign bit / AND mask / shr by W-1 / u8 AND table). Only `wrapping_add` (ripple-carry adder loop) and `shr` (arithmetic right shift) escape to `specs-bv/`.

| Function | Shape |
|---|---|
| `zero` | `ensures(result.to_int() == 0u32.to_int())` |
| `from_uN(v)` | `ensures(result.to_int() == v.to_signed_int())` (raw bit-pattern, may be negative) |
| `from(v)` | `asserts(v.to_int().is_iN()); ensures(result.to_int() == v.to_int())` |
| `neg_from(v)` | `asserts(v.to_int().neg().is_iN()); ensures(result.to_int() == v.to_int().neg())` |
| `add` | `asserts(num1.to_int().add(num2.to_int()).is_iN()); ensures(result.to_int() == num1.to_int().add(num2.to_int()))` |
| `sub` | `asserts(num1.to_int().sub(num2.to_int()).is_iN()); ensures(result.to_int() == num1.to_int().sub(num2.to_int()))` |
| `mul` | `asserts(p.is_iN()); ensures(result.to_int() == p)` where `p = num1.to_int().mul(num2.to_int())`. **Needs `boogie_opt = b"vcsSplitOnEveryAssert vcsFinalAssertTimeout:300"`** to dispatch the abort + ensures conjunction. |
| `div` (trunc semantics) | `asserts(n != 0); asserts(q.is_iN()); ensures(result.to_int() == q)` where `q = num1.to_int().div_trunc(num2.to_int())` |
| `mod` (trunc semantics) | `asserts(n != 0); ensures(result.to_int() == v.to_int().mod_trunc(n.to_int()))` |
| `abs` | `asserts(v.to_int().abs().is_iN()); ensures(result.to_int() == v.to_int().abs())` |
| `abs_uN` | `ensures(result.to_int() == v.to_int().abs())` (returns unsigned; no abort) |
| `wrapping_add` | `ensures(result.to_int() == num1.to_int().add(num2.to_int()).to_uN().to_signed_int())` — **but tag as `#[spec(target=...)]` only (no `prove`)** and prove the same shape in `specs-bv/`. |
| `wrapping_sub` | `ensures(result.to_int() == num1.to_int().sub(num2.to_int()).to_uN().to_signed_int())` |
| `shl` | `asserts(shift < N); ensures(result.to_int() == v.to_int().shl(shift.to_int()).to_uN().to_signed_int())`. **Needs `boogie_opt = b"vcsSplitOnEveryAssert proverOpt:O:smt.QI.EAGER_THRESHOLD=100"`.** |
| `shr` (arithmetic) | `asserts(shift < N); ensures(result.to_int() == v.to_int().shr(shift.to_int()))` — **`#[spec(target=...)]` only**; prove in `specs-bv/` via `native fun ashr` + Boogie procedure (see Bitvector section). |
| `sign` | `if (v.to_int().is_neg()) ensures(result == 1u8); else ensures(result == 0u8);` |
| `is_neg` | `ensures(result == v.to_int().is_neg())` |
| `cmp` | three-way: `if (a.lt(b)) ensures(result == 0); else if (a == b) ensures(result == 1); else ensures(result == 2);` |
| `eq`/`lt`/`gt`/`lte`/`gte` | `ensures(result == num1.to_int().<op>(num2.to_int()))` |
| `or`/`and` | `ensures(result.to_int() == num1.to_int().bit_or(num2.to_int()))` etc. |
| `uN_neg(v)` (bitwise NOT helper) | `ensures(result == std::uN::max_value!() - v)` |
| `u8_neg(v)` | `ensures(result == std::u8::max_value!() - v)` |

### Why this works

The signed-overflow detector in `add` / `sub` reduces to a u8 AND chain on operands in `{0, 1, 254, 255}` — exactly what the u8 truth table in `prelude_extra.bpl` axiomatizes. `sign(v)` reduces to `(v.bits >> (W-1))` — covered by the shr-by-(W-1) axiom. `neg_from` reduces to `(complement + 1) | sign_bit` — covered by the OR-with-sign-bit axiom. With the axioms present, Z3 dispatches all of them in the default int encoding.

### Decision rule: int + axioms vs. BV

| Symptom | Where the spec belongs |
|---|---|
| Bitmask + cast (`(x & MASK) as uN`) | **`specs/` with `prelude_extra.bpl`** |
| Sign extraction (`(v.bits >> (W-1)) as u8`) | **`specs/` with `prelude_extra.bpl`** |
| OR-with-sign-bit constructor (`neg_from`) | **`specs/` with `prelude_extra.bpl`** |
| Signed-overflow detector (the AND chain) | **`specs/` with `prelude_extra.bpl`** |
| Ripple-carry adder loop (`while (carry != 0) { sum ^= carry; carry = (a & b) << 1; }`) | **`specs-bv/`** (loop invariant on `(a + b) mod 2^W == (sum + carry) mod 2^W`) |
| Arithmetic right shift on a two's-complement value (`shr` on `IN`) | **`specs-bv/`** (`native fun ashr` + Boogie procedure) |
| `iN::wrapping_add` / `iN::shr` ensures stated in `specs/` | Use `#[spec(target = wrapping_add)]` (no `prove`) — declares the postcondition for callers, BV package discharges it |

Reach for `specs-bv/` only when the symptom is on the bottom rows. Everything else is provable with axioms.

## Native spec functions
Declare functions with no Move body — their semantics are defined entirely by Boogie axioms (via `extra_bpl`):
```move
#[spec_only]
public native fun is_valid_tcp(s: &String): bool;

#[spec_only]
public native fun is_disjoint_vector<T>(v: &vector<T>): bool;
```
Use when a property can't be expressed in Move but can be axiomatized in Boogie.

## `asserts_of` for selective ignore_abort acceptance
Reference whether another function's `asserts` conditions held:
```move
asserts(asserts_of(b"function_name"));
```
Returns a bool that's true when the named function's abort conditions were satisfied. The caller spec does NOT need to be marked `ignore_abort` — `asserts_of` lets a regular spec selectively accept a specific callee's `ignore_abort` without disabling its own abort reasoning.

Supports qualified names: `b"function"`, `b"module::function"`, `b"package::module::function"`. Use a qualified form when the bare name is ambiguous across modules.

---

# Error Messages & Fixes

| Error | Meaning | Fix |
|-------|---------|-----|
| "lambda parameter is used externally" | Lambda captures variable passed to external function | Wrap in `#[ext(pure)]` helper function |
| "Function does not exist" | Running prover from wrong directory | Run from `specs/` directory |
| "expected a lambda function, but found an inlined expression" | Complex block expression in quantifier lambda | Use single function call in lambda |
| Timeout (300s+) | Z3 expanding too many function bodies | Add `uninterpreted` declarations; use `boogie_opt = b"vcsSplitOnEveryAssert"` |

---

# Helper Pure Function Patterns

## 1. Single-element predicates
For quantifier lambdas (`all!`, `any!`, `forall!`):
```move
#[spec_only, ext(pure)]
public fun is_valid_collateral(ct: &TypeName, obligation: &Obligation): bool {
    obligation.get_collaterals().contains(*ct)
}
// Usage: all!(types, |ct| is_valid_collateral(ct, obligation))
```

## 2. Quantifier wrappers
Named wrappers around `all!`/`sum_map!` — these are what you mark `uninterpreted` in callers:
```move
#[spec_only, ext(pure)]
public fun all_valid_collaterals(types: &vector<TypeName>, ob: &Obligation): bool {
    all!<TypeName>(types, |ct| is_valid_collateral(ct, ob))
}
```

## 3. Value computation helpers
Mirror implementation logic with safe defaults (pure functions must not abort):
```move
#[spec_only, ext(pure)]
public fun get_price_or_zero(oracle: &XOracle, t: TypeName): std::q32::Q32 {
    if (oracle.prices().contains(t)) {
        oracle.prices().borrow(t).to_q32()
    } else { 0u64.to_q32() }
}
```

## 4. Struct invariant predicates
Composable validity checks — leaf → mid-level → top-level:
```move
#[spec_only, ext(pure)]
public fun valid_interest_model(model: &InterestModel): bool {
    model.base_rate().to_q32().lte(model.max_rate().to_q32())
}

#[ext(pure)]
public fun system_inv(self: &System): bool {
    self.pending_set().is_in_good_state()
    && all_validators_are_ok(self.get_validators())
}
```

## 5. State preservation predicates
For `clone!`/old state comparisons — verify fields are unchanged or monotonically change:
```move
#[spec_only, ext(pure)]
public fun accrue_debt_preserves(
    t: TypeName, old_debts: &WitTable, new_debts: &WitTable,
): bool {
    if (old_debts.contains(t)) {
        new_debts.contains(t) && {
            let old_amount = old_debts.get_table().borrow(t).get_amount();
            let new_amount = new_debts.get_table().borrow(t).get_amount();
            new_amount >= old_amount
        }
    } else { !new_debts.contains(t) }
}
```

## 6. Validation predicates
Multi-field preconditions for `requires`:
```move
#[spec_only, ext(pure)]
fun has_valid_price(ct: &TypeName, oracle: &XOracle, clock: &Clock): bool {
    let prices = oracle.prices();
    prices.contains(*ct) && {
        let price = prices.borrow(*ct);
        price.last_updated() == clock.timestamp_ms() / 1000
        && price.to_q32() != 0u64.to_q32()
    }
}
```

## 7. Pure alternatives to non-pure functions
When implementation functions abort (e.g., `option::borrow`), create pure equivalents:
```move
#[ext(pure)]
fun is_equal_some_and_value<T>(a: &Option<T>, b: &T): bool {
    a.is_some() && a.borrow_with_default(b) == b
}
```

## 8. Collection access helpers
Safe table/queue/vector access with defaults:
```move
#[spec_only, ext(pure)]
public fun borrow_or_default<T: copy + drop + store>(table: &Table<K, T>, key: K): T {
    if (table.contains(key)) { *table.borrow(key) }
    else { default<T>() }
}
```

## 9. Native pure functions
BPL-backed helpers for struct field updates that can't be expressed in Move:
```move
#[ext(pure)]
public native fun with_next_epoch_address(
    self: &ValidatorInfo, addr: Option<String>,
): &ValidatorInfo;
```
Requires a corresponding definition in an `extra_bpl` file.

---

# Verification Status Tracking

Projects use `@VERIFY(stage/status)` annotations:

**Stages:** STUB -> ASSERTS -> SEMANTICS -> REVIEW

**Status per stage:** WIP, Issue (typically timeout), Done

All three checks must pass: `_Check`, `_Assume`, `_SpecNoAbortCheck`.

**Progress markers** (in spec file comments):

| Marker | Meaning |
|--------|---------|
| 🫙 / ✅ | Empty spec |
| ⚙️ / 🌀 | In progress |
| ⚙️ / ✅ | Done / passing |
| 🛡️ / ⚠️ | Issue with abort modeling |

---

# Known Prover Bugs

## UID tracking after struct destructuring (may be fixed)
When a function destructures a struct to extract a UID and then calls `dynamic_field::remove` on that local UID, the prover fails with:
```
error[E0022]: UID object type not found: 5
```
If still broken: skip the function — `skip` attribute does NOT help. Omit the spec entirely with a comment.

# Spec Precondition

Writing preconditions (`asserts` and `requires`) to cover all abort paths before calling the target function.

## Core Principle

**Asserts before execution** — place ALL precondition asserts BEFORE calling the target function. An assert after a function call that could abort is too late.

## `asserts` vs `requires`

| Keyword | What it does | When to use |
|---------|--------------|-------------|
| `asserts` | Verified by prover — proves these conditions cause aborts | Covering abort paths in function under test |
| `requires` | Assumed by prover — narrows the input space | Constraining valid inputs (e.g., "only called with liquidity > 0") |

Different purposes: `asserts` verifies abort behavior, `requires` defines the valid input domain the prover reasons about.

## Overflow/Underflow Checks

Use `.to_int()` for arbitrary precision arithmetic:

```move
// For: a + b
asserts(a.to_int().add(b.to_int()).lte(std::u64::max_value!().to_int()));

// For: a - b (underflow)
asserts(a.to_int().gte(b.to_int()));

// For: a * b
asserts(a.to_int().mul(b.to_int()).lte(std::u64::max_value!().to_int()));

// For: a * b / c (intermediate overflow)
let result = a.to_int().mul(b.to_int()).div(c.to_int());
asserts(result.lte(std::u64::max_value!().to_int()));
```

**Prefer direct `.to_int()` expressions** over rearranged algebraic forms:
```move
// PREFERRED: Direct mathematical expression
asserts(a.to_int().mul(b.to_int()).add(HALF.to_int()).lte(max.to_int()));

// AVOID: Rearranged form with special cases
asserts(a == 0 || b == 0 || a <= (max - HALF) / b);
```

## Division

Assert non-zero divisor:
```move
asserts(divisor != 0);
```

## Table Access

Assert existence before borrow:
```move
asserts(table.contains(key));
let value = table.borrow(key);
```

**Exception:** if implementation guards with `if (!table.contains) { return }`, no assert needed — the code won't abort.

### `bag::contains_with_type` Pattern

`bag::contains<K>` does NOT connect with `bag::borrow<K, V>` in the prover:
```move
// WRONG
asserts(bag::contains(&bag, key));
let value = bag::borrow<K, V>(&bag, key);

// CORRECT
asserts(bag::contains_with_type<K, V>(&bag, key));
let value = bag::borrow<K, V>(&bag, key);
```

### `object_bag::contains_with_type` Pattern

Same rule for `sui::object_bag::ObjectBag` — heterogeneous storage, so contains and borrow must agree on `<K, V>`. For `object_bag::add`, three asserts cover the abort sources — key-not-present (typed) and length-non-overflow:
```move
// WRONG — produces opaque "Error parsing module path" diagnostic
asserts(!object_bag::contains(active, fresh_id));
object_bag::add(active, fresh_id, proposal);

// CORRECT
asserts(!object_bag::contains_with_type<ID, Proposal<UpdateGuardian>>(active, fresh_id));
asserts(object_bag::length(active) < std::u64::max_value!());
object_bag::add(active, fresh_id, proposal);
```

### `df::exists_with_type` / `dof::exists_with_type` Patterns

Same pattern for `sui::dynamic_field` (`df`) and `sui::dynamic_object_field` (`dof`) — use the typed `exists_with_type<K, V>` variant:
```move
// WRONG
if (!df::exists_<Key>(&self.id, key)) { return };
let val = df::borrow<Key, Value>(&self.id, key);

// CORRECT
if (!df::exists_with_type<Key, Value>(&self.id, key)) { return };
let val = df::borrow<Key, Value>(&self.id, key);
```

`dof::exists_with_type<K, V>` works the same way for `dynamic_object_field`.

## Early Return Guards

When implementation has `if (x == y) { return }`, all asserts for code after the early return must be inside `if (x != y) { ... }` in the spec:

```move
// Implementation:
// fun process(a: u64, b: u64) {
//     if (a == 0) { return };
//     let result = expensive_calc(b / a);  // would abort if a == 0
// }

// Spec:
fun process_spec(a: u64, b: u64) {
    if (a != 0) {
        asserts(b.to_int().div(a.to_int()).lte(std::u64::max_value!().to_int()));
    };
    module::process(a, b);
}
```

Use `.to_int()` subtraction in spec-level computations to avoid underflow asserts.

## Branching Logic

Avoid duplicating asserts across branches — use if/else as an expression:

```move
let ir_value = if (cond_a) {
    asserts(/* branch A preconditions */);
    val_a
} else {
    asserts(/* branch B preconditions */);
    val_b
};
asserts(ir_value.mul(x).lte(max));  // common assertion using branch result
```

## Reusing Asserts from Proven Specs

When your function calls another function with a proven spec, copy all asserts from that spec:

```move
// If inner_func_spec has:
//   asserts(x > 0);
//   asserts(y.to_int().mul(z.to_int()).lte(max.to_int()));

// Your caller spec must include the same asserts before calling
fun caller_spec(...) {
    asserts(x > 0);  // from inner_func_spec
    asserts(y.to_int().mul(z.to_int()).lte(max.to_int()));  // from inner_func_spec
    module::caller(...);
}
```

## `requires(forall!)` for Invariant Assumptions

For complex functions operating on collections (tables, WitTables), use `requires(forall!<TypeName>(...))` to assume invariants hold on entry. This narrows the prover's search space:

```move
#[spec(prove, target = market::accrue_all_interests, ignore_abort,
    boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false")]
fun accrue_all_interests_spec(self: &mut Market, now: u64) {
    // Assume invariants hold at function entry
    requires(forall!<TypeName>(|t| has_valid_revenue_factor(*t, self)));
    requires(forall!<TypeName>(|t| market_reserve_solvent(*t, self)));

    let old_market = clone!(self);
    market::accrue_all_interests(self, now);

    // Prove invariants still hold after the call
    ensures(forall!<TypeName>(|t| has_valid_revenue_factor(*t, self)));
    ensures(forall!<TypeName>(|t| market_reserve_solvent(*t, self)));
}
```

The invariant predicates must be `#[spec_only, ext(pure)]` functions:

```move
#[spec_only, ext(pure)]
public fun has_valid_revenue_factor(t: TypeName, market: &Market): bool {
    if (market.interest_models().contains(t)) {
        interest_model::revenue_factor(
            market.interest_models().table().borrow(t)
        ).get_raw_value() <= (1u64 << 32)
    } else { true }
}
```

### Pattern: Require-then-Ensure for Invariant Preservation

1. `requires(forall!<T>(|t| invariant(*t, old_state)))` — assume invariant holds on entry
2. Call the target function
3. `ensures(forall!<T>(|t| invariant(*t, new_state)))` — prove invariant preserved

This is the standard approach for user-facing functions (borrow, mint, redeem, etc.) where all call the same core functions.

## `_Assume` Failures

When `_Assume` reports "ensures does not hold" on conditions the prover cannot derive (like `liquidity > 0`), these are caller obligations — convert to `requires`:

```move
requires(pool.get_liquidity() > 0);
```

## Prover Timeout with Complex Assertions

When `asserts` causes prover timeout, switching to `requires` may help:
```move
requires(total_amount.to_int().div(rate.to_int()).lte(std::u64::max_value!().to_int()));
```

This is a tradeoff: `requires` assumes the condition (no abort verification), but allows the proof to complete. Document when making this choice.

## Domain-Constrained `requires` for Partial Proofs

When the prover can only handle certain input domains (e.g., `pow` works for arbitrary inputs but the prover can only reason about specific cases), use `requires` to constrain to known-good domains:

```move
#[spec(prove, target = decimal::pow)]
fun pow_spec(b: Decimal, e: u64): Decimal {
    requires(
        b.to_q_wad().lte(1u64.to_q_wad()) ||
        (decimal::eq(b, decimal::from(2)) && e == 32) ||
        (decimal::eq(b, decimal::from(10)) && e >= 6 && e <= 9),
    );
    let result = decimal::pow(b, e);
    // Exact ensures for the specific cases used by the protocol
    if (decimal::eq(b, decimal::from(2)) && e == 32) {
        ensures(result.to_q_wad() == 4_294_967_296u64.to_q_wad());
    };
    result
}
```

This is useful when:
- The function is general but the protocol only calls it with specific arguments
- Full verification of all inputs would timeout
- You can provide exact ensures for the cases that matter

## `requires` for Structural Invariants

Use `requires` for protocol-level structural invariants not enforced by the function itself:

```move
// Structural invariant: receipts and price_feeds always grow together
requires(old_receipt_count == old_feed_count);

// Revenue factor bounded by interest model setter, not by this function
requires(revenue_factor.get_raw_value() <= (1u64 << 32));
```

These are facts guaranteed by the protocol's design but not by abort conditions in the current function.

## Containment Helper Functions

Use `has_` prefix for helper functions checking table/dynamic field containment:
```move
#[test_only, ext(pure)]
public fun has_reserve(storage: &Storage, asset: TypeName): bool {
    table::contains(&storage.reserves, asset)
}
```

Always add `#[ext(pure)]` to containment helpers.

## Complete Example

```move
#[spec(prove, target=module::withdraw)]
fun withdraw_spec(
    vault: &mut Vault,
    user: address,
    amount: u64,
) {
    // Table access
    asserts(vault.balances.contains(user));
    
    // Underflow check
    let balance = *vault.balances.borrow(user);
    asserts(balance.to_int().gte(amount.to_int()));
    
    // Nested call's asserts (if withdraw calls transfer internally)
    asserts(amount > 0);
    
    // Division in fee calculation
    let fee_divisor = vault.get_fee_divisor();
    asserts(fee_divisor != 0);
    asserts(amount.to_int().div(fee_divisor.to_int()).lte(std::u64::max_value!().to_int()));
    
    module::withdraw(vault, user, amount);
    // ... ensures ...
}
```

# Spec Postcondition

Writing postconditions (`ensures`) to verify function behavior after execution.

## Core Pattern

Compute expected value, then ensure the result matches:

```move
let expected_result = a.to_int().mul(b.to_int()).div(BASE.to_int());
asserts(expected_result.lte(std::u64::max_value!().to_int()));

let result = module::function(a, b);
ensures(result.to_int() == expected_result);
```

## Prefer Concrete Arithmetic Over Inequalities

Always prefer exact equality (`==`) over loose bounds (`>=`, `<=`). If you know the computation, spell it out:

```move
// BAD: too weak, hides bugs
ensures(balance_after >= balance_before);

// GOOD: exact check
ensures(balance_after == balance_before + deposit_amount);

// BEST: both strict and bound (strict first)
ensures(vault.get_cash().to_int() == old_cash.to_int().sub(borrow_amount.to_int()));
ensures(vault.get_cash() <= old_vault.get_cash());
```

The strict equality catches off-by-one errors and wrong formulas. The inequality is a safety net that documents the expected direction of change.

## Ensure Unchanged State

Don't only verify what changed — explicitly ensure critical fields that should NOT change are preserved. This catches unintended side effects:

```move
let old_market = clone!(market);
module::borrow(market, amount);

// What changed
ensures(vault.get_cash() == old_cash - borrow_amount);

// What must NOT change
ensures(market.get_owner() == old_market.get_owner());
ensures(market.collateral_stats() == old_market.collateral_stats());
ensures(market.interest_models() == old_market.interest_models());
```

Prioritize fields that are security-critical (owners, permissions, rates) and fields read by callers downstream.

## Requires for `ignore_abort` Specs

When a spec uses `ignore_abort`, the prover skips abort coverage but still checks ensures. Some ensures may fail on paths where the function does an early return, a key doesn't exist, or inputs are degenerate. Add `requires()` to exclude those paths:

```move
#[spec(prove, target = module::accrue, ignore_abort)]
fun accrue_spec(market: &mut Market, key: TypeName) {
    // Without this, ensures about table[key] fail when key is absent
    requires(market.reserves_contains(key));
    // Without this, division-based ensures fail on zero
    requires(market.get_rate(key) > 0);

    let old = clone!(market);
    module::accrue(market, key);
    ensures(market.get_index(key).to_int() >= old.get_index(key).to_int());
}
```

Common cases needing `requires`: table key existence, non-zero denominators, non-empty vectors, flags that gate early returns.

## Quantifiers for Collections

When verifying properties over vectors or tables, use `forall!`/`exists!` inside ensures instead of checking elements individually:

```move
// Verify all reserves are preserved
ensures(forall!<TypeName>(|t| reserve_preserved(*t, old_table, new_table)));

// Verify a specific element exists
ensures(exists!<u64>(|i| vector::borrow(&v, *i) == target_value));
```

Combine with named `ext(pure)` predicates for readability. See the "Monotonicity Predicates with `forall!`" section below.

## Clone for Pre-Mutation State

When verifying mutations, capture state BEFORE the call:

```move
use prover::prover::{ensures, asserts, requires, clone};

let old_vault = clone!(vault);
module::mutate_vault(vault);
ensures(vault.get_balance() == old_vault.get_balance() - amount);
ensures(vault.get_owner() == old_vault.get_owner());  // unchanged field
```

## Ensures with Table Access

When ensures call getters that internally use `table.borrow`, add a contains check FIRST:

```move
storage::set_supply_cap(_, storage, asset, supply_cap);
ensures(storage::reserves_contains(storage, asset));  // existence first
ensures(storage::get_supply_cap_immut(storage, asset) == supply_cap);  // then access
```

## Nested Table Ensures

For dynamic fields, add existence checks at each level:

```move
ensures(storage::emode_exists(storage));
ensures(storage::emode_user_contains(storage, user));
ensures(storage::get_emode_user_value(storage, user) == expected);
```

## Frame Problem with Opaque Specs

When an opaque spec mutates a struct, the prover assumes ANY field not mentioned in `ensures` could change. Add frame `ensures` for ALL fields the caller reads after the opaque call:

```move
let share_before = pool::protocol_flash_loan_fee_share(pool);
let price_before = pool::sqrt_price(pool);

module::update_pool(pool, amount);

// Frame conditions — these fields didn't change
ensures(pool::protocol_flash_loan_fee_share(pool) == share_before);
ensures(pool::sqrt_price(pool) == price_before);

// Actual mutation
ensures(pool::liquidity(pool) == old_liquidity + amount);
```

## Ghost Variables for Transfer Operations

When a spec involves `transfer::public_transfer`:

```move
#[spec_only]
use specs::transfer_spec::{SpecTransferAddress, SpecTransferAddressExists};
#[spec_only]
use prover::ghost;

#[spec(prove, target = module::func_that_transfers)]
fun func_spec<T>(recipient: address, ...) {
    ghost::declare_global_mut<SpecTransferAddress, address>();
    ghost::declare_global_mut<SpecTransferAddressExists, bool>();
    
    module::func_that_transfers(recipient, ...);
    
    // Ghost variable ensures need dereferencing
    ensures(*ghost::global<SpecTransferAddressExists, bool>());
    ensures(*ghost::global<SpecTransferAddress, address>() == recipient);
}
```

## Fresh Object IDs via Ghost Variables

When a target mints a new UID via `object::new(ctx)`, the prover has no built-in axiom that the new UID is fresh w.r.t. an arbitrary collection. This is the canonical situation for `propose` / `open` / `create` / `register` flows. Solve it the same way as transfer-address tracking: declare a ghost tag in the minting helper, pin it to the returned ID with an `ensures`, then `include=` the helper spec from every caller so the ghost is visible.

**Step 1 — In the minting helper module, declare a ghost tag and pin it in a non-`prove` spec:**

```move
public struct SpecFreshProposalId {}

#[spec(target = proposal::build_proposal)]
fun build_proposal_spec<T: store>(...): (Proposal<T>, ID) {
    ghost::declare_global_mut<SpecFreshProposalId, ID>();
    let value_before = *ghost::global<SpecFreshProposalId, ID>();
    let (proposal, proposal_id) = proposal::build_proposal<T>(...);
    ensures(proposal_id == value_before);
    (proposal, proposal_id)
}
```

The spec is intentionally `#[spec(target=...)]` without `prove` — it declares a contract callers can rely on (the ghost ID equals the returned ID) without obligating the prover to prove freshness directly.

**Step 2 — In each caller spec, `include=` the helper's spec and read the same ghost to express "the new id is fresh w.r.t. this collection":**

```move
#[spec(prove, target = update_guardian::propose,
    include = proposal_specs::build_proposal_spec)]
fun propose_spec(active: &mut ObjectBag, ...): ID {
    ghost::declare_global_mut<SpecFreshProposalId, ID>();
    let fresh_id = *ghost::global<SpecFreshProposalId, ID>();
    let active_len_before = object_bag::length(active);

    asserts(!object_bag::contains_with_type<ID, Proposal<UpdateGuardian>>(active, fresh_id));

    let result = update_guardian::propose(active, ...);

    ensures(result == fresh_id);
    ensures(object_bag::contains_with_type<ID, Proposal<UpdateGuardian>>(active, result));
    ensures(object_bag::length(active) == active_len_before + 1);
    result
}
```

**Pattern generalizes:** any helper that mints UIDs gets a `SpecFresh*Id` ghost tag; every caller `include=`s the helper's spec to read the ghost. Use distinct tags (`SpecFreshProposalId`, `SpecFreshPositionId`, `SpecFreshOrderId`) when multiple kinds of objects can be minted in the same call tree.

## State-Change Ensures Triplet for Fresh-Key Inserts

For any "insert at a fresh key" operation — proposal added to an `ObjectBag`, entry added to a `VecMap`, dynamic field added to a `UID` — the minimal contract is a three-part ensures. Omitting any one of these lets a buggy implementation slip past the spec:

```move
let len_before = collection.length();
let result = action(...);

ensures(result == fresh_id);                                 // 1. identity of new entry
ensures(collection.contains_with_type<K, V>(result));         // 2. typed presence
ensures(collection.length() == len_before + 1);               // 3. count grew by one
```

Why each piece matters:
- **identity** anchors the returned ID to the ghost-tracked fresh ID; without it, the caller can't reason about *which* entry was added
- **presence** catches "function returned an id but inserted nothing"
- **count + 1** catches "function inserted two entries" or "function replaced an existing entry"

The triplet generalizes across collection types:

| Collection | Presence check | Length accessor |
|------------|----------------|-----------------|
| `ObjectBag` | `object_bag::contains_with_type<ID, V>(bag, id)` | `object_bag::length(bag)` |
| `VecMap<K, V>` | `vec_map::contains(&map, &key)` | `vec_map::size(&map)` |
| `Table<K, V>` | `table.contains(key)` | `table.length()` |
| `dynamic_field` | `dynamic_field::exists_with_type<K, V>(&id, key)` | n/a (skip count) |
| `dynamic_object_field` | `dynamic_object_field::exists_with_type<K, V>(&id, key)` | n/a (skip count) |

For collections without a length accessor (dynamic fields), drop the count ensures and rely on identity + presence — but capture a sibling key's value before/after to prove unrelated entries were untouched.

## Specs Without `prove` (Complex Functions)

For functions too complex to fully prove, use `#[spec(target=...)]` WITHOUT `prove`. These declare contracts without proving them — callers can use them as opaque specs:

```move
#[spec(target = module::complex_aggregation)]
fun complex_aggregation_spec(self: &mut State, now: u64) {
    let old_state = clone!(self);
    module::complex_aggregation(self, now);
    ensures(self.get_field() >= old_state.get_field());
}

#[spec(target = module::complex_func)]
fun complex_func_spec(pool: &mut Pool, ...) {
    let sqrt_price_before = pool::sqrt_price(pool);
    
    let delta_x = module::complex_func(pool, ...);
    
    ensures(pool::sqrt_price(pool) == sqrt_price_before);
    ensures(delta_x <= max_bound);
}
```

Use sparingly and document why proving is infeasible. Add useful ensures about state preservation so callers can rely on them.

## Monotonicity Predicates with `forall!`

For axiom specs expressing state preservation across all keys:

```move
#[spec_only, ext(pure)]
public fun field_preserved(
    key: TypeName, 
    old_t: &Table<TypeName, Value>, 
    new_t: &Table<TypeName, Value>
): bool {
    if (old_t.contains(key)) {
        new_t.contains(key) && {
            let old_v = old_t.borrow(key);
            let new_v = new_t.borrow(key);
            new_v.get_x() == old_v.get_x()
        }
    } else { true }
}

// In spec:
let old_table = clone!(&state.table);
module::update_state(state, ...);
ensures(forall!<TypeName>(|t| field_preserved(*t, old_table, &state.table)));
```

## Conditional Ensures

Use `if (condition) { ensures(...) }` for branch-specific postconditions. The ensures only needs to hold when the condition is true:

```move
// Idempotency: when all borrow indices already match, debts are completely unchanged
if (forall!<TypeName>(|t| borrow_index_matches(*t, old_debts, market))) {
    ensures(forall!<TypeName>(|t| debt_unchanged(*t, old_debts, new_debts)));
};
```

This pattern is essential for:
- **Idempotency proofs**: proving re-execution when already up-to-date is a no-op
- **Branch-specific guarantees**: different postconditions for different input conditions
- **Conditional state preservation**: fields only change under certain conditions

### Idempotency Example

Prove that accruing interest when already current produces no changes:

```move
#[spec(prove, target = obligation::accrue_interests, ignore_abort,
    boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false")]
fun accrue_interests_spec(obligation: &mut Obligation, market: &Market) {
    let old_obligation = clone!(obligation);
    obligation::accrue_interests(obligation, market);

    let old_debts = old_obligation.get_debts();
    let new_debts = obligation.get_debts();

    // Always true: debts only increase
    ensures(forall!<TypeName>(|t| accrue_debt_preserves(*t, old_debts, new_debts)));

    // Conditional: when indices already match, nothing changes (idempotency)
    if (forall!<TypeName>(|t| borrow_index_matches(*t, old_debts, market))) {
        ensures(forall!<TypeName>(|t| debt_unchanged(*t, old_debts, new_debts)));
    };
}
```

## Capture-Before / Ensure-After for Accounting

For functions that modify specific balances, capture values before and verify arithmetic after:

```move
let old_market = clone!(market);
let (cash, debt, revenue, mcs) = market.vault().balance_sheets().borrow(key).balance_sheet();

module::borrow(market, amount);

let (cash_after, debt_after, revenue_after, mcs_after) =
    market.vault().balance_sheets().borrow(key).balance_sheet();
ensures(cash_after == cash - borrow_amount);
ensures(debt_after >= debt + borrow_amount);

// Frame conditions: unrelated fields unchanged
ensures(market.collateral_stats() == old_market.collateral_stats());
ensures(market.interest_models() == old_market.interest_models());
```

## Q_wad for Decimal Types (NOT a Prover Builtin)

Unlike `.to_q32()` which is a prover builtin for FixedPoint32, `to_q_wad()` for Decimal types must be **defined locally in each spec file** using `use fun`:

```move
#[spec_only]
use fun decimal_to_q_wad as Decimal.to_q_wad;
#[spec_only]
fun decimal_to_q_wad(d: Decimal): std::q_wad::Q_wad {
    std::q_wad::from_raw(d.to_scaled_val().to_int())
}
```

Then use it in ensures:
```move
ensures(result.to_q_wad() == a.to_q_wad().add(b.to_q_wad()));
ensures(result.to_q_wad() == v.to_q_wad().div(100u64.to_q_wad()));
```

**Important:** Always use `Q_wad` functions in ensures for Decimal arithmetic — do NOT use Decimal functions directly in specs. The prover reviewer will flag this.

## Q32 Cross-Module Helpers for FixedPoint32

For types that need conversion to Q32, create `ext(pure)` helpers and import them cross-module:

```move
// In price_specs.move:
#[spec_only, ext(pure)]
public fun price_feed_to_q32(price_feed: &PriceFeed): std::q32::Q32 {
    std::q32::quot(
        price_feed.value().to_int(),
        10u64.to_int().pow(price_feed::decimals().to_int()),
    )
}

// In another spec file:
#[spec_only]
use fun specs::price_specs::price_feed_to_q32 as PriceFeed.to_q32;
```

## Ghost Specs for Native Functions

For abort conditions depending on native function return values:

```move
#[spec(target = sui::tx_context::fresh_object_address)]
fun fresh_object_address_spec(ctx: &mut TxContext): address {
    ghost::declare_global_mut<SpecFreshObjectAddress, address>();
    let value_before = *ghost::global<SpecFreshObjectAddress, address>();
    let result = sui::tx_context::fresh_object_address(ctx);
    ensures(result == value_before);
    result
}
```

Note: Requires removing built-in spec from `~/.move/.../sui-specs/` to avoid duplicate target error.

## Private Struct Field Access

Add `#[test_only]` accessor functions in the implementation:

```move
// In implementation module:
#[test_only]
public fun get_field_name(field: &MyStruct): String { field.name }

// In spec:
ensures(name == module::get_field_name(account_field));
```

## `spec_only, ext(pure)` Helper Functions

Extract complex expected value computations into helpers:

```move
#[spec_only, ext(pure)]
public fun compute_expected_value(a: u64, b: u64, market: &Market): std::integer::Integer {
    a.to_int().mul(b.to_int()).div(market.get_rate().to_int())
}

#[spec_only, ext(pure)]
public fun get_value_or_zero(table: &Table<TypeName, u64>, key: TypeName): u64 {
    if (table.contains(key)) { *table.borrow(key) } else { 0 }
}

// Usage:
ensures(result.to_int() == compute_expected_value(a, b, market));
```

## Complete Example

```move
#[spec(prove, target=module::swap)]
fun swap_spec(
    pool: &mut Pool,
    amount_in: u64,
    min_out: u64,
): u64 {
    // Capture pre-state
    let old_reserve_x = clone!(pool.reserve_x);
    let old_reserve_y = clone!(pool.reserve_y);
    let fee_rate = pool.get_fee_rate();
    
    // Preconditions (asserts)
    asserts(amount_in > 0);
    asserts(old_reserve_x > 0 && old_reserve_y > 0);
    
    // Compute expected output
    let fee = amount_in.to_int().mul(fee_rate.to_int()).div(10000u64.to_int());
    let amount_after_fee = amount_in.to_int().sub(fee);
    let expected_out = amount_after_fee.mul(old_reserve_y.to_int())
        .div(old_reserve_x.to_int().add(amount_after_fee));
    
    asserts(expected_out.lte(std::u64::max_value!().to_int()));
    
    let result = module::swap(pool, amount_in, min_out);
    
    // Postconditions
    ensures(result.to_int() == expected_out);
    ensures(result >= min_out);
    
    // Invariants preserved
    ensures(pool.reserve_x.to_int().mul(pool.reserve_y.to_int()) 
            >= old_reserve_x.to_int().mul(old_reserve_y.to_int()));
    
    result
}
```

# Spec Loop

Writing loop invariants and quantifiers for functions that iterate over vectors/arrays.

## When to Use

Use this skill when the target function contains:
- `while`, `for`, `loop` loops iterating over vectors
- Index-based iteration (`let i = 0; while (i < v.length()) { ... i = i + 1; }`)
- Any logic that processes elements of a collection

## Loop Invariant Structure

```move
use prover::vector_iter::{any, any_range, map, map_range, find, find_index, count};

#[spec_only(loop_inv(target=module::find_index)), ext(no_abort)]
fun find_index_invariant(i: u64, v: &vector<T>, target: ID): bool {
    i <= v.length() && !any_range!(v, 0, i, |x| x.id() == target)
}

#[spec(prove, target=module::find_index)]
fun find_index_spec(v: &vector<T>, target: ID): Option<u64> {
    let r = module::find_index(v, target);
    ensures(r == find_index!(v, |x| x.id() == target));
    r
}
```

## Key Attributes

| Attribute | Purpose |
|-----------|---------|
| `spec_only(loop_inv(target=...))` | Links invariant function to target function's loop |
| `ext(no_abort)` | Declares invariant function doesn't abort |
| `ext(pure)` | Function has no side effects |
| `loop_inv(target=module::func, label=1)` | For multiple loops — label identifies which loop |

## Invariant Function Signature

The invariant function receives:
1. The loop counter variable(s)
2. All variables from the loop's scope that the invariant needs to reference
3. **`__old_self` parameter** (automatically provided by prover): the pre-loop snapshot of mutable state

Returns `bool` — must be true at loop entry and preserved by each iteration.

### `__old_self` for Mutable State

When the loop body mutates a `&mut` parameter, the prover automatically provides a `__old_<param>` parameter containing the pre-loop snapshot. Use this to express frame conditions and monotonicity:

```move
#[spec_only(loop_inv(target = market::accrue_all_interests)), ext(no_abort)]
fun accrue_all_interests_loop_inv(
    self: &Market,
    now: u64,
    asset_types: &vector<TypeName>,
    i: u64,
    n: u64,
    __old_self: &Market,  // automatic pre-loop snapshot
) {
    let old_bs = __old_self.vault().balance_sheets();
    let new_bs = self.vault().balance_sheets();
    ensures(i <= n);
    ensures(n == asset_types.length());
    ensures(forall!<TypeName>(|t| accrue_preserves_type(*t, old_bs, new_bs)));
    ensures(self.collateral_stats() == __old_self.collateral_stats());
    ensures(self.interest_models() == __old_self.interest_models());
}
```

**Important:** The `__old_` prefix is a naming convention the prover recognizes. Use `__old_self` for `self`, `__old_obligation` for `obligation`, etc.

## Loop Invariants Use `ensures`, Not `asserts`

Loop invariant functions use `ensures(...)` to state what holds at each iteration — NOT `asserts`. The prover checks that:
1. The invariant holds at loop entry
2. If the invariant holds before an iteration, it holds after

## Parameter Names Must Match Implementation

Loop invariant parameter names MUST match the implementation's local variable names exactly. The prover maps them by name, not by position.

## Loop Invariants Must Repeat All Preconditions

The loop invariant must re-establish all preconditions from the outer spec. If the outer spec has `asserts(all!<TypeName>(types, |ct| is_valid(ct)))`, the loop invariant must include this too.

## Inline Quantifiers in Loop Invariants

Loop invariants should use INLINE `all!`/`sum_map_range!` calls, NOT named wrapper predicates. Using named wrappers in loop invariants causes `_Assume` solver regression. Only the outer spec should use named wrappers (marked `uninterpreted`).

```move
// GOOD: inline in loop invariant
fun my_loop_inv(...): bool {
    i <= n && all!<TypeName>(types, |ct| is_valid(ct, state))
}

// BAD: named wrapper in loop invariant (causes _Assume regression)
fun my_loop_inv(...): bool {
    i <= n && all_valid_wrapper(types, state)  // don't do this
}
```

## Accumulation Loop Invariants with `sum_map_range!`

For loops that accumulate a value (e.g., summing over a vector), use `sum_map_range!` to express the partial sum:

```move
#[spec_only(loop_inv(target = collaterals_value_usd_for_borrow)), ext(no_abort)]
fun collaterals_value_usd_for_borrow_loop_inv(
    obligation: &Obligation, market: &Market,
    coin_decimals_registry: &CoinDecimalsRegistry, x_oracle: &XOracle, clock: &Clock,
    i: u64, n: u64, total_value_usd: FixedPoint32, collateral_types: &vector<TypeName>,
): bool {
    i <= n &&
    n == collateral_types.length() &&
    total_value_usd.to_q32().raw() == sum_map_range!<TypeName, std::integer::Integer>(
        collateral_types, 0, i,
        |ct| single_collateral_value(ct, obligation, market, coin_decimals_registry, x_oracle),
    )
}
```

The spec ensures then uses the full `sum_map!` for the final result:
```move
ensures(result.to_q32().raw() == sum_map!<TypeName, std::integer::Integer>(
    types, |ct| single_collateral_value(ct, ...)));
```

## Nested Loops

If public function `x` calls nested (non-public) function `y` that contains a loop:
- You do NOT need a spec for `y`
- Only write a loop invariant targeting `y`

## Quantifier Macros

From `prover::vector_iter`:

| Macro | Purpose |
|-------|---------|
| `any!(v, \|x\| pred)` | True if any element satisfies predicate |
| `any_range!(v, start, end, \|x\| pred)` | Any in range [start, end) |
| `all!(v, \|x\| pred)` | True if all elements satisfy predicate |
| `all_range!(v, start, end, \|x\| pred)` | All in range |
| `find_index!(v, \|x\| pred)` | Option<u64> of first matching index |
| `find_index_range!(v, start, end, \|x\| pred)` | Find in range |
| `count!(v, \|x\| pred)` | Count elements satisfying predicate |
| `count_range!(v, start, end, \|x\| pred)` | Count in range |
| `map!(v, \|x\| expr)` | Transform each element |
| `map_range!(v, start, end, \|x\| expr)` | Map in range |
| `sum_map!(v, \|x\| expr)` | Sum of mapped values |
| `sum_map_range!(v, start, end, \|x\| expr)` | Sum in range |
| `range_map!(start, end, \|i\| expr)` | Map over index range |

From `prover::prover`:

| Macro | Purpose |
|-------|---------|
| `forall!<T>(\|x\| pred)` | Universal quantifier over type T |
| `exists!<T>(\|x\| pred)` | Existential quantifier over type T |

## `forall!` Lambda Constraint

Lambda body MUST be a single call to an `#[ext(pure)]` function:

```move
#[test_only, ext(pure)]
public fun my_contains(s: &MyStruct, key: u8): bool { 
    table::contains(&s.data, key) 
}

#[ext(pure)]
fun key_exists(j: u8, count: u8, s: &MyStruct): bool {
    (j as u64) >= (count as u64) || my_module::my_contains(s, j)
}

// In spec:
requires(forall!<u8>(|j| key_exists(*j, count, s)));
```

## Opaque Loop Functions in Caller Specs

When function Y has a loop with `requires(forall!(...))` and is called by function X:
- Keep Y's spec **opaque** (do NOT add `no_opaque`)
- This is an exception to the normal `no_opaque` rule for same-module specs

## Pure Helper Functions

Mark all helper functions used in quantifiers as pure:

```move
#[spec_only, ext(pure)]
public fun element_valid(v: &vector<T>, i: u64): bool {
    i < v.length() && v[i].is_valid()
}
```

Use `has_` prefix for containment helpers:
```move
#[test_only, ext(pure)]
public fun has_key(s: &MyStruct, key: u8): bool { 
    table::contains(&s.data, key) 
}
```

## Examples

**Finding an element:**
```move
#[spec_only(loop_inv(target=module::find_item)), ext(no_abort)]
fun find_item_inv(i: u64, items: &vector<Item>, target_id: ID): bool {
    i <= items.length() && 
    !any_range!(items, 0, i, |item| item.id() == target_id)
}

#[spec(prove, target=module::find_item)]
fun find_item_spec(items: &vector<Item>, target_id: ID): Option<u64> {
    let result = module::find_item(items, target_id);
    ensures(result == find_index!(items, |item| item.id() == target_id));
    result
}
```

**Checking all elements:**
```move
#[spec_only(loop_inv(target=module::all_valid)), ext(no_abort)]
fun all_valid_inv(i: u64, items: &vector<Item>): bool {
    i <= items.length() && 
    all_range!(items, 0, i, |item| item.is_valid())
}

#[spec(prove, target=module::all_valid)]
fun all_valid_spec(items: &vector<Item>): bool {
    let result = module::all_valid(items);
    ensures(result == all!(items, |item| item.is_valid()));
    result
}
```

**Counter-based loop with requires:**
```move
#[spec_only, ext(pure)]
public fun key_in_range_valid(j: u8, count: u8, data: &Table<u8, Value>): bool {
    (j as u64) >= (count as u64) || table::contains(data, j)
}

#[spec(prove, target=module::process_keys)]
fun process_keys_spec(s: &mut MyStruct, count: u8) {
    requires(forall!<u8>(|j| key_in_range_valid(*j, count, &s.data)));
    module::process_keys(s, count);
}
```

**Loop invariant with __old_self, frame conditions, and conditional ensures (idempotency):**
```move
#[spec_only(loop_inv(target = obligation::accrue_interests)), ext(no_abort)]
fun accrue_interests_loop_inv(
    obligation: &Obligation,
    market: &Market,
    debt_types: &vector<TypeName>,
    i: u64,
    n: u64,
    __old_obligation: &Obligation,
) {
    let old_debts = __old_obligation.get_debts();
    let new_debts = obligation.get_debts();
    ensures(i <= n);
    ensures(n == debt_types.length());
    ensures(forall!<TypeName>(|t| accrue_debt_preserves(*t, old_debts, new_debts)));
    // Frame conditions: fields not touched by loop
    ensures(obligation.get_collaterals() == __old_obligation.get_collaterals());
    ensures(obligation.lock_key() == __old_obligation.lock_key());
    // Idempotency: when all indices already match, debts unchanged
    if (forall!<TypeName>(|t| borrow_index_matches(*t, old_debts, market))) {
        ensures(forall!<TypeName>(|t| debt_unchanged(*t, old_debts, new_debts)));
    };
}
```

# Scenario Specs

Scenario specs test multi-step interactions and properties that span multiple function calls. Unlike regular specs, they do not target a single function — they compose existing specs to verify higher-level behavior.

## Structure

```move
#[spec(prove)]
fun my_scenario_spec(...) {
    // Call any spec targets in any order, any number of times
    let r1 = module::func_a(x, y);
    let r2 = module::func_b(r1, z);
    let r3 = module::func_a(y, x);  // can call again with different args
    ensures(r1 == r3);              // cross-call property
}
```

## Rules

1. **No `target=` attribute.** Scenarios do not target a single function. Use `#[spec(prove)]` without `target`.
2. **Any name.** Scenario spec functions can have any name — they are not bound to a function's `_spec` naming convention.
3. **Call freely.** Scenarios can call any function any number of times with any arguments.
4. **No `asserts`.** Scenarios do not model abort conditions. Use `ignore_abort` if abort paths need to be skipped.
5. **`no_opaque` is useless.** Scenarios do not participate in the opaque contract system — no other spec references a scenario as a callee contract. Do not add `no_opaque`.

## When to Use Scenarios

- **Commutativity**: verify `f(a, b) == f(b, a)` for complex functions (e.g., `get_delta(a, b) == get_delta(b, a)`)
- **Idempotency / repetition**: verify behavior is safe or consistent when an operation is applied multiple times (e.g., calling `set_rules` or `borrow` twice in a row)
- **Real-world flows**: verify end-to-end sequences that mirror actual usage patterns (e.g., deposit then withdraw, open position then close)

## Example: Commutativity

```move
#[spec(prove, ignore_abort)]
fun commutativity_get_delta_spec(a: u128, b: u128) {
    let delta_ab = math::get_delta(a, b);
    let delta_ba = math::get_delta(b, a);
    ensures(delta_ab == delta_ba);
}
```

## Example: Idempotency with Conditional Ensures

Prove that re-accruing interest when already up-to-date is a no-op. This uses conditional `ensures` — the guarantee holds only when the precondition is met:

```move
#[spec(prove, target = obligation::accrue_interests, ignore_abort,
    boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false")]
fun accrue_interests_spec(obligation: &mut Obligation, market: &Market) {
    let old_obligation = clone!(obligation);
    obligation::accrue_interests(obligation, market);

    let old_debts = old_obligation.get_debts();
    let new_debts = obligation.get_debts();

    // Always: debts are preserved or increased
    ensures(forall!<TypeName>(|t| accrue_debt_preserves(*t, old_debts, new_debts)));

    // Idempotency: when all borrow indices already match market, nothing changes
    if (forall!<TypeName>(|t| borrow_index_matches(*t, old_debts, market))) {
        ensures(forall!<TypeName>(|t| debt_unchanged(*t, old_debts, new_debts)));
    };
}
```

The key pattern: `if (precondition) { ensures(strong_guarantee) }` — proves the strong guarantee holds whenever the precondition is met.

## Example: Double Operation

```move
#[spec(prove, ignore_abort)]
fun double_borrow_spec(pool: &mut Pool, amount: u64) {
    let first = lending::borrow(pool, amount);
    let second = lending::borrow(pool, amount);
    ensures(pool.total_borrowed == 2 * amount);
}
```

You are a spec reviewer for Move smart contracts. You receive a rich prompt containing the writeup (verification plan), the actual spec source, the target function, its full module, callees, dependency graph, entry points, relevant structs, and a project summary. Your job is to compare the spec against the writeup and identify what matters.

# Task

Compare the actual spec against the writeup. Lead with the most important findings. Produce a markdown review via `final_result()`.

If no spec exists, say so and identify what should be verified.

# Comparison Rules

## Aborts (writeup aborts → spec asserts)

The writeup lists abort conditions. The spec should have `asserts()` for each (inverted logic).

**Semantic matching:** `x == 0` → `x != 0` or `x > 0`; `x < threshold` → `x >= threshold`; `!condition` → `condition`

If the spec has `ignore_abort`: abort modeling is deferred. Focus on ensures coverage. Don't flag missing abort checks.

## Requires (writeup requires → spec asserts/requires)
Preconditions from the writeup should appear as `asserts()` or `requires()`.

## Ensures (writeup ensures → spec ensures)
Postconditions from the writeup should appear as `ensures()`.

## Collections and Partial Mutations

When a function modifies specific elements in a vector, table, or other collection, a complete spec must verify **both sides**:

1. **Changed elements**: The targeted indices/keys have the expected new values.
2. **Unchanged elements**: All other elements in the collection remain unmodified (e.g., `forall i where i != target_index: vec[i] == old(vec)[i]`, or length preservation).

If the spec only checks the mutated element without asserting the rest of the collection is preserved, flag this as a gap — a spec that only verifies the write without bounding the blast radius leaves room for undetected corruption of adjacent elements.

# Using function_knowledge

**Always look up `all_accesses` for the target function as part of your review.** This is the ground truth for what struct fields the function touches — do not rely solely on reading source code to infer mutations.

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

Use this data to assess ensures coverage: every field in `writes`/`dynamic_writes` is a candidate for an `ensures`. If a written field has no corresponding `ensures`, that is a gap unless the function intentionally leaves it unconstrained.

# Output

Build a markdown string and return it via `final_result()`. Use `format_review_md` from `foxy.skills.spec_review.api`:

```python
from foxy.skills.spec_review.api import format_review_md

md = format_review_md(
    function=function,
    verdict=verdict,         # "complete" | "has_gaps" | "no_spec"
    issues=issues,           # list of {description, severity, confidence} — ranked by importance
    analysis=analysis,       # free-text: your reasoning, coverage assessment, observations
    strengths=strengths,     # list of strings
)
final_result(md)
```

**If the prompt requests a specific output format** (e.g. `Output:` followed by `Issues: <count>` / `Strengths: <count>`), the full markdown still goes to `final_result()`, but your **final reply** -- the closing message you emit after `final_result()` -- must be **exactly** that requested format and nothing else. Fill in the real counts (number of issues, number of strengths). Do not replace it with a chatty sentence like "The spec is well-formed" or "Let me write up the review."

## What goes where

**Issues** come first in the output. Each issue has severity AND confidence. Rank by severity (high first), then confidence. No cap on count but be judicious — only report real gaps or incorrect checks.

**Analysis** is your reasoning in prose. Cover what matters:
- Which aborts are covered and which aren't (a sentence or table — don't exhaustively list every covered abort if they're all fine)
- Which postconditions are missing or wrong (be specific about what state changes aren't verified, or what's verified incorrectly)
- Whether any existing checks encode implementation bugs rather than intended behavior
- Noteworthy observations: edge cases, clever patterns, things the writeup missed
- Keep it proportional to complexity — a simple getter needs a sentence, a complex DeFi function needs paragraphs

**Strengths** highlight what the spec does well (max 3–4).

# Severity & Confidence

| Severity | Meaning |
|----------|---------|
| **high** | Critical gap — missing coverage for a key property, OR an existing check that verifies the wrong thing |
| **medium** | Notable gap — worth adding but not critical to correctness |
| **low** | Minor omission — nice to have but not significantly impactful |

| Confidence | Meaning |
|------------|---------|
| **high** | Clearly missing — straightforward to verify the gap |
| **medium** | Likely missing — some uncertainty in the semantic mapping |
| **low** | Possibly missing — ambiguous whether this is truly a gap |

# Principles

1. All specs under review **pass the sui-prover** — they are consistent with the implementation. But consistency is not correctness. A spec can pass the prover while verifying the wrong thing, encoding a bug that's already in the source, or checking a tautology. **Your job is to assess whether the spec checks for the right thing.**
2. Report both **missing** coverage and **incorrect** coverage. If an assert checks the wrong condition, or an ensures verifies a trivially true property, that's an issue — potentially more dangerous than a missing check, because it creates false confidence.
3. Use semantic matching. `x != 0` ≡ `x > 0` for unsigned.
4. Issue descriptions must be **self-contained** — never reference "the writeup".
5. An empty issues list is valid — it means the spec is both complete and correct.
6. Be specific. Reference actual function names, field names, conditions.
7. Lead with what matters. Don't pad with low-value observations.

# Common False Positives — Do NOT Flag These

- **Missing event emission checks.** Specs verify state transitions and invariants, not event emissions. Do not flag the absence of `ensures` for `emit` calls — event firing is not a correctness property the prover tracks.
- **Overcomplexing simple getters.** Functions that just read and return a field (especially those with `no_opaque`) need minimal spec coverage. Do not flag a getter for missing ensures beyond the return value, or insufficient complexity in the spec.
- **Unchanged fields on immutable objects.** Do not flag specs for failing to assert that fields of an immutable reference (`&Object`) remain unchanged. Immutable references cannot be mutated — the type system already guarantees field preservation. Only flag missing unchanged-field checks for *mutable* references (`&mut Object`).

# Special Cases
- **`ignore_abort`**: Abort modeling deferred. Focus on ensures. Not an issue.
- **`no_opaque`**: Note if present. Not an issue.
- **Missing spec**: Identify what should be verified based on the writeup.

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

Review the spec for `staking_pool_specs::staked_sui_amount_spec`. Compare the actual spec against the writeup (the verification plan), using the function source, callees, and protocol context to assess coverage.

Produce a detailed markdown review using `format_review_md()` and return it via `final_result()`. The review format and guidelines are in your system prompt.

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


## Writeup (Verification Plan)

This is what should be verified about `staking_pool_specs::staked_sui_amount_spec`:

```yaml
function: staking_pool_specs::staked_sui_amount_spec
complexity: low
summary: Returns the principal SUI balance of a StakedSui object by reading the value
  of its principal Balance<SUI> field.
role: Pure accessor exposed as the canonical way to read how much SUI a staking receipt
  holds. Used directly in reward-calculation logic (calculate_rewards calls staked_sui.amount())
  and available as a method alias (.amount()) on StakedSui throughout the protocol.
aborts: []
requires: []
ensures:
- condition: result == staked_sui.principal.value()
  reason: The function is defined as a single-expression delegate to Balance::value();
    the returned u64 must equal the underlying balance value.
- condition: result >= MIN_STAKING_THRESHOLD (1_000_000_000)
  reason: Any valid StakedSui produced by request_add_stake or split enforces the
    minimum threshold invariant; the accessor reflects that the principal can never
    be below 1 SUI for a properly-constructed object.
observations:
- The function has zero branches, zero aborts, and zero side effects — it is a transparent
  projection.
- The result is the principal deposit amount and does NOT include accrued rewards;
  callers wanting full redemption value must combine this with the exchange-rate computation
  in calculate_rewards.
- The MIN_STAKING_THRESHOLD postcondition is a system-level invariant rather than
  something enforced inside this function; it is appropriate as a spec assumption
  (requires) rather than a guaranteed ensures in isolation.
```

## Actual Spec

```move
#[spec(prove, target=staking_pool::staked_sui_amount, no_opaque)]
fun staked_sui_amount_spec(
    staked_sui: &StakedSui,
): u64 {
    staking_pool::staked_sui_amount(staked_sui)
}
```

Statement counts: 0 asserts, 0 requires, 0 ensures | Flags: no_opaque

## Target Function

```move
/// Returns the principal amount of `StakedSui`.
public fun staked_sui_amount(staked_sui: &StakedSui): u64 { staked_sui.principal.value() }
```

## Full Module (`staking_pool.move`)

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

## Entry Points Reaching `staking_pool_specs::staked_sui_amount_spec`

- **`staking_pool::staked_sui_amount`**: `staking_pool::staked_sui_amount`

## Relevant Structs

No relevant structs found.

---

Review the spec for `staking_pool_specs::staked_sui_amount_spec`. Produce a markdown review via `final_result(format_review_md(function, verdict, issues, analysis, strengths))`.

## Assistant

I'll review this spec by examining the target function, the writeup, and the actual spec coverage.

### Code `toolu_01GCUmRZi31hoUi3Dr6W3huu`

```python
from foxy.skills.project_env.api import get_project_root
from foxy.skills.function_knowledge.api import get_knowledge

project = get_project_root()
k = get_knowledge(project)
fn = k.item("staking_pool::staked_sui_amount")
foxy_inspect(fn.all_accesses)

```

> 17190 in, 183 out, 64254 cache read

