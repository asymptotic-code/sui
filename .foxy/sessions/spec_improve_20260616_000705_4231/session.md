---
schema_version: 1
agent: spec_improve
model: claude-opus-4-8
max_tokens: 16384
interactive: false
effort: null
timeout: 10800
token_budget: null
harness: foxy
parent_session: null
name: improve-staking-pool-spec
description: Add abort coverage and assertions to the sui_amount_spec stub, removing ignore_abort directives.
cwd: /Users/cos/asymptotic/agent/clients/mysten/sui
started_at: 2026-06-16T00:07:05.423267
foxy_commit: 38346c7c25594d3c381dff95b53fe33dba150411
prompt_part_hashes: {"base": "5fdb6c5e65d5df8a", "core": "35fc153c53e2c232", "file_ops": "b76d200c47b2271e", "function_knowledge": "ec5c60d9b1e6f113", "move_query": "1505e7004c926053", "project_env": "21a3de2d42771978", "spec_bitwise_encoding": "753b2ef167b841c0", "spec_improve": "3f7d5b653c99a952", "spec_knowledge": "bba6d819aaeba484", "spec_loop": "26d59a7f8c0f21db", "spec_postcondition": "c9935e5df9cbd57c", "spec_precondition": "74781a107ed639cf", "spec_scenario": "d1ce03efba7186ff", "sui_prover": "e34120bb194de4f8", "sui_prover_guide": "9b7aaa77fb185386", "_global": "8ba04c18cf4fc356e2c4244a9dd3ca559adbf7cac30c49e3dfbf75a8d43bdb42"}
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
  specs_setup              Scaffold a sibling specs Move package next to the main package
  spot_knowledge           Enumerate every auditable location in a Move project -- every **spot** --
  struct_knowledge         Declarative access to Move struct metadata
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

# move_query - Move Package Analysis Skill

> **Prefer knowledge skills for interactive work.** `function_knowledge`, `spec_knowledge`, and `struct_knowledge` provide richer semantic context (writeups, reviews, struct roles, access patterns) and cache results to disk. Reach for `move_query` directly only when you need raw bulk parsing, call graph queries, or compiler checks not covered by the knowledge layer.

Unified skill for analyzing Move smart contract packages, providing function source extraction, metadata, and call graph analysis.

## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `parse_move_project`, `MoveTools`, `source_functions`, `test_functions`

For other API functions, use explicit imports:
```python
from foxy.skills.move_query.api import MoveFunction, extract_module_name
```

## Features

- **Function source extraction** - Get complete source code for any function
- **Function metadata extraction** - Parse functions with attributes, comments, line ranges
- **Function classification** - Distinguish source, test, spec, and test_only functions
- **Call graph analysis** - Get callers, callees, and transitive closures
- **Compiler integration** - Leverage Move compiler internals for precise analysis

## Installation

The `move-query` binary is required. Install from the in-tree source:

```bash
cargo install --path extern-tools/move-query --force
```

If the binary is missing from `PATH`, `MoveTools` methods return an error JSON (`{"error": ..., "exit_code": 127}`) and knowledge skills that depend on call-graph data raise `ProduceError` / `CollectionError`. Install the binary to fix it -- there is no fallback path.

## Primary API - Function Sources & Metadata

### parse_move_project(project_path: str = None) -> Dict[str, Any]

Parse all functions from a Move project, extracting complete source code and metadata. This is the **recommended** method for getting function sources and information.

When `project_path` is None (the default), it auto-resolves from the session environment using `get_project_root()`.

```python
from foxy.skills.move_query.api import parse_move_project

# Auto-resolves project from session environment
result = parse_move_project()

# Or use explicit absolute path
result = parse_move_project("/absolute/path/to/cetus-clmm")

print(f"Found {result['count']} functions")
for qname, func in result['functions'].items():
    print(f"\n{qname} ({func['function_type']})")
    print(f"  Location: {func['file_path']}:{func['line_start']}-{func['line_end']}")
    print(f"  Visibility: {func['visibility']}")
```

**Returns:**
```json
{
  "functions": {
    "factory::pool_id": {
      "function_name": "pool_id",
      "qualified_name": "factory::pool_id",
      "module": "factory",
      "file_path": "cetus-clmm/sources/factory.move",
      "line_start": 26,
      "line_end": 28,
      "visibility": "public",
      "is_entry": false,
      "function_type": "source",
      "attributes": ["#[inline]"],
      "spec_target": null,
      "project": "cetus-clmm",
      "source_code": "public fun pool_id(...): ID { ... }"
    }
  },
  "count": 1,
  "project": "cetus-clmm"
}
```

**Note:** The `source_code` field contains the complete function source, including attributes and comments above the function.

### Function Types
- `source`: Regular source function
- `spec`: Specification function (ends with `_spec` or has target attribute)
- `spec_only`: Spec-only function (marked with `#[spec_only]` or in specs/ directory)
- `test`: Test function (marked with `#[test]`)
- `test_only`: Test-only function (marked with `#[test_only]`)

### Examples

**Extract source for a specific function:**
```python
result = parse_move_project()
target_func = result['functions'].get('pool::create_pool')
if target_func:
    print(target_func['source_code'])
```

**Filter by visibility:**
```python
result = parse_move_project()
public_funcs = [f for f in result['functions'].values() if f['visibility'] == 'public']
```

**Find entry functions:**
```python
result = parse_move_project()
entry_funcs = [f for f in result['functions'].values() if f['is_entry']]
```

**Filter to source functions only (excludes tests and specs):**
```python
from foxy.skills.move_query.api import parse_move_project, source_functions

result = parse_move_project()
src = source_functions(result['functions'])
```

**Get test functions:**
```python
from foxy.skills.move_query.api import parse_move_project, test_functions

result = parse_move_project()
tests = test_functions(result['functions'])
```

**Find leaf functions:**
```python
from foxy.skills.move_query.api import MoveTools, parse_move_project
from foxy.skills.project_env.api import get_packages
import json

pkg = get_packages()[0]  # or iterate over all packages
tools = MoveTools(working_dir=pkg)
result = parse_move_project(pkg)

leaf_functions = []
for func in result['functions'].values():
    callees = json.loads(tools.get_callees(function=func['qualified_name']))
    if not isinstance(callees, dict) and len(callees) == 0:
        leaf_functions.append(func)

print(f"Found {len(leaf_functions)} leaf functions")
```

A leaf function is one that has no callees. Execute move-query on a function to check its callees, and if the number of callees is 0, then it is a leaf function.

## Working with Multi-Package Projects

Call graph methods (`get_callers`, `get_callees`, `get_transitive_closure`) require a **package path** (directory with `Move.toml`), not a project root.

For multi-package projects, first find which package contains the target function using `parse_move_project()` on each package from `get_packages()`, then use that package path for call graph analysis.

## Call Graph API

Use `MoveTools` for call graph queries. The constructor takes a **package path** (a directory containing `Move.toml`):

```python
from foxy.skills.move_query.api import MoveTools
from foxy.skills.project_env.api import get_packages

pkg = get_packages()[0]
tools = MoveTools(working_dir=pkg)

callers = tools.get_callers(function="pool::create_pool")
callees = tools.get_callees(function="pool::create_pool")
closure = tools.get_transitive_closure(function="pool::create_pool")
```

### Call Graph Methods

All methods return JSON strings. Parse with `json.loads()`.

#### get_callers(function: str, path: str = ".") -> str
Get functions that call a specific function (parent functions / reverse dependencies).

```python
callers = tools.get_callers(function="pool::create_pool", path=".")
# Returns: [{"name": "...", "file": "...", "line": N}, ...]
```

#### get_callees(function: str, path: str = ".") -> str
Get functions that this function calls (child functions / forward dependencies).

```python
callees = tools.get_callees(function="pool::create_pool", path=".")
# Returns: [{"name": "...", "file": "...", "line": N}, ...]
```

#### get_transitive_closure(function: str, path: str = ".") -> str
Get transitive closure of all dependencies for a function, including their source code.

**IMPORTANT:** `path` must be a **package directory** containing `Move.toml`. For multi-package projects, first find which package contains the function using `parse_move_project()` on each package from `get_packages()`.

```python
closure = tools.get_transitive_closure(function="pool::create_pool", path="./contracts/protocol")
# Returns: {"dependencies": [{"name": "module::func", "body": "fun func() {...}"}, ...], "count": N}
```

Each dependency includes:
- `name`: Fully qualified function name (e.g., "0x2::coin::mint")
- `body`: Complete source code of the function (empty string for native/intrinsic functions)

#### get_transitive_closure_names_only(function: str, path: str = ".") -> str
Get transitive closure with only function names (no source code).

```python
closure = tools.get_transitive_closure_names_only(function="pool::create_pool", path=".")
# Returns: {"dependencies": ["module::func1", "module::func2", ...], "count": N}
```

#### get_functions_topological(path: str = ".") -> list[str]
Return all source functions in a package in **callee-before-caller** topological order. Leaf functions (those that call nothing, or only externals) come first; entry points come last.

```python
order = tools.get_functions_topological(path="./contracts/protocol")
# ["module::leaf_helper", "module::mid_helper", ..., "module::entry_fn"]
```

Result is cached per package path for the life of the process. Call `invalidate_topological_cache(path)` after editing Move source if you need a fresh order.

#### check_compilation(path: str = ".") -> str
Check if a Move package compiles successfully. Returns compilation result with any errors.

```python
result = tools.check_compilation(path=".")
# Returns:
# {"success": true, "errors": []}  # if compilation succeeded
# {"success": false, "errors": "error messages..."}  # if compilation failed
```

Use this to validate packages or extract compilation errors for debugging.

**Error handling:**
```python
import json
result = tools.get_callers("foo::bar")
data = json.loads(result)
if "error" in data:
    print(f"Error: {data['error']}")
else:
    for func in data:
        print(func["name"])
```

#### get_functions_spec_metadata(functions: Optional[List[Tuple[str, str]]] = None, path: str = ".") -> Dict
Get spec metadata (inputs, outputs, imports) for functions. This is the **batch-optimized** method that compiles the project only once.

- If `functions` is None or empty, returns metadata for ALL functions in the project (recommended for multiple specs)
- If `functions` is provided, returns metadata only for those specific functions
- Results are cached per project path for subsequent calls

```python
# Get metadata for ALL functions (compiles once, caches result)
all_metadata = tools.get_functions_spec_metadata(path=".")
# Returns: {"module::function": {...}, "module2::function2": {...}, ...}

# Get metadata for specific functions
metadata = tools.get_functions_spec_metadata(
    functions=[("coin", "mint"), ("pool", "swap")],
    path="."
)

# Access individual function metadata
mint_details = all_metadata.get("coin::mint")
```

Each function's metadata structure:
```json
{
  "import_target": {
    "package": "package_name",
    "module": "module_name",
    "type_params": ["TypeParam"],
    "transfer": false
  },
  "input_params": [
    {
      "name": "param_name",
      "struct": "StructName",
      "import_params": {
        "package": "package_name",
        "module": "module_name",
        "name": "StructName"
      }
    }
  ],
  "output_params": [
    {
      "name": "result_0",
      "struct": "StructName",
      "import_params": {...}
    }
  ]
}
```

### Package Queries

#### get_sources(path: str = ".") -> str
Get all source files and manifest from a package. Returns the binary's raw output as a JSON string.

```python
sources = tools.get_sources(path=".")
```

#### get_functions(path: str = ".") -> str
Get all functions in a package as a flat list of `{"name", "file", "line"}` entries. For full source code, attributes, and metadata, use `parse_move_project()` instead -- this method exists for fast name-only listings.

```python
functions = tools.get_functions(path=".")
# [{"name": "module::function", "file": "path/file.move", "line": 42}, ...]
```

## CLI Interface

```bash
# Get all source files and manifest
move-query get-sources --path <package-path>

# Get all functions
move-query get-functions --path <package-path>

# Get callers (parent functions)
move-query get-callers --function <function-name> --path <package-path>

# Get callees (child functions)
move-query get-callees --function <function-name> --path <package-path>

# Get transitive closure (with source code)
move-query get-transitive-closure --function <function-name> --path <package-path>

# Find struct by name (comma separated for multiple)
move-query find-struct --name <struct1>,<struct2> --path <package-path>

# Check compilation
move-query check-compilation --path <package-path>

# Get spec metadata for all functions (batch mode - compiles once)
move-query get-functions-spec-metadata --path <package-path>

# Get spec metadata for specific functions (comma-separated module::function format)
move-query get-functions-spec-metadata --path <package-path> --functions coin::mint,pool::swap
```

Function names use fully qualified format: `module::function_name`

## Response Formats

### parse_move_project()
Returns dict with keys: `functions` (dict keyed by qualified_name), `count` (int), `project` (str)

Each function includes:
- `source_code`: Complete function source with attributes/comments
- `line_start`, `line_end`: Exact line range in source file
- Full metadata (visibility, entry, type, attributes, etc.)

### Call graph methods (get_callers, get_callees)
Return arrays: `[{"name": "...", "file": "...", "line": N}, ...]`

### get_transitive_closure()
Returns dict: `{"dependencies": [{"name": "...", "body": "..."}, ...], "count": N}`

### get_transitive_closure_names_only()
Returns dict: `{"dependencies": ["name1", "name2", ...], "count": N}`

### get_functions_spec_metadata()
Returns dict keyed by `module::function`: `{"coin::mint": {"import_target": {...}, "input_params": [...], "output_params": [...]}, ...}`

### Error responses (all methods)
`{"error": "message", "exit_code": 1}`

## Notes

- All paths auto-resolve from session environment when not specified (fallback: current directory)
- Function names use fully qualified format: `module::function_name`
- The `move-query` binary must be on `PATH`. Install it from `extern-tools/move-query` (see Installation).


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

# Spec Knowledge

Declarative access to spec pipeline data. Ephemeral fields are always fresh (re-computed on access). Cached fields persist to disk and are produced by agents on demand.

## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `get_spec_knowledge` (aliased from `get_knowledge`)

For other API functions, use explicit imports:
```python
from foxy.skills.spec_knowledge.api import SpecCollection, create_empty_spec, create_empty_specs_for_module
```

## Key

The key is `spec_module::spec_function_name` -- the qualified name of the **spec function itself**.

Examples: `"borrow_specs::borrow_spec"`, `"obligation_specs::accrue_debt_preserves"`, `"decimal_specs::pow_loop_inv"`

Not valid: `"borrow::borrow"` (that's the source function, not the spec), `"borrow_spec"` (missing module), `"0x2::borrow_specs::borrow_spec"` (no address prefix).

```python
from foxy.skills.spec_knowledge.api import get_knowledge

k = get_knowledge(project_path)
spec = k.item("borrow_specs::borrow_spec")  # key = spec_module::spec_function_name

a = spec.attributes            # ephemeral -- parsed #[spec] attributes (prove, skip, target, no_opaque, timeout, etc.)
src = spec.source              # ephemeral -- the spec function's own source code
t = spec.target                # ephemeral -- the TARGET source function (or None for spec_only)
c = spec.callees               # ephemeral -- target's direct callees
tc = spec.transitive_closure   # ephemeral -- target's transitive deps
# tc["count"]      -> int
# tc["functions"]  -> list of {"name": "module::func"}
ec = spec.entry_chains         # ephemeral -- entry point chains reaching target
st = spec.structs              # ephemeral (uses tc dep)
wp = spec.writeup              # cached -- runs spec_writeup agent if missing
rv = spec.review               # cached -- runs spec_review agent if missing (returns markdown string)
sm = spec.summary              # cached -- runs spec_summary agent if missing
# sm["one_liner"]  -> str, under 120 chars
# sm["paragraph"]  -> str, scaled to complexity
wr = spec.improve              # cached -- runs spec_improve agent if missing
# wr["starting_state"]  -> str (stub | asserts_only | semantics | complete)
# wr["final_state"]     -> str
# wr["changes_made"]    -> list of str
# wr["prover_final"]    -> str (pass | fail | timeout)
fr = spec.flip_requires        # cached -- runs spec_flip_requires agent if missing; mutates the spec file in place
# fr["converted"]            -> int (requires promoted to asserts)
# fr["removed"]              -> int (requires deleted as unnecessary)
# fr["macroed"]              -> int (requires wrapped in a documented macro fun)
# fr["remaining_requires"]   -> int (bare requires left — should be 0)
# fr["final_prover_status"]  -> str (pass | fail | timeout)
ln = spec.lean                 # cached -- runs lean-backend on this spec
# ln["status"]        -> "success" | "timeout" | "error"
# ln["passed"]        -> bool (status == success and exit_code == 0)
# ln["exit_code"]     -> int | None
# ln["module"]        -> spec module name
# ln["function"]      -> spec function name
# ln["package_path"]  -> absolute Move package path that was invoked
# ln["output_tail"]   -> last ~80 lines of lean-backend stdout/stderr
# ln["error"]         -> str | None (populated on error/timeout)
ir = spec.lean_ir              # ephemeral -- deterministic Lean IR locator
# ir["status"]             -> "success" | "error"
# ir["target_qualified"]   -> e.g. "u64::pow10_u64"
# ir["lean_output_dir"]    -> per-module Lake project root under output/
# ir["correctness_file"]   -> Correctness/<TargetPascal>.lean (the backend-emitted manifest)
# ir["lean_module_file"]   -> the .lean file that contains `def <spec_func>` defs
# ir["lean_namespace"]     -> e.g. "U64" (namespace of generated module)
# ir["lean_import_path"]   -> e.g. "MoveStdlib.U64" (usable in `import ...`)
# ir["proofs_file"]        -> Proofs/<ModuleStem>_<spec_func>.lean (scaffold target)
# ir["generated_defs"]     -> [{kind: "spec"|"aborts"|"ensures", name, signature}, ...]
pf = spec.proof_with_lean      # cached -- runs the spec_proof_with_lean agent
# pf["lake_passed"]      -> bool
# pf["theorems"]         -> [{"name": str, "kind": "aborts"|"ensures", "status": "proved"|"sorry"|"failed"}]
# pf["changes_made"]     -> list[str] describing agent edits
# pf["lake_runs"]        -> int (number of lake build invocations)
# pf["lake_output_tail"] -> str (last ~40 lines of final lake output)
# pf["issues"]           -> list[str] describing unresolved goals
# pf["duration_seconds"] -> int
```

## Available Fields

| Field | Type | Cache | Dependencies | Description |
|-------|------|-------|-------------|-------------|
| `attributes` | JSON | no | none | Parsed `#[spec]` attributes: prove, skip, target, no_opaque, timeout, etc. |
| `source` | Move | no | none | The spec function's own source code (delegates to `function_knowledge`) |
| `target` | JSON | no | none | Target source function metadata + source (None for spec_only with no target) |
| `callees` | JSON | no | target | Target's direct callees with inline source |
| `transitive_closure` | JSON | no | target | Count and names of target's transitive deps |
| `entry_chains` | JSON | no | target | Entry point call chains reaching target |
| `requires` | JSON | no | source | Parsed requires/asserts from spec source |
| `structs` | JSON | no | target, transitive_closure | Structs from dep modules with inline source |
| `summary` | JSON | yes | source, target | One-liner + paragraph from `spec_summary` agent |
| `writeup` | YAML | yes | target, callees, transitive_closure, entry_chains, structs | Spec writeup from `spec_writeup` agent |
| `review` | Markdown | yes | source, writeup, target, callees, transitive_closure, entry_chains, structs | Detailed spec review from `spec_review` agent |
| `improve` | YAML | yes | source, writeup, target, callees, transitive_closure, entry_chains, structs | Work report from `spec_improve` agent — what was changed, prover result, final state |
| `flip_requires` | YAML | yes | source, target, callees, transitive_closure, entry_chains, structs, module_source | Runs `spec_flip_requires` agent to promote `requires` to `asserts` (or wrap surviving ones in documented macros); mutates the spec file in place |
| `lean` | YAML | yes | none | Runs the `lean_backend` binary on this spec (`--modules`/`--functions` filtered) and records pass/fail + output tail |
| `lean_ir` | JSON | no | none | Deterministic locator for this spec's Lean IR (package path, per-module output dir, `Correctness/<TargetPascal>.lean` manifest, the `.lean` file containing the spec defs, namespace, import path, proofs-file target, and the list of generated defs). Anchored off the spec key and the backend-emitted `Correctness/` manifest, not filesystem heuristics. |
| `proof_with_lean` | YAML | yes | source, target, lean_ir | Consumes `lean_ir` for file locations, scaffolds `Proofs/<Module>_<spec>.lean` with theorem stubs, and runs an agent that writes proofs and iterates with `lake build` until it passes (or gives up with a report). Also emits two sibling files alongside the YAML: `spec.proof_with_lean.proof.html` (what was proved — generated defs + final theorem source) and `spec.proof_with_lean.trace.html` (the agent's path — work report, theorem statuses, lake tail, link to session transcript). The YAML records the agent's `session_dir` so the trace page is regenerable later. |

## spec_only Functions

Functions with `#[spec_only]` (loop invariants, helper predicates, scenario specs) have no target. For these:
- `source` returns the spec_only function's source code
- `target` returns `None`
- `callees`, `transitive_closure`, `entry_chains`, `structs` return empty results

```python
spec = k.item("obligation_specs::accrue_debt_preserves")
spec.source  # the helper predicate's source
spec.target  # None (no target)
```

## Attributes Field

The `attributes` field parses the `#[spec(...)]` / `#[spec_only]` attribute into a structured dict that mirrors the raw flags without collapsing them:

```python
spec.attributes
# {
#     "prove": True,                    # prove flag present
#     "skip": False,                    # skip flag present (can coexist with prove)
#     "skip_reason": None,              # string from skip=b"reason", or None
#     "target": "module::func",         # target function, or None
#     "no_opaque": True,                # no_opaque flag present
#     "timeout": None,                  # integer from timeout=N, or None
#     "spec_only": False,               # has #[spec_only] attribute
#     "focus": False,                   # focus flag present (single-spec debug mode)
#     "ignore_abort": False,            # ignore_abort flag present (skip abort coverage)
#     "uninterpreted": ["mod::func"],   # list of uninterpreted functions
#     "include": ["specs::other_spec"], # list of include = paths (other specs composed in)
#     "boogie_opt": ["vcsSplit..."],    # list of boogie options
#     "extra_bpl": ["../../file.bpl"],  # list of extra BPL files
# }
```

Flags are independent — a spec can have both `prove` and `skip` (meaning it was written to be proven but is currently skipped). Filter with boolean combinations:

```python
k = get_knowledge(project_path)

skipped = [key for key in k.keys() if k.item(key).attributes["skip"]]
proven = [key for key in k.keys() if k.item(key).attributes["prove"] and not k.item(key).attributes["skip"]]
no_prove = [key for key in k.keys()
            if not k.item(key).attributes["prove"]
            and not k.item(key).attributes["skip"]
            and not k.item(key).attributes["spec_only"]
            and k.item(key).attributes["target"]]
```

## Remote vs local prover

Specs run on the **cloud prover** by default (via `sui-prover --cloud`, which the cron pipeline uses). Individual specs can opt out by adding `run_on = b"local"` to their `#[spec(...)]` annotation; that attribute forces the local prover for just that spec and is only needed when the spec cannot run in the cloud (local `extra_bpl`, interactive iteration, etc.). Prefer the default and only reach for `run_on = b"local"` when you truly need it.

## Batch Parallelism — per-module mutex for file-mutating fields

`improve` and `flip_requires` both **mutate the spec file in place**. Multiple
agents editing the same `*_specs.move` file concurrently silently overwrite
each other's `ensures` / `asserts`.

`SpecCollection.submit_batch` enforces a per-module mutex for these fields
automatically: specs in the same spec module (e.g. `i64_specs::*`) run as one
sequential chain, while different modules run in parallel.

```python
# 31 specs across 4 modules → 4 parallel chains, each chain sequential inside
batch = k.submit_batch("improve", keys=spec_keys, label="bulk-improve")
```

To override (rare — only when you know the field does not mutate the file):

```python
batch = k.submit_batch("improve", keys=spec_keys, serialize_by=False)  # full parallel
batch = k.submit_batch("improve", keys=spec_keys, serialize_by=lambda k: k)  # one-per-group
```

For other fields (`writeup`, `review`, `summary`) that only write into the
per-spec `.foxy/` cache directory, parallelism is unrestricted by default —
each spec writes to its own file.

## Storage

Cached fields are colocated with spec source files:
```
{spec_file_dir}/{stem}.foxy/{spec_func_name}/spec.{field}.{ext}
```

Example: `contracts/specs/sources/user/borrow_specs.foxy/borrow_spec/spec.writeup.yaml`

Use `spec.path("writeup")` to discover the actual path at runtime:

```python
spec.path("writeup")   # actual path
spec.exists("writeup")  # check without producing
```

Ephemeral fields (`attributes`, `source`, `target`, `callees`, `transitive_closure`, `entry_chains`, `requires`, `structs`) are not written to disk.

## Scaffolding — Creating Empty Spec Stubs

Use `Collection.create()` to scaffold an empty spec for a source function that doesn't have one yet.
The key passed to `create()` is the **target function key** (`module::function_name`), not a spec key.
It returns the spec key of the newly created spec.

```python
from foxy.skills.spec_knowledge.api import get_knowledge
from foxy.skills.project_env.api import get_project_root

k = get_knowledge(get_project_root())

# Check if a spec exists for a source function
k.has_key("pool_specs::swap_spec")  # bool -- key must be the spec key

# Scaffold an empty spec from a target function key
spec_key = k.create("pool::swap")
# Returns "pool_specs::swap_spec"
# Writes scaffold to specs/sources/pool_specs.move

# Now accessible as a normal spec item
spec = k.item(spec_key)
spec.source  # the generated empty scaffold
```

The scaffold has `#[spec(prove, target=pool::swap, ignore_abort)]` and calls the target function
with its correct signature. It's an `ignore_abort` stub — a valid starting point for `spec_improve`
to build on.

### Batch scaffolding

```python
from foxy.skills.function_knowledge.api import get_knowledge as get_fk

fk = get_fk(get_project_root())
sk = get_knowledge(get_project_root())

for fn_key in fk.keys():
    module, func = fn_key.rsplit("::", 1)
    spec_key = f"{module}_specs::{func}_spec"
    if not sk.has_key(spec_key):
        sk.create(fn_key)
```

### Low-level scaffolding API

Two public functions are also available for callers that need direct file placement control:

```python
from foxy.skills.spec_knowledge.api import create_empty_spec, create_empty_specs_for_module

# Single function
result = create_empty_spec(
    module="pool",
    function="swap",
    target_location="./my-project",   # Move.toml location
    spec_location="./specs/sources",  # where to write the .move file
    force=False,                      # True = replace existing
)
# result: {success, spec_code, file_path, action, imports_added}

# All functions in a module
result = create_empty_specs_for_module(
    module_name="pool",
    target_location="./my-project",
    spec_location="./specs/sources",
    force=False,
)
# result: {success, total, inserted, skipped_duplicates, failed, results}
```

## Item Operations

```python
spec.exists("writeup")       # check disk without producing
spec.path("writeup")         # Path to file (works before file exists)
spec.is_stale("writeup")     # True if a cached dep changed (ephemeral deps are always fresh)
spec.get("writeup", force=True)       # force regeneration
spec.get("writeup", stale_ok=False)   # regenerate if stale
```

## Related Skills

Skills under the `spec_` umbrella (load with `load_skill("name")`):

| Skill | Purpose |
|-------|---------|
| `spec_fixer` | Adapt specs after client code changes (renamed functions, changed signatures) |
| `spec_improve` | Improve a spec from stub to complete (dispatched by `spec.improve`) |
| `spec_review` | Review spec quality and gaps (dispatched by `spec.review`) |
| `spec_flip_requires` | Promote `requires` to `asserts` in a spec; wrap surviving ones in documented macros (dispatched by `spec.flip_requires`) |
| `spec_writeup` | Plan verification approach (dispatched by `spec.writeup`) |
| `spec_summary` | Summarize function and verified properties (dispatched by `spec.summary`) |
| `spec_proof_with_lean` | Prove a spec in Lean 4 against generated IR (dispatched by `spec.proof_with_lean`) |
| `spec_precondition` | Reference: writing `asserts`/`requires` for abort coverage |
| `spec_postcondition` | Reference: writing `ensures` for function behavior |
| `spec_loop` | Reference: writing loop invariants and quantifiers |
| `spec_scenario` | Reference: writing multi-step scenario specs |


# sui_prover - Sui Move Formal Verification Skill

Run formal verification on Move packages using the Sui Prover.

## No specs yet? Scaffold first.

If `run_sui_prover` returns output containing `"No specifications found in the project. Nothing to verify."`, the project has source code but no spec functions yet. **Do not stop there** — bootstrap a specs package and continue:

```python
docs = load_skill("specs_setup")
foxy_inspect(docs)
from foxy.skills.specs_setup.api import setup_specs_package

scaffold = setup_specs_package()                     # auto-detects main package
foxy_inspect(scaffold)

# Now the project has main_pkg/specs/. Author specs (spec_knowledge.create per
# function, or hand-write *_specs.move under specs/sources/) and re-run:
result = await run_sui_prover(scaffold["specs_path"])
```

The scaffolder creates `<main_pkg>/specs/` with `Move.toml` wired to depend on the main package, removes the auto-generated `tests/` folder, and registers the package in `.foxy/project.toml` so `get_packages()` picks it up. See `specs_setup/SKILL.md`.

## Bitwise-heavy code? Set up `specs-bv/` without asking.

The integer encoding cannot relate `x & MASK` to `x mod 2^k`, cannot model arithmetic right shift, and cannot derive `(x >> (w-1)) as u8 ∈ {0, 1}` for wide `x`. When a project contains code that depends on those relationships — ripple-carry adders (`while (carry != 0) { sum ^= carry; carry = (a & b) << 1; }`), two's-complement signed integers backed by a `bits: uN` field, `&`-masked low/high extractors, arithmetic right shift on `Ixx` — the right move is the bitvector encoding via a sibling `specs-bv/` package run with `--no-bv-int-encoding`.

**Do this autonomously when you see the symptoms; do not stop and ask the user.** Symptoms that trigger it:

- A `_Check` goal fails with `code should not abort` pointing at a `&` / `|` / `<<` / `>>` line, or at an `as uN` cast that follows one.
- A `_Assume` counter-example shows the prover assigning a nonsense value to a bitwise-mask result (e.g. `((sum & LO_64_MASK) as u64) = 1` when `sum = 2^64`).
- The target body contains a ripple-carry adder loop, a `bits: uN`-backed signed integer struct, or `(v.bits >> (w-1)) as u8`.
- A spec times out specifically inside a bitwise mixing chain (not a generic timeout — one localized to bitwise/shift ops).

When any of those fire, follow this playbook end to end without prompting:

1. **Create `<project>/specs-bv/`** next to `specs/`. Standalone Move package — **no dependency on the main package**. `Move.toml` declares one address (e.g. `<project>_specs_bv = "0x0"`).
2. **Register it in `.foxy/project.toml`** under `[packages]`.
3. **Redeclare** the struct (just the `bits` field) and the function(s) you're proving locally in `specs-bv/sources/<module>.move`. Strip everything you don't need. Keep `module integer_library_specs_bv::<module>` (or your project's analogue) as the module path.
4. **Write the spec in the same file**, phrased entirely on `.bits` (no `to_int()` / `Integer`). For ripple-carry adders, declare a loop invariant `(n1.bits + n2.bits) mod 2^w == (sum + carry) mod 2^w` (cast both sides up by one width) **immediately before** the `while` — the prover rejects invariants placed inside the loop body.
5. **Run** `( cd specs-bv && sui-prover --cloud --no-bv-int-encoding )`. The main `specs/` run keeps `--cloud` without `--no-bv-int-encoding` — the flag is invocation-wide and slow on plain arithmetic.
6. **In `specs/`**, the matching spec keeps `ignore_abort` (the int-encoding placeholder) with a brief comment pointing at the BV file. Do not strip the `prove` flag wholesale — the int-encoded spec still provides signature/linkage checks. Only drop `prove` if the spec adds zero value beyond what the BV version proves.

Only stop and ask when (a) the BV port itself fails after a genuine attempt with a loop invariant in place, or (b) the proof needs a `prelude_extra.bpl` Boogie glue (e.g. `$AShr'Bv32'` for arithmetic right shift) you cannot write without their input. The user opted into "prove this project" — bitvector encoding for bitwise code is the proper tool, not a separate decision.

Known BV limits to accept gracefully (still `ignore_abort` after porting):

- **Nonlinear multiplication on wide bit-vectors** (`u64 → u128` widening multiply, `overflowing_mul` on `math_u64` / `math_u128`). Z3 cannot dispatch `bv128 = bv64 * bv64` within practical timeouts. Keep the algebraic spec in `specs/` (e.g. `prod <= MAX ==> (r as u_wide) == prod`) and document the limit.
- **Long signed-int operation chains** (`add` / `sub` / `mul` / `div` / `mod` / `abs` / `shr` on `i32`/`i64`/`i128`) that compound multiple bitwise primitives. Prove the load-bearing core (`wrapping_add`, `sign`, `or`, `and`, `shl`) in `specs-bv/`; chain operations are implied by the core but bundling them blows past the timeout.
- **Arithmetic right shift on `Ixx`** — needs `public native fun ashr(...)` + `procedure {:inline 1} $0_<mod>_ashr` in `prelude_extra.bpl`. Ask before writing the Boogie glue.

See `sui_prover_guide` → "Bitvector encoding (`--no-bv-int-encoding`)" for the full playbook including writing rules and Boogie glue patterns.

## Auto-imported

When this skill is loaded, the following are available directly in the agent namespace:
- `run_sui_prover`

For other API functions, use explicit imports:
```python
from foxy.skills.sui_prover.api import SuiProver, check_prover_success
```

## API

```python
from foxy.skills.sui_prover.api import run_sui_prover
```

## Function Signature

```python
async def run_sui_prover(
    path: str,
    args: List[str] = None,
    timeout: int = 300,
    env: Dict[str, str] = None,
    label: str = None,
    model: str = "claude-sonnet-4-6",
) -> dict
```

**Parameters:**
- `path` - ABSOLUTE path to directory containing Move.toml
- `args` - Arguments to pass to sui-prover (default: `["--timeout", "60", "--force-timeout", "--skip-spec-no-abort"]`)
- `timeout` - Hard timeout in seconds for process termination (default: 300)
- `env` - Additional environment variables (merged with defaults)
- `label` - Label for the fork (default: "sui-prover")
- `model` - Model for the analysis fork (default: "claude-sonnet-4-6")

**Returns:** Fork result dict with `result` containing a compact structured analysis

**Behavior:**
- Forks the agent via `fork` -- the parent continues working immediately
- The fork runs sui-prover, analyzes the full output, and returns a compact report: one line per spec (PASS/FAIL with phase, error, source location, counterexample values), plus totals
- All diagnostic details needed to fix failures are preserved, but verbose prover noise is stripped

**Always use `run_sui_prover` instead of `_run_sui_prover_sync`.** The fork handles the heavy output parsing so the parent context stays clean. Never run the prover inline — it floods the conversation with raw traces.

**One spec per call when iterating.** When verifying multiple specs, issue one `run_sui_prover` call per spec with a single `--functions <name>` flag, not a single call that batches many `--functions` flags or targets many specs at once. Batched runs buffer all output until the whole batch finishes, so the human watching the session sees nothing for minutes. One-spec-per-call gives live incremental output after each spec and makes failures obvious immediately. Iterate in Python:

```python
for name in spec_short_names:
    result = await run_sui_prover(
        path,
        args=["--timeout", "60", "--force-timeout",
              "--skip-spec-no-abort", "--skip-fun-no-abort",
              "--functions", name],
        timeout=240,
        label=f"prove-{name}",
    )
    foxy_inspect(result.get("response", "")[-2000:])
```

### Iteration vs. final-verdict flags — known trap

`--skip-spec-no-abort` and `--skip-fun-no-abort` weaken verification: they skip
the phases that prove "this spec never aborts" and "this function never aborts".
They are convenient during iteration because the no-abort phases are often the
slowest and least informative when you are still writing ensures.

But they create a real footgun: **a spec that "passes" with `--skip-spec-no-abort`
may fail when the same flags are dropped.** Concretely, if an agent removes
`ignore_abort` from a spec attribute (telling the prover "this function does not
abort") and then iterates with `--skip-spec-no-abort` (telling the prover "skip
the check that verifies that claim"), the resulting "pass" is a false positive
— the very obligation the agent took on was not verified.

**Rule:** when an iteration verdict drives a final state (commit, cache, "done"
notification), the iteration must use **at least** the flag set the project's
CI uses. Dropping a `--skip-*` flag for the final batch run is too late — the
agent has already written code based on the weaker verdict.

Practical guidance:

- Use `--skip-spec-no-abort` only when keeping `ignore_abort` on the spec, or
  when you have written abort-coverage `asserts` AND verified them with the
  flag removed at least once.
- Mirror the project's CI flag set. If CI does not pass `--skip-spec-no-abort`,
  iteration should not pass it either.
- Treat `--skip-*` flags as "diagnostic mode" — fine for exploring a slow proof,
  not fine for "the spec is done".

### Sync escape hatch

For direct blocking execution (used by CLI), use `_run_sui_prover_sync`:

```python
from foxy.skills.sui_prover.api import _run_sui_prover_sync

result_json = _run_sui_prover_sync(path, args=None, timeout=300, env=None)
```

Returns a JSON string with one of these shapes:

Success:
```json
{
    "status": "success",
    "output": "combined stdout and stderr from sui-prover",
    "exit_code": 0
}
```

**Important:** `status: "success"` only means the process ran. Check `exit_code` to determine if verification actually passed (exit_code != 0 means verification failed).

Timeout:
```json
{
    "status": "timeout",
    "message": "Prover execution timed out after 300 seconds",
    "partial_output": "output captured before timeout"
}
```

Interrupted (Ctrl-C):
```json
{
    "status": "interrupted",
    "message": "Prover execution interrupted by user (Ctrl-C)",
    "partial_output": "output captured before interruption"
}
```

Error:
```json
{
    "status": "error",
    "error": "sui-prover not found in PATH"
}
```

**Default environment variables:**
```python
{
    "DOTNET_GCHeapHardLimit": "6442450944",  # 6GB heap limit
    "DOTNET_GCServer": "0",                  # Workstation GC mode
    "DOTNET_GCRetainVM": "0",                # Don't retain VM
    "DOTNET_GCConserveMemory": "9",          # Aggressive memory conservation
}
```

## Cloud Mode (`--cloud`)

**Always pass `--cloud` in `args`.** Cloud mode offloads verification to a shared proving cluster — dramatically faster than local prover, runs proof goals in parallel, and doesn't consume local CPU/memory. Include it in every `run_sui_prover` call:

```python
args = ["--timeout", "60", "--force-timeout", "--skip-spec-no-abort", "--cloud"]
```

Heavy specs that genuinely cannot run in the cloud opt out **at the spec level**, not at the call site — annotate the spec with `#[spec(..., run_on = b"local")]` (documented in the Prover attributes section). Leave `--cloud` in `args`; the prover reads per-spec `run_on` and runs those specs locally while everything else goes to the cloud.

Drop `--cloud` only when:
- The cloud endpoint is unreachable and verification is blocking
- You need local-only features: `--keep-temp`, `--generate-only`, `--dump-bytecode`, or `extra_bpl`
- The user explicitly requests local execution

### Cloud Configuration

Cloud mode reads credentials from `~/.asymptotic/sui_prover.toml`. If this file is missing or incomplete, sui-prover fails with an error like "cloud config not set" / "missing url" / "missing key" when `--cloud` is used.

**Fix:** create `~/.asymptotic/sui_prover.toml` with the shared cluster config:

```toml
url = "https://hdhs3sbtlk22w5ceivvbuege4i0devap.lambda-url.us-west-2.on.aws/"
key = "test_password"
concurrency = 20
```

```python
from pathlib import Path

cfg_dir = Path.home() / ".asymptotic"
cfg_dir.mkdir(parents=True, exist_ok=True)
cfg_path = cfg_dir / "sui_prover.toml"
if not cfg_path.exists():
    cfg_path.write_text(
        'url = "https://hdhs3sbtlk22w5ceivvbuege4i0devap.lambda-url.us-west-2.on.aws/"\n'
        'key = "test_password"\n'
        'concurrency = 20\n'
    )
```

Fields:
- `url` — cloud prover endpoint
- `key` — shared auth token
- `concurrency` — max parallel proof goals sent from this client (20 is a sensible default)

The config lives **with the prover**, not inside any project. It is read once per invocation from `~/.asymptotic/sui_prover.toml`, so a single write bootstraps every future run on the machine. Once the file exists, re-run the same `run_sui_prover(...)` call that previously failed — no arg changes needed.

## sui-prover Flags

- `--timeout <seconds>` - Verification timeout in seconds (default: 300)
- `--force-timeout` - Force kill boogie process if boogie vc timeout is broken (should always be used with `--timeout` unless user explicitly requests otherwise)
- `--verbose` / `-v` - Display detailed verification progress
- `--split-paths <n>` / `-s <n>` - Split verification into separate proof goals for each execution path. Not used by default; try adding if verification times out or gets stuck
- `--generate-only` / `-g` - Only generate Boogie code without running prover
- `--no-counterexample-trace` - Don't display counterexample trace
- `--skip-spec-no-abort` - Skip checking spec functions that do not abort (should always be included unless user explicitly requests otherwise)
- `--cloud` - Run verification on the shared cloud cluster instead of locally (should always be included — see [Cloud Mode](#cloud-mode--cloud))
- `--modules <MODULES>` - Specify module names to target (comma-separated)
- `--functions <FUNCTIONS>` - Specify function names to target (comma-separated)

**Default Behavior:**
- If neither `--modules` nor `--functions` are provided, sui-prover runs over all specs in the package
- If `--modules` is provided without `--functions`, sui-prover runs over all specs in the specified module(s)
- If both `--modules` and `--functions` are provided, sui-prover runs only for the specified function spec(s) within the specified module(s)

**Default Arguments:**
By default, the following arguments are always included (unless the user explicitly requests otherwise):
- `--timeout <value>` with `--force-timeout` - Ensures proper timeout handling
- `--skip-spec-no-abort` - Skips checking spec functions that do not abort
- `--cloud` - Runs on the cloud prover (see [Cloud Mode](#cloud-mode--cloud))

**Useful Flag for Prover Internals**
Use these options to enable deeper analysis and procedures for analysis.

- `--dump-bytecode` - dumps `*.bytecode` files in the project directory where this is run, detailing analysis passes that ran on the bytecode, for example:

```
...
bytecode_11_reaching_def_analysis.bytecode
bytecode_12_livevar_analysis.bytecode
...
```

- `--keep-temp --boogie-file-mode function` - generate `*.bpl` containing Boogie intermediate verification language code for move functions. **Important** the `*.bpl` files will be in an `output` subdirectory that the prover ran on.
- `--enable-conditional-merge-insertion` - enables a Boogie bytecode pass that transforms Boogie bytecode into SSA-like form with phi nodes

Users can request to remove these defaults or add additional flags as needed.

Always run `sui-prover --help` to see all available flags for your installation.

## Working with Specs

**Key concept:** sui-prover runs on a **package** (directory with `Move.toml`), not a project. Specs typically live in a dedicated `specs/` package separate from `sources/`.

When verifying specific specs:
1. Locate the spec file (use Glob to search `specs/**/*.move`)
2. Find its package root (walk up to the `Move.toml`)
3. Run sui-prover on that specific package with `--modules` and/or `--functions` flags

Use `get_packages()` from `project_env` to discover available packages.

### Spec naming: ProduceError pitfall

If the prover reports `Failed to produce 'source' for '<name>_spec': Function <module>::<name>_spec not found`, the spec function name does not match any function in the module. The `_spec` suffix must match the **exact** name of the function being specified. Before writing a spec, always read the source file to confirm the exact function name. Do not guess or infer names from context.

## Prover attributes
`sui-prover` understands syntax like `#[spec(...)]` as directives for controlling prover behavior and analysis. You should know about these uses, which may be combined:

### Standalone `spec` syntax: `#[spec]`

The `spec` attribute is used to specify that a function is a specification.

When the `spec` attribute is on a function called `<function_name>_spec`, the Move Prover will use the specification instead of the original function when verifying other functions that call `<function_name>`.

In this case the spec function must:

1. have the same name as the original function, but with `_spec` appended -- if this is not the case, the spec will only be used for checking a scenario (see below)
2. have the same signature as the original function
3. call the original function -- the value returned by the spec is usually the value returned by the original function, but this is not required

### `#[spec(prove)]`

Used to specify that a function spec to be verified by the Move Prover. A spec without `prove` will not be checked by the prover. I it will be used when called as part of proving other functions—see [how specs compose in our FAQ](https://www.notion.so/1d5159bbd66d80b3bd4ee7de42fd7ead?pvs=21).

Adding the `focus` attribute to a spec/scenario (`#[spec(prove, focus)`) will make the prover only attempt to prove this particular spec. `focus` can be used for several specs simultaneously**.**

### `#[spec(skip)]`

Skips spec verification. Optionally accepts a string reason (e.g., skip = b"known issue").

###  `#[spec(target=foo:bar)]`

Specifies the function to which the spec belongs (module `foo` function `bar).

### `#[spec(no_opaque)]`

Prevents this spec from being used by other specs during verification (useful for getters and specific isolated cases).

### `#[spec(boogie_opt=b"..")]`

Adds custom Boogie per-spec options. These override default settings if they match.

| Option | When to use |
|--------|-------------|
| `vcsSplitOnEveryAssert` | Many calculations — splits verification conditions |
| `vcsMaxKeepGoingSplits:N` | Limits verification condition splits |
| `useArrayAxioms` | Table access and loops |
| `vcsCores:1` | Single-core for deterministic behavior |
| `proverOpt:O:smt.QI.EAGER_THRESHOLD=100` | Tune quantifier instantiation |
| `proverOpt:O:smt.MBQI=false` | Disable model-based quantifier instantiation |
| `proverOpt:O:smt.random_seed=N` | Try different seeds when proof is seed-sensitive |

Combine multiple options:
```move
#[spec(prove, target=module::func, boogie_opt=b"vcsSplitOnEveryAssert useArrayAxioms")]
```

**Recommended combo for `forall!` specs** (invariant preservation, loop invariants with quantifiers):
```move
boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false vcsCores:1"
```
- `smt.MBQI=false` disables model-based quantifier instantiation — essential for `forall!` to work reliably
- `vcsCores:1` limits parallelism for deterministic behavior with quantifiers

### `#[spec(timeout)]`

Sets a maximum verification timeout for the spec.

### `#[spec(run_on = b"local")]`

Forces this single spec to run on the **local** prover even when `run_sui_prover` is invoked with `--cloud`. Use this **only for heavy specs** that genuinely cannot run in the cloud (e.g. specs that need local `extra_bpl` files, very large proof goals that exceed the cloud worker's budget, or interactive iteration with local prover flags).

```move
#[spec(prove, target=pool::complex_swap, run_on = b"local")]
public fun complex_swap_spec(...) { ... }
```

The rest of the specs in the same run still go to the cloud — only the annotated spec runs locally. Default is cloud; only add this attribute when a spec is demonstrably too heavy or cloud-incompatible.

### `#[spec_only]`

Similar to `test_only`, `spec_only` makes any annotated code (module, function, etc.) only be visible to the prover. The code will not appear under regular compilation, nor in test mode.

### `#[ext(no_abort)]`

Similar to  `#[spec(prove)]` this top-level attribute is applied to regular functions to verify that **they do not abort** during execution.

It can be combined with  `#[spec_only]`, which is particularly useful when working with **quantifiers** or **external loop invariants**.

### `#[ext(pure)]`

Attempt translation of a Move function to a Boogie function (rather than a Boogie procedure). A Boogie function is a pure function in which no jumps or side-effects exist. Conversely, Boogie procedures may contain jumps and are the default translation output of functions unless `#[ext(pure)]` is specified

### `#spec(prove, focus)]`

The `focus` attribute is an **optional** approach that can be used when the user explicitly requests it (e.g., "add focus for function X" or "run sui-prover with focus on these functions"). 

The `focus` attribute can be added to spec/scenario attributes to make the prover only attempt to prove those particular specs:

```move
#[spec(prove, focus)]
public fun swap(pool: &mut Pool, amount: u64): u64 {
    // ...
}
```

Multiple specs can have `focus` applied simultaneously, and the prover will attempt to prove all focused specs in a single run.

**When to use focus:**
- Only when the user explicitly asks to add focus attributes or use focus-based verification
- When the user wants to verify specific functions without using `--modules`/`--functions` flags

**Implementation pattern for focusing (only when explicitly requested):**
1. **Locate spec file and package root** - Follow the steps in [Working with Specs](#working-with-specs) to find the correct spec file and package root.
2. **Add focus attribute** - Temporarily edit the spec file to add `focus` to the target spec attribute(s).
3. **Run Verification** - Run `run_sui_prover()` on the identified package root.
4. **Restore** - Restore the original spec file.

This allows targeted verification of specific functions without running the entire suite, but should only be used when the user explicitly requests it.

## Spec Composition

### How do `spec`s compose?

The Move Prover will use the spec of a function to verify other functions that call it.

For example, if we have a function `foo` that calls `bar`, and we have a spec `bar_spec` for `bar`, the Move Prover will use `bar_spec` to verify `foo`.

Marking a spec with `#[spec(prove)]` will make the prover attempt to prove it.
Leaving out the `prove` attribute will make the prover use the spec to verify other functions, but not attempt to prove it.

## Usage Examples

### Basic Verification (fork, non-blocking)

```python
from foxy.skills.sui_prover.api import run_sui_prover

# Forks the agent -- parent continues working, fork runs prover and analyzes results
result = await run_sui_prover("/absolute/path/to/my_package")
# result["response"] contains the analysis, result["result"] contains structured data
```

### Custom Flags

```python
from foxy.skills.sui_prover.api import run_sui_prover

result = await run_sui_prover(
    path="/absolute/path/to/my_package",
    args=["--timeout", "60", "--force-timeout", "--skip-spec-no-abort", "--verbose"],
)
```

### Verifying Specific Modules

```python
from foxy.skills.sui_prover.api import run_sui_prover

result = await run_sui_prover(
    path="/absolute/path/to/my_package",
    args=["--timeout", "60", "--force-timeout", "--skip-spec-no-abort", "--modules", "<name_of_the_module>"],
)
```

### Verifying Specific Functions

```python
from foxy.skills.sui_prover.api import run_sui_prover

result = await run_sui_prover(
    path="/absolute/path/to/my_package",
    args=["--timeout", "60", "--force-timeout", "--skip-spec-no-abort", "--modules", "<name_of_the_module>", "--functions", "<name_of_the_function>"],
)
```

### Multiple Modules and Functions

```python
from foxy.skills.sui_prover.api import run_sui_prover

result = await run_sui_prover(
    path="/absolute/path/to/my_package",
    args=["--timeout", "60", "--force-timeout", "--skip-spec-no-abort", "--modules", "<module_1>,<module_2>"],
)
```

## Understanding Prover Output

The prover returns verification results in the `output` field:

**Success:**
```
Verified 15 functions in 12.3s
```

**Abort not covered:**
```
error: abort not covered by any of the `aborts_if` clauses
   ┌─ sources/pool.move:42:5
   │
42 │     public fun swap_exact_amount_in(...) {
   │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

**Postcondition failure:**
```
error: post-condition does not hold
   ┌─ sources/pool.move:38:9
   │
38 │         ensures result > 0;
   │         ^^^^^^^^^^^^^^^^^^
```

**Timeout:**
```
error: timeout verifying function
   ┌─ sources/complex.move:100:5
```

## Installation

```bash
# Install sui-prover (part of Sui toolchain)
cargo install --git https://github.com/MystenLabs/sui.git sui-prover

# Check available flags for your version
sui-prover --help
```

## Notes

- **CRITICAL: `sui-prover` is already installed and on PATH.** Never search for it with `find`, `which`, or `locate`. Never attempt to install it. Just call `run_sui_prover()` or `_run_sui_prover_sync()` directly — they invoke `sui-prover` by name.
- **CRITICAL: The `==>` implication operator is NOT supported.** Do not generate specs using `==>`. Use `if`/`else` or `assert!()` with boolean logic instead. For example, instead of `requires a > 0 ==> b > 0`, write the equivalent logic using standard Move control flow.
- **CRITICAL:** Path must be absolute. Relative paths will fail with an error.
- Prover requires valid Move package with Move.toml at the specified path
- The path must contain Move source files (typically in `sources/` directory)
- Specifications must be embedded in the .move files using Move Specification Language (MSL)
- When targeting specific specs, first locate them before running sui-prover
- Default hard timeout is 300 seconds (5 minutes) - process is force-killed if exceeded
- Prover uses Boogie and Z3 for verification (managed via .NET runtime)
- **Non-blocking:** `run_sui_prover` forks the agent -- the parent continues working while the fork runs the prover and analyzes results
- **Command visibility:** The fork prints the exact sui-prover command and working directory before execution
- **Output streaming:** The fork streams sui-prover output to stdout incrementally as verification runs
- **Interruptible:** Press Ctrl-C to interrupt prover execution at any time

## Fork behavior

`run_sui_prover` automatically forks the agent via `fork`. The fork:
1. Runs the prover (blocking, up to `timeout` seconds)
2. Analyzes the full output into a compact structured report
3. Returns the report via `final_result` — one line per spec with all failure details

The parent agent continues working immediately and receives a `[FORK DONE]` notification when the fork completes with the compact analysis.

**IMPORTANT: Do not call `run_sui_prover` or `fork` from inside a fork.** Fork depth is capped at 1. If you are already running as a fork (the prompt starts with `[FORK CONTEXT]`), use `_run_sui_prover_sync` directly instead.

## Troubleshooting

**Unexpected argument errors:**
```
error: unexpected argument '--some-flag' found
```

If you encounter this, your sui-prover version may not support that flag. Check available flags:
```bash
sui-prover --help
```

Then update your `args` parameter to only use supported flags. The default args (`["--timeout", "60", "--force-timeout", "--skip-spec-no-abort"]`) work with current sui-prover versions.

**Exit code 2 with no clear error:**
This usually means verification failed but output may not clearly indicate why. Try adding `--verbose` to the args for more detailed output.

**Cloud config not set / missing url / missing key:**
The cloud prover needs `~/.asymptotic/sui_prover.toml` with `url`, `key`, and `concurrency`. If verification fails because this file is missing or incomplete, create it and re-run — see [Cloud Mode → Cloud Configuration](#cloud-configuration). Do **not** react to this error by dropping `--cloud` / switching to local mode; the fix is to write the config, not abandon cloud verification.

**Known false positive: postconditions on shared objects not returned to caller.**
When a spec asserts postconditions on a shared object (e.g., a Pool passed by `&mut` reference) that is not returned by the function, the prover may report a postcondition failure even though the code is correct. This is a Boogie axiom limitation — the prover cannot track mutations to shared objects that remain in the caller's scope. Ignore them.


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

# Spec Bitwise Encoding

The Sui Prover ships two SMT encodings of Move integers, and several Move idioms are unprovable
in the default one. This guide explains how to recognize those idioms, when to escape the default
encoding, and how to author the escape.

## TL;DR — decision tree

```
Target uses ^, &, |, <<, arithmetic >> ?
├── No → ignore this skill, write a normal ensures
└── Yes
    ├── It's a small/wide constant pattern (xor x MAX, or x with sign-bit,
    │   and x with LO_MASK / HI_MASK, shr x by W or W-1) ?
    │   └── Add an axiom to specs/prelude_extra.bpl, prove under integer encoding.
    └── It's a loop-carry adder (sum = a^b; carry = (a&b)<<1; while (carry)...)
        or an arithmetic-shift (`shr` that sign-extends) ?
        └── Move the spec into a sibling specs-bv/ Move package, prove with
            --no-bv-int-encoding (bitvector encoding).
```

## The two encodings

| Encoding | Flag | What it can prove | What it can't |
|---|---|---|---|
| **Integer** (default) | (no flag) | Linear arithmetic, `+`, `-`, `*`, `div`, `mod`. Custom Boogie axioms can teach it specific bitwise patterns. | Free bitwise operators on wide types — quantified reasoning over `^`, `&`, `|` blows up SMT. |
| **Bitvector** | `sui-prover --no-bv-int-encoding` | Native `bvW` operators: `$Xor'BvW'`, `$And'BvW'`, `$Shl'BvW'`, `$Shr'BvW'`, `$AShr'BvW'`. Loop-carry adders, sign-extending shifts, anything pure-bitwise. | Cross-encoding mixing: a bitvector spec cannot rely on integer-encoded library lemmas in the same package. |

You **cannot mix encodings in one Move package** — the flag is per-`sui-prover` invocation.
That's why the canonical pattern uses **two sibling packages**.

## Pattern 1 — Axiom prelude (preferred, cheap)

If the target only touches a small number of fixed bitwise constants, teach the integer encoding
about them via `specs/prelude_extra.bpl`. The file is auto-loaded from the package root by
`sui-prover`; no Move.toml entry is needed.

### Axiom recipes

```boogie
// One's complement: xor with all-ones equals max-minus-x. Width W ∈ {8,32,64,128}.
axiom (forall x: int :: {$xorInt'uW'(x, $MAX_UW)}
    $xorInt'uW'(x, $MAX_UW) == $MAX_UW - x
);

// Set sign bit: or with 2^(W-1).
const $POW_TWO_HI: int;
axiom $POW_TWO_HI == <2^(W-1)>;
axiom (forall x: int :: {$orInt'uW'(x, $POW_TWO_HI)}
    $orInt'uW'(x, $POW_TWO_HI) == if x < $POW_TWO_HI then x + $POW_TWO_HI else x
);

// Low-W mask = mod 2^W.
const $LO_MASK: int;
axiom $LO_MASK == <2^W - 1>;
const $TWO_POW_W: int;
axiom $TWO_POW_W == <2^W>;
axiom (forall x: int :: {$andInt'u2W'(x, $LO_MASK)}
    $andInt'u2W'(x, $LO_MASK) == x mod $TWO_POW_W
);

// High-W mask = (x div 2^W) * 2^W.
const $HI_MASK: int;
axiom $HI_MASK == <2^(2W) - 2^W>;
axiom (forall x: int :: {$andInt'u2W'(x, $HI_MASK)}
    $andInt'u2W'(x, $HI_MASK) == (x div $TWO_POW_W) * $TWO_POW_W
);

// Shift right by W bits = division by 2^W.
axiom (forall x: int :: {$shr(x, W)}
    $shr(x, W) == x div $TWO_POW_W
);

// u8 AND truth table — needed for sign-bit / overflow-flag computations that
// AND multiple bytes. Enumerate the small set of constants you actually use.
axiom ($andInt'u8'(0, 0) == 0);
axiom ($andInt'u8'(0, 1) == 0);
axiom ($andInt'u8'(1, 0) == 0);
axiom ($andInt'u8'(1, 1) == 1);
// ... extend to whichever constants the target uses
```

The `{...}` braces inside `forall` are SMT triggers — they tell Z3 *when* to instantiate the
axiom. Triggers must mention every quantified variable. The pattern above is the standard
"match on the LHS of the equation."

### When the prelude is enough

- `i128::u128_neg(v) == MAX_U128 - v` — proves via the XOR axiom.
- `i*::sign(v)` — proves via the SHR axiom (right-shift by W-1).
- `math_u128::lo(n) == n mod 2^64` and `hi(n) == n div 2^64` — proves via AND-mask axioms.
- `math_u128::from_lo_hi` — proves via the same axioms in the inverse direction.

### When the prelude is NOT enough

If the function performs *quantified* bitwise reasoning — typically:
- A `while` loop with `^`/`&`/`<<` in the body (loop-carry add or shift-and-add multiply)
- An arithmetic `shr` (sign extension via a runtime-computed mask)

…then no finite axiom set covers it. Move to Pattern 2.

## Pattern 2 — Sibling specs-bv/ Move package

This is the escape hatch. Build a second Move package that compiles with bitvector encoding,
contains only the specs that need it, and **does not depend on the main library**.

### Package layout

```
<main_pkg>/
├── Move.toml                            (the production library)
├── sources/...                          (the production sources)
├── specs/                               (primary specs — integer encoding)
│   ├── Move.toml                        (depends on the main package)
│   ├── prelude_extra.bpl                (axioms for Pattern 1)
│   └── sources/*_specs.move
└── specs-bv/                            (secondary specs — bitvector encoding)
    ├── Move.toml                        (NO [dependencies])
    ├── prelude_extra.bpl                (native-fun shims)
    └── sources/*.move                   (struct + function + spec, all inlined)
```

### specs-bv/Move.toml

```toml
[package]
name = "<MainPackageName>SpecsBV"
edition = "2024.beta"

[addresses]
<main_package>_specs_bv = "0x0"
```

Note: **no `[dependencies]` block**. The bitvector package re-declares the struct and function
locally. This is what lets `--no-bv-int-encoding` apply only to the spec we care about, without
inflicting bitvector encoding on the whole protocol.

### specs-bv/sources/<module>.move skeleton

```move
module <main_package>_specs_bv::<module>;

#[spec_only]
use prover::prover::{ensures, asserts, invariant};

public struct I32 has copy, drop, store { bits: u32 }

// Verbatim copy of the production function — same body, same name.
public fun wrapping_add(num1: I32, num2: I32): I32 {
    let mut sum = num1.bits ^ num2.bits;
    let mut carry = (num1.bits & num2.bits) << 1;
    invariant!(|| {                                            // <-- loop invariant
        ensures(
            ((num1.bits as u64) + (num2.bits as u64)) % (1 << 32)
            == ((sum as u64) + (carry as u64)) % (1 << 32),
        );
    });
    while (carry != 0) {
        let a = sum;
        let b = carry;
        sum = a ^ b;
        carry = (a & b) << 1;
    };
    I32 { bits: sum }
}

#[spec(prove, target = wrapping_add)]
public fun wrapping_add_spec(num1: I32, num2: I32): I32 {
    let result = wrapping_add(num1, num2);
    ensures(result.bits == (((num1.bits as u64) + (num2.bits as u64)) % (1 << 32)) as u32);
    result
}
```

Run with:

```bash
cd <main_pkg>/specs-bv
sui-prover --no-bv-int-encoding
```

### CI wiring

```yaml
- run: cd specs && sui-prover                           # integer encoding (+ prelude axioms)
- run: cd specs-bv && sui-prover --no-bv-int-encoding   # bitvector encoding
```

## Pattern 3 — Loop-carry-add invariant

Kogge-Stone / parallel-prefix adders show up across the i32 / i64 / i128 modules:

```move
let mut sum = a ^ b;
let mut carry = (a & b) << 1;
while (carry != 0) {
    let a' = sum;
    let b' = carry;
    sum = a' ^ b';
    carry = (a' & b') << 1;
}
return sum;
```

The mathematical invariant: **`(sum + carry) mod 2^W == (a + b) mod 2^W`** at every iteration.
Width-widen by one type (`uW → u(2W)`) so the `+` inside the predicate doesn't itself overflow.

Place the invariant *before* the `while`:

```move
invariant!(|| {
    ensures(
        ((a as u_2W) + (b as u_2W)) % (1 << W)
        == ((sum as u_2W) + (carry as u_2W)) % (1 << W),
    );
});
while (carry != 0) { ... }
```

The `invariant!` macro carries the predicate through every iteration; the prover then discharges
the final `ensures(result.bits == ((a + b) % 2^W) as uW)` from "loop exits when carry == 0, so
sum + 0 == a + b mod 2^W."

### Width table

| Target | u_W (operand) | u_2W (invariant widening) | Modulus |
|---|---|---|---|
| `i32::wrapping_add` | u32 | u64 | `1 << 32` |
| `i64::wrapping_add` | u64 | u128 | `1 << 64` |
| `i128::wrapping_add` | u128 | u256 | `1 << 128` |

For `i128` the widening reaches `u256`, which is the upper bound on Move integer widths. Above
that, you need bitvector encoding for the *invariant* itself — but this case does not arise in
the standard integer library.

## Pattern 4 — `native fun` ↔ Boogie procedure binding

Operations with no Move-side implementation (typical for arithmetic shift right) are exposed as
`native fun` and bound to a Boogie procedure in `prelude_extra.bpl`.

### Move side

```move
public native fun ashr(x: u32, y: u32): u32;
```

No body. The function is never called by production code — it only appears in spec ensures:

```move
#[spec(prove, target = shr)]
public fun shr_spec(v: I32, shift: u8): I32 {
    asserts(shift < 32);
    let result = shr(v, shift);
    ensures(result.bits == ashr(v.bits, shift as u32));
    result
}
```

### Boogie side (`specs-bv/prelude_extra.bpl`)

```boogie
procedure {:inline 1} $0_i32_ashr($t0: bv32, $t1: bv32) returns ($ret0: bv32) {
  $ret0 := $AShr'Bv32'($t0, $t1);
}

procedure {:inline 1} $0_i64_ashr($t0: bv64, $t1: bv64) returns ($ret0: bv64) {
  $ret0 := $AShr'Bv64'($t0, $t1);
}

procedure {:inline 1} $0_i128_ashr($t0: bv128, $t1: bv128) returns ($ret0: bv128) {
  $ret0 := $AShr'Bv128'($t0, $t1);
}
```

### Symbol naming rules

`$<address>_<module>_<func>` where:
- `<address>` is the numeric address (typically `0` for `0x0`).
- `<module>` is the Move module name (lowercase, underscored).
- `<func>` is the Move function name.

Bitvector type signatures: `bvW` (e.g. `bv32`, `bv64`, `bv128`) for arguments and return.
Return values are written to `$ret0`, `$ret1`, … via assignment.

### Available SMT bitvector operators

| SMT-LIB / Boogie symbol | Meaning |
|---|---|
| `$Add'BvW'` | bitvector add (wrapping) |
| `$Sub'BvW'` | bitvector sub (wrapping) |
| `$Mul'BvW'` | bitvector mul (wrapping) |
| `$Xor'BvW'` | bitvector xor |
| `$And'BvW'` | bitvector and |
| `$Or'BvW'` | bitvector or |
| `$Not'BvW'` | bitvector not |
| `$Shl'BvW'` | logical shift left |
| `$Shr'BvW'` | logical shift right |
| `$AShr'BvW'` | arithmetic shift right (sign-extending) |

Substitute `W ∈ {8, 16, 32, 64, 128, 256}` as needed.

## How to choose between Pattern 1 and Pattern 2

```
Q: Does the target's bitwise op appear in a non-quantified, finite-constant context?
   (e.g. `n & 0xff...`, `x ^ MAX`, `n >> 64`, `n | (1 << 31)`)

   YES → Pattern 1: add an axiom to specs/prelude_extra.bpl, leave the spec in specs/.

   NO  → Pattern 2: scaffold a sibling specs-bv/ package and re-prove there.
```

The "is this finite-constant" test is the cheapest diagnostic. Loops, shifts by a variable, and
sign-extending masks built at runtime all fail the test and need bitvector encoding.

## Integer library checklist

When proving an `integer_library`-style project (i32, i64, i128, math_u64, math_u128, math_u256,
full_math_*), the canonical end-state has:

- `specs/` package depending on the main library, integer encoding + axiom prelude.
- `specs-bv/` package, no dependency, bitvector encoding for:
  - Every `wrapping_add` (uses XOR/AND loop-carry adder)
  - Every `shr` on signed types (uses sign-extending mask)
- `specs/prelude_extra.bpl` covering: xor-with-MAX, or-with-sign-bit, and-with-LO/HI mask,
  shr-by-W, and the u8 AND truth table.
- `specs-bv/prelude_extra.bpl` covering `<addr>_<module>_ashr` shims per width.
- CI runs both packages back-to-back.

If a spec drops `prove` or attaches `skip = b"axiomatic"` for any of `wrapping_add` / `shr` /
`u*_neg`, it means the author hit this skill's territory and didn't apply it. Reach for the
pattern explicitly.

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


# Spec Improvement Agent

You are a senior formal verification engineer improving a spec for a single Sui Move function using sui-prover. You receive a spec that may be in any state — from an empty stub to a nearly complete spec — and your job is to improve it as far as possible.

## Context Variables

You receive these context variables (already in your namespace):

- `spec_key` — The spec function's qualified name (e.g., `borrow_specs::borrow_spec`)
- `target_key` — The target function's qualified name (e.g., `borrow::borrow`), or None for spec_only
- `starting_state` — One of: `stub`, `asserts_only`, `semantics`, `complete`
- `spec_source` — Current spec source code
- `target_source` — Target function source code (or None for spec_only)
- `writeup` — Verification plan (YAML) from spec_writeup
- `callees` — Direct callees with source
- `structs` — Relevant struct definitions
- `all_accesses` — Struct field reads/writes across the callee graph
- `project_path` — Absolute path to the project root
- `spec_file_path` — Absolute path to the spec file
- `specs_package_path` — Absolute path to the specs package (for prover)

## Spec File Conventions

### Naming

Spec functions mirror implementation with `_spec` suffix and same signature:
```move
// For function: fun my_function(a: u64, b: u64): u64
#[spec(prove, target=module::my_function)]
fun my_function_spec(a: u64, b: u64): u64 {
    // asserts go here (before the call)
    let result = module::my_function(a, b);  // call target ONCE
    // ensures go here (after the call)
    result
}
```

**Important:** Call the target function exactly ONCE in the spec. All `asserts` and `requires` go before the call, all `ensures` go after.

### Imports

```move
#[spec_only]
use prover::prover::{ensures, asserts, requires};
```

## Workflow

### 1. Assess Current State

Read `starting_state` and `spec_source` to understand where you are:

| State | Meaning | What to do |
|-------|---------|------------|
| `stub` | Empty scaffold with `ignore_abort` | Add asserts for all abort paths, remove `ignore_abort` |
| `asserts_only` | Has asserts, no ensures | Add postconditions (ensures) |
| `semantics` | Has asserts and ensures | Review completeness, fill gaps |
| `complete` | Fully specified | Validate, look for edge cases |

### 2. Understand the Function

Use the provided context (`target_source`, `writeup`, `callees`, `structs`, `all_accesses`) to understand:
- All abort paths (overflow, division, table access, assertions)
- State mutations (what fields change and how)
- Return values and their relationship to inputs
- Loop structures requiring invariants

### 3. Write Improvements

Apply changes based on assessment:

**For stubs → asserts:**
- Trace every abort path through the function and its callees
- Add `asserts()` for each abort condition BEFORE the function call
- Handle early return guards with conditional asserts
- Remove `ignore_abort` from the spec attribute

**For asserts_only → semantics:**
- Add `ensures()` after the function call for:
  - Return value correctness
  - State mutations (use `clone!` for old state comparison)
  - Unchanged fields (frame conditions)
  - Collection preservation (unchanged elements)
- **Prefer concrete arithmetic over inequalities.** Write `ensures(balance_after == balance_before + deposit_amount)` not just `ensures(balance_after >= balance_before)`. When exact computation is known, use `==`. You can add BOTH: the strict equality AND a weaker bound (e.g., `ensures(x == a + b); ensures(x >= a);`) but the strict check must come first.
- **Ensure unchanged state for critical fields.** Don't only check what changed — explicitly ensure that important fields that should NOT change are preserved: `ensures(market.owner() == old_market.owner())`. This catches unintended side effects.
- **Use `requires` with `ignore_abort` specs.** When a spec uses `ignore_abort`, you may need `requires()` to exclude inputs where the function does an early return, a table key doesn't exist, or the function would trivially abort. Without these, ensures can fail on unreachable paths.
- **Use quantifiers for collections.** When checking properties over vectors or tables, use `forall!`/`exists!` inside ensures rather than checking elements one by one: `ensures(forall!<u64>(|i| vector_property(i, &v)))`.

**For semantics → complete:**
- Review coverage against the writeup
- Add missing edge cases
- Strengthen weak ensures (replace inequalities with exact arithmetic where possible)

### 4. Run the Prover

**ALWAYS run the prover after making changes.** Use `run_sui_prover`:

```python
from foxy.skills.sui_prover.api import run_sui_prover

result = await run_sui_prover(
    specs_package_path,
    args=["--functions", spec_key.split("::")[1], "--timeout", "60", "--force-timeout", "--skip-spec-no-abort"],
)
```

Iterate on failures:
- `_Check` failures → add more asserts/requires
- `_Assume` failures → assert condition is wrong or should be `requires`
- `_SpecNoAbortCheck` failures → spec arithmetic needs guarding
- Timeouts → add `uninterpreted` for complex callees, or `boogie_opt`
- **Bitwise-dominated function timing out / counter-examples on a bitwise op** → consider in this order:
   1. **`specs/prelude_extra.bpl` axioms first.** For bitmask wrap-around (`(x & MASK) as uN`), sign extraction (`(v.bits >> (W-1)) as u8`), `neg_from`'s OR-with-sign-bit, and the branchless signed-overflow detector in `iN::add` / `sub`, the right move is to drop a project-wide `specs/prelude_extra.bpl` with the canonical XOR / OR / AND / shr / u8-AND truth-table axiom set — see `sui_prover_guide` → "Canonical `prelude_extra.bpl` for two's-complement / bitmask code" for the verbatim file. This covers ~80% of bitwise constructs in the default int encoding. Pair with `Integer` + `.to_signed_int()` + `.is_iN()` (`sui_prover_guide` → "Two's-complement signed integers in the int encoding") to express signed specs cleanly.
   2. **`boogie_opt` tuning.** If a spec drives the prelude axioms but Z3 still times out, raise `EAGER_THRESHOLD` (`100` → `200`) and/or add `vcsSplitOnEveryAssert`. Hot recipes are in `sui_prover_guide` → "Recipes for the patterns the prelude axioms drive".
   3. **Bitvector encoding (`specs-bv/`) as a fallback.** Only for the cases the axioms can't reach: ripple-carry adders (`while (carry != 0) { sum ^= carry; carry = (a & b) << 1; }`) and arithmetic right shift on `Ixx`. Redeclare the type + function locally, attach a loop invariant relating `(a + b) mod 2^w` to `(sum + carry) mod 2^w`, and back `shr` with `public native fun ashr(...)` + a `procedure {:inline 1} $0_<module>_ashr` in `specs-bv/prelude_extra.bpl`. In the main `specs/` spec, use `#[spec(target = ...)]` without `prove` so the int-encoded run skips it; the BV package discharges it. See `sui_prover_guide` → "Bitvector encoding (`--no-bv-int-encoding`)" for the full playbook.

**Act on bitwise symptoms autonomously through the three-step ladder — do not ask first.** When you see `_Check` failing with "code should not abort" on a `&` / `|` / `<<` / `>>`, an `_Assume` counter-example where the prover assigns nonsense to a bitwise-mask result, a ripple-carry adder body, or `(x >> (w-1)) as u8` failing on a wide type, do not stop and ask. Walk the ladder:
1. Drop in `specs/prelude_extra.bpl` with the canonical bitwise axiom set (the verbatim file in `sui_prover_guide`).
2. Add the `boogie_opt` recipe that matches the spec shape (EAGER_THRESHOLD=100 for mask / shift ensures, `vcsSplitOnEveryAssert vcsFinalAssertTimeout:300` for chain-arithmetic).
3. If after steps 1-2 the spec is still a ripple-carry adder or signed arithmetic right shift, set up `specs-bv/` for that one function and prove it there.

The user has already opted into "prove this project"; choosing among prelude axioms, boogie_opt tuning, and BV encoding is tool selection inside an authorized task, not a separate scope question. Only stop to ask when (a) the BV port itself fails after a real attempt with a correctly placed loop invariant, or (b) the work needs a `prelude_extra.bpl` Boogie glue not covered by the canonical set (e.g. domain-specific axioms for a CLMM math helper).

### 5. Update the @VERIFY Annotation

After making changes, update the `// @VERIFY(stage/status)` annotation above the spec to reflect the new state.

For example, after adding ensures and prover passes: `// @VERIFY(⚙️/✅)`

### 6. Return the Work Report

When done (prover passes, or you've exhausted your iteration budget), call `final_result()` with a work report dict:

```python
final_result({
    "starting_state": starting_state,
    "final_state": "semantics",  # stub | asserts_only | semantics | complete
    "changes_made": [
        "Added 3 asserts for overflow checks",
        "Added 2 ensures for balance mutations",
        "Removed ignore_abort",
    ],
    "prover_runs": 4,
    "prover_final": "pass",  # pass | fail | timeout
    "issues": [
        "Nested table access in ensures causes timeout — left as asserts_only for that property"
    ],
    "duration_seconds": 45,
})
```

## Rules

### Rule 1: NEVER add `ignore_abort`; NEVER remove it without verified abort coverage
`ignore_abort` skips abort coverage — only the user can add it.

**Removing `ignore_abort`** is also a commitment: the prover must verify the
spec function never aborts (the `_SpecNoAbortCheck` phase). Before removing it:

1. Trace every abort path in the target and its callees:
   - Explicit `assert!(...)` calls
   - Division / modulo (abort on zero divisor)
   - Narrowing casts (`as uN`) on values that may exceed `MAX_UN`
   - Table / dynamic-field access (abort on missing key)
   - Vector indexing (abort on out-of-bounds)
   - Arithmetic overflow on the implementation's intermediate types

2. Add `asserts()` covering each path **before** the call to the target.

3. **Verify with the abort-check phase enabled.** Drop `--skip-spec-no-abort`
   for at least one prover run on this spec — that is the only run whose
   "pass" actually confirms abort-freedom. A pass with `--skip-spec-no-abort`
   is silent on the obligation you just took on.

**Anti-pattern that bit real specs:** removing `ignore_abort`, adding zero
asserts ("the function never aborts so no asserts needed"), iterating with
`--skip-spec-no-abort`, declaring success. The full-package run with the flag
absent then fails `_SpecNoAbortCheck`. If you really believe the function
never aborts, prove it: leave the asserts empty AND run the check.

### Rule 2: NEVER leave the spec worse than you found it
If you can't make progress, return the spec to its original state and report the issue.

### Rule 3: Always read the file before editing
Use `read()` from file_ops before any `edit()` call — never edit from memory.

### Rule 4: No comments in specs
Don't add comments to spec code.

### Rule 5: `no_opaque` for same-module caller/callee specs
`no_opaque` controls how OTHER specs use THIS spec. Use when caller spec needs to see inside callee's implementation (both public in same file).

**Do NOT use `include=` for `no_opaque` specs** — `include=` brings in opaque contracts, but `no_opaque` specs are inlined by the prover anyway, making `include=` useless.

### Rule 6: Module-level `include=` for cross-module specs
```move
#[spec_only(
    include = specs::market_specs,
    include = specs::reserve_specs,
)]
module specs::borrow_specs;
```

Without this, functions from other spec modules are inlined causing timeouts.

### Rule 7: Use `uninterpreted` for scalability
When specs timeout, mark complex callees as `uninterpreted`.

### Rule 8: Constants must be duplicated
Constants from implementation modules must be duplicated in spec files — there is no import mechanism. Keep them in sync manually with the implementation.

### Rule 9: Method syntax rules
Method syntax works only when function is defined in the same module as receiver type:
- `bag::contains(bag, key)` → `bag.contains(key)` works
- `dynamic_field::borrow(uid, key)` → cannot use method syntax

### Rule 10: Two-pass debugging
First run with `--no-counterexample-trace` for compact errors, then without for detailed traces.

```bash
# Pass 1: Quick overview
sui-prover --functions my_spec --force-timeout --no-counterexample-trace

# Pass 2: Detailed trace
sui-prover --functions my_spec --force-timeout
```

### Rule 11: Public spec functions for cross-module reuse
When spec functions or pure helpers are referenced from other spec modules (via `include=` or `uninterpreted=`), declare them `public`.

### Rule 12: `include=` must match current implementation
Opaque spec's `asserts` become caller preconditions and `ensures` become postconditions. Only include specs that match the current implementation state.

### Rule 13: Workflow ordering for large functions
Write specs for all big/complex called functions BEFORE writing the caller spec.

### Rule 14: Prover visibility
The prover has access to private/package functions and structs by default.

### Rule 15: Stay in `to_int()` arithmetic, drop to bits only when forced
The prover's integer theory (over `Int`, via `to_int()`) is linear-arithmetic
backed and fast. Bitwise reasoning (`x & MASK`, `x ^ MASK`, `x << k`) requires
bit-blasting axioms or extra `extra_bpl` files; the solver explores a much
larger search space and easily times out.

**Prefer:**
```move
let product_int = n1.to_int().mul(n2.to_int());
ensures(result == product_int.to_u128());
ensures(overflow == product_int.gt(std::u128::max_value!().to_int()));
```

**Avoid (real case that timed out at 600s vs. 2s):**
```move
let r = (n1 as u256) * (n2 as u256);
let hi_mask: u256 = 0xff..ff00..00;
let lo_mask: u256 = 0x00..00ff..ff;
asserts((r & hi_mask) >> 128 <= max_u128_u256);
asserts(r & lo_mask <= max_u128_u256);
ensures(overflow == ((r & hi_mask) >> 128 > 0));
ensures((result as u256) == r & lo_mask);
```

Use bitwise encodings only when the implementation does something genuinely
bit-level (e.g., `i32` two's-complement sign bit, XOR-based negation, ripple-
carry loops). Even then, a custom `extra_bpl` is usually needed; do not assume
the default theory will close the goal.

### Rule 16: One conditional ensures beats N independent ensures
A single `if (cond) ensures(...) else ensures(...)` directly characterises the
result from the inputs. N independent `ensures` force the solver to combine N
lemmas to discharge the same obligation — fine for `Check`/`Assume`, brittle
on `_SpecNoAbortCheck` and on large goals.

**Prefer (proves in 2s):**
```move
ensures(if (num1 > num2) { result == num1 } else { result == num2 });
```

**Avoid (real case that timed out at 600s on the abort-check phase):**
```move
ensures(result >= num1);
ensures(result >= num2);
ensures(result == num1 || result == num2);
```

The same pattern bites `min`, `clamp`, `wrapping_mul`, `overflowing_mul`, and
any spec where the implementation has a clear control-flow split. Mirror the
control flow in the ensures.

### Rule 17: Auxiliary `.bpl` files must be spec-pinned, never shared-mutable
When you reach for `extra_bpl=b"../my_axioms.bpl"`, treat that file as **owned
by this spec**. Do not share one `.bpl` across multiple specs that other
agents may edit in parallel — concurrent appends easily produce:

- Two axioms with the same SMT trigger but different right-hand sides
  (`{$andInt'u128'(x, MAX)}` → bound, then equation). Matching loops and
  quantifier instantiation order become solver-dependent and flaky.
- References to constants the other agent did not define (`$POW_2_64` vs.
  `$POW_TWO_31`), silently dropping the axiom.
- Race conditions: agent A verifies against version N of the `.bpl`, agent B
  appends version N+1, the final batch run uses N+1, A's spec regresses.

**Practical rule:** name `.bpl` files after the spec (or the module) that owns
them — `math_u64_overflow_axioms.bpl`, not `bitwise_axioms.bpl`. If multiple
specs really need the same axiom, factor it into the project's curated
`prelude_extra.bpl` once, with the user's review, and pin it via repository
state — not via parallel agent edits.


### Rule 18: One agent per spec module at a time — never edit the same file concurrently

`spec_improve` mutates a `.move` file (`<module>_specs.move`). When the
orchestrator runs multiple `spec.improve` calls in parallel, it MUST serialize
calls for specs in the **same module**. Two agents editing the same file race
in real-world batches we ran:

- Agent A reads the file, computes its edit, writes the result.
- Agent B reads the (stale, pre-A) file, computes its edit on the stale base,
  writes — silently overwriting A's `ensures` / `asserts`.
- The batch reports both as "success" but only B's spec actually lands.

The collection framework enforces this for the `improve` and `flip_requires`
fields via `SpecCollection._default_serialize_by`: specs in the same module
run as one sequential chain, while different modules still run in parallel.
The agent must honor the same rule when scheduling improvements manually:

- **Safe**: 8 modules × 1 agent each = 8 parallel chains, each chain serial inside.
- **Unsafe**: 4 agents on `math_u128_specs::*` concurrently — pick one at a time.

If you bypass `submit_batch` (e.g. raw `run_agent` calls or a custom executor),
group your work-list by `key.split("::")[0]` and feed one group sequentially.
Reach for `submit_batch(serialize_by=False)` only when you explicitly know the
field does not mutate the spec file.

### Rule 19: Don't ship `asserts_only` when the function has observable semantics

`final_state: asserts_only` is a partial result, not a finish line. It's the
right answer only when:

1. The function is a void-return state mutator with no observable return
   value or field-level invariant worth expressing, **AND**
2. You attempted at least one `ensures` and it failed (timeout, counterexample
   that can't be eliminated by the int → bitwise ladder), **AND**
3. The `issues` list in your work report documents *what you tried* and
   *why it didn't land* — not just "added asserts, didn't add ensures".

If the function has any of these signals, `asserts_only` is wrong by default:

- Returns a value derived from inputs (`min`, `max`, `cmp`, `eq`, `lt`, `gt`,
  `lte`, `gte`, `sign`, `is_neg`, `abs`, `from`, `as_*`, `wrapping_*`,
  `overflowing_*`, getters, predicates).
- Sets a struct field whose new value is a known function of inputs.
- Has a clear arithmetic identity (`wrapping_sub`, `mul_div_floor`, `shrw`).

For these, write the `ensures` on the **first pass**. The orchestrator's
status-check protocol (below) will requeue any spec that lands at
`asserts_only` for one of these function shapes — save the round trip and
land the ensures the first time.

A practical heuristic: when you've removed `ignore_abort` and the spec
passes, ask "given an input, can I write `result == f(input)` for any `f`?"
If yes, you owe an `ensures`. If no — the function genuinely is just a state
mutator with no derivable return contract — say so in `issues`.

### Rule 20: Status-check protocol — verify ensures coverage before reporting "done"

When the user asks "status?" or you're about to report a batch as finished,
do NOT just count `prover_final == pass`. Also count `final_state`:

- Sum specs by `final_state`: `stub`, `asserts_only`, `semantics`, `complete`.
- Cross-reference function shape (return type, mutation pattern) against
  Rule 19. Any spec at `asserts_only` whose target has a derivable return
  value or known field update is a candidate for **automatic re-improvement**,
  not a finished result.
- Surface the breakdown in the status line: not just "30/31 passed", but
  "30/31 passed; of those, 8 are `asserts_only` and 5 of those 8 have
  derivable semantics — re-improving."

After the batch you control finishes, sweep the `asserts_only` set, requeue
the ones that fail the Rule 19 check, and only then report "done".

## Spec Attributes Reference

| Attribute | Usage |
|-----------|-------|
| `#[spec(prove, target=module::func)]` | Standard spec targeting a function |
| `#[spec(prove, focus)]` | Run only this spec (debugging) |
| `#[spec(prove, no_opaque)]` | Prevents OTHER specs from using this spec as opaque contract |
| `#[spec(prove, ignore_abort)]` | **RESTRICTED** — only user can add |
| `#[spec(prove, uninterpreted=mod::fn)]` | Prevent solver from expanding function definition (scalability) |
| `#[spec(prove, include=specs::other_specs)]` | Bring in cross-module opaque specs |
| `#[spec(prove, boogie_opt=b"vcsSplitOnEveryAssert")]` | Boogie options |
| `#[spec(prove, run_on=b"local")]` | Force this spec to run on the local prover instead of the cloud backend |
| `#[spec(target=module::func)]` (no `prove`) | Axiom spec — declares contract without proving it |
| `#[spec_only(include=specs::mod)]` on module | Module-level include for all specs in the module |

### Remote vs local execution

Specs run on the **cloud prover** by default — this is what `sui-prover --cloud` drives and what the cron pipeline uses. Add `run_on = b"local"` to a single `#[spec(...)]` annotation only when the spec genuinely cannot run in the cloud (e.g., depends on local-only `extra_bpl` or `boogie_opt` paths, or needs to be iterated interactively on your machine). When in doubt, leave it out and let the spec run in the cloud.

## @VERIFY Annotations

Use `// @VERIFY(stage/status)` annotations above each spec. Always use the two-symbol format.

**Stages** (what level of verification):
| Symbol | Name | Meaning |
|--------|------|---------|
| 🫙 | STUB | Empty spec stub that passes prover (with ignore_abort) |
| 🛡️ | ASSERTS | Abort conditions fully modeled (passes WITHOUT ignore_abort) |
| ⚙️ | SEMANTICS | Function fully modeled (asserts AND ensures) |
| 🔍 | REVIEW | Reviewed by third party / ready to ship |

**Status** (current state):
| Symbol | Name | Meaning |
|--------|------|---------|
| 🌀 | WIP | Currently working on this |
| ⚠️ | Issue | Made an attempt, there's an issue |
| ✅ | Done | Completed |

**Examples:**
- `// @VERIFY(🫙/✅)` — stub completed
- `// @VERIFY(🛡️/✅)` — abort conditions fully modeled and passing
- `// @VERIFY(⚙️/🌀)` — working on full semantics
- `// @VERIFY(🛡️/⚠️)` — issue with abort condition modeling
- `// @VERIFY(⚙️/✅)` — fully proved with semantics

## `uninterpreted` for Scalability

**This is the primary scalability mechanism.** When specs timeout, mark functions as `uninterpreted` to prevent the solver from expanding their definitions:

```move
#[spec(
    prove,
    target = module::complex_function,
    uninterpreted = specs::helper_specs::quantifier_predicate,
    uninterpreted = std::q32::mul,
    boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms",
)]
```

**CRITICAL: `uninterpreted` can only be applied to pure functions** (functions with the `ext(pure)` attribute). Marking a non-pure function as `uninterpreted` will cause compile error.

Use `uninterpreted` for:
- Complex math functions (fixed-point arithmetic, sqrt calculations)
- Pure helper functions used in quantifiers
- Functions whose implementation details don't matter for the current spec

### Design Pattern: Named Wrappers + `uninterpreted`

For complex quantified expressions, create named pure predicates and mark them `uninterpreted` at callsites:

```move
#[spec_only, ext(pure)]
public fun all_valid_collaterals(types: &vector<TypeName>, obligation: &Obligation): bool {
    all!<TypeName>(types, |ct| is_valid_collateral(ct, obligation))
}

#[spec_only, ext(pure)]
public fun collateral_value_sum(types: &vector<TypeName>, ...): std::integer::Integer {
    sum_map!<TypeName, std::integer::Integer>(types, |ct| single_value(ct, ...))
}
```

Then mark them `uninterpreted` in the caller spec:
```move
#[spec(prove, target = evaluator::check_health,
    uninterpreted = specs::collateral_value_specs::all_valid_collaterals,
    uninterpreted = specs::collateral_value_specs::collateral_value_sum,
    uninterpreted = specs::debt_value_specs::all_valid_debts,
    uninterpreted = specs::debt_value_specs::weighted_debt_value_sum,
    uninterpreted = specs::price_specs::price_feed_to_q32,
)]
```

**Key rule:** Loop invariants use INLINE `all!`/`sum_map_range!` calls, NOT wrapper predicates — this avoids `_Assume` solver regression. Only the outer spec uses the named wrappers.

## Prover Phases

| Phase | What it verifies |
|-------|------------------|
| `_Check` | Assumes all asserts/requires hold, verifies implementation doesn't abort and ensures hold |
| `_Assume` | Verifies asserts actually correspond to abort conditions (abort coverage) |
| `_SpecNoAbortCheck` | Verifies the spec body itself doesn't abort (e.g., u64 overflow in spec-level computations) |

**Fix errors in the phase where they appear:**
- `_Check` failures → add more asserts/requires
- `_Assume` failures → the assert condition is wrong or should be `requires`
- `_SpecNoAbortCheck` failures → the spec's own arithmetic needs guarding

## Source Code Modifications for Specs

Modifying the source implementation to support formal verification is standard practice. Common patterns:

### 1. `#[ext(pure)]` on Existing Getters
Add `#[ext(pure)]` to existing public getter functions to tell the prover they are side-effect-free:
```move
#[ext(pure)]
public fun balance_sheets(vault: &Reserve): &WitTable<...> { &vault.balance_sheets }
#[ext(pure)]
public fun contains<T, K, V>(self: &WitTable<T, K, V>, k: K): bool { ... }
```

### 2. `#[test_only]` New Getters for Private Fields
Add new getter functions for struct fields that specs need to access:
```move
#[test_only, ext(pure)]
public fun get_debts(self: &Obligation): &WitTable<...> { &self.debts }
#[test_only, ext(pure)]
public fun get_amount(self: &Debt): u64 { self.amount }
#[test_only, ext(pure)]
public fun balance_sheet_cash(bs: &BalanceSheet): u64 { bs.cash }
```
The `#[test_only]` annotation makes these available to the prover (which runs in test mode) without polluting production code.

### 3. `#[spec_only]` Getters (Rare)
For getters only needed by the prover, not even by tests:
```move
#[spec_only, ext(pure)]
public fun last_update_of(entry: &MinPriceHistory): u64 { entry.last_update }
```

### 4. Stubbing Dynamic Fields
When the prover cannot handle dynamic field operations, stub them with hardcoded values:
```move
public fun is_allow_all(uid: &UID): bool {
    // df::exists_(uid, AllowAllKey {})  // original
    true  // stubbed for prover
}
```

Also stub dynamic field reads in business logic when needed:
```move
// let borrow_limit = *df::borrow<BorrowLimitKey, u64>(market::uid(market), key);
let borrow_limit = 18446744073709551615u64;  // max u64 to skip limit check
```

### 5. New Public Containment Checkers
Add `has_*` functions to support the `asserts(table.contains(key))` pattern:
```move
public fun has_decimals(registry: &CoinDecimalsRegistry, type_name: TypeName): bool {
    table::contains(&registry.table, type_name)
}
```

### 6. Deprecated Functions
Set deprecated functions to `abort 0` and spec them with `asserts(false)`:
```move
// Implementation:
public fun deprecated_function(...) { abort 0 }
// Spec:
#[spec(prove, target = module::deprecated_function)]
fun deprecated_function_spec(...) {
    asserts(false);
    module::deprecated_function(...);
}
```

## BPL Axioms

### Bit-shift semantics
```bpl
const $POW_TWO_32: int;
axiom $POW_TWO_32 == 4294967296;
axiom (forall x: int :: {$shr(x, 32)} $shr(x, 32) == x div $POW_TWO_32);
```

### Extra BPL prelude files
When prover fails with `use of undeclared function: $X_module_native_func$pure`:
```move
#[spec_only(extra_bpl = b"mymodule_prelude.bpl")]
module specs::mymodule;
```

## Project Structure

```
contracts/
├── sources/           # Implementation
├── tests/             # Unit tests
└── specs/             # Formal verification
    ├── Move.toml      # Depends on local implementation package
    └── sources/
        └── *_specs.move
```

The specs `Move.toml` should ONLY depend on the protocol sources package. Do NOT add sui, stdlib, or prover dependencies — they are injected automatically by sui-prover.

## `skip` Attribute for Cross-Package Specs

When a spec is proved in a different package (e.g., `dependencies_specs`), mark it with `skip` to avoid re-verification:
```move
#[spec(target = wit_table::add, skip = b"Proved in another package")]
fun add_spec<T: drop, K: copy + drop + store, V: store>(...) { ... }
```
The spec still serves as an axiom for callers — it just won't be verified in this package.

## WitTable `keys_accessible` Pattern

For tables with optional key tracking, create a reusable containment predicate:
```move
#[spec_only, ext(pure)]
public fun keys_accessible<T: drop, K: copy + drop + store, V: store>(
    self: &WitTable<T, K, V>,
): bool {
    let with_keys = self.get_with_keys();
    let keys = self.get_keys();
    !with_keys || keys.is_some()
}

// Usage:
asserts(keys_accessible(obligation.get_collaterals()));
```

## Failed Specs Package

When a spec reveals a **bug in the source code** (not a spec error), move the spec to a separate `failed_specs` package:

```
contracts/
├── specs/              # Main specs (all passing)
└── failed_specs/       # Specs that prove source bugs
    ├── Move.toml       # Same deps as specs, different package name
    └── sources/
```

```toml
[package]
name = "failed_specs"
edition = "2024.beta"

[dependencies]
MyProtocol = { local="../protocol" }

[addresses]
failed_specs = "0x0"
```

When to use:
- Spec correctly models the function but prover shows the function aborts on valid inputs (source bug)
- Division by zero from incorrect overflow guards (e.g., `is_safe_mul(0, y)` doing `MAX / 0`)
- Missing input validation (e.g., no check for zero amounts)
- Annotate with `// @VERIFY(🛡️/⚠️) Bug L-N: description`

### Example

```move
module failed_specs::u128_specs;

// @VERIFY(🛡️/⚠️) Bug L-9: is_safe_mul(0, y) aborts — U128_MAX / 0 is division by zero.
// 0 * y = 0 never overflows, so is_safe_mul(0, y) should return true.
#[spec(prove, target = u128::is_safe_mul, no_opaque)]
fun is_safe_mul_spec(x: u128, y: u128): bool {
    let expected_result = x.to_int().mul(y.to_int()).lte(std::u128::max_value!().to_int());
    let result = u128::is_safe_mul(x, y);
    ensures(expected_result == result);
    result
}
```

## Bitvector-encoding Sibling Package (`specs-bv`)

When a spec is dominated by bitwise / shift operations (carry-chain adders, arithmetic right shift, bit-mask packing), the default Boogie integer encoding times out — Z3 can't relate `&`/`|`/`^`/`<<`/`>>` to integer arithmetic. The fix is to prove that handful of specs under **bitvector encoding** via `sui-prover --no-bv-int-encoding`.

The flag is invocation-wide (every spec in the run is BV-encoded), so we split the BV-only specs into a sibling package:

```
project/
├── specs/              # default encoding -- the bulk of specs
└── specs-bv/           # bitvector encoding -- shift-heavy specs only
    ├── Move.toml       # `integer_library_specs_bv = "0x0"`
    ├── prelude_extra.bpl  # native ashr definitions
    └── sources/
        └── i32.move    # self-contained: re-declares source + spec
```

In the main `specs/` package, leave the BV-only spec **without `prove`** so the integer run skips it, and point at the sibling in a comment:

```move
/*
 ⚠️ Proved in a separate package as it requires a custom prover configuration.
*/
#[spec(target = wrapping_add)]                    // no `prove`
public fun wrapping_add_spec(num1: I32, num2: I32): I32 { ... }
```

In `specs-bv/sources/`, re-declare the struct and the source function locally (the BV package doesn't depend on the main package — it gives the encoder a clean view), then write the spec. For arithmetic right shift, declare a native `ashr` and back it with Boogie's `$AShr'BvN'` in `prelude_extra.bpl`:

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

CI runs both packages: `cd specs && sui-prover && cd ../specs-bv && sui-prover --no-bv-int-encoding`.

See `sui_prover_guide.md` → **Bitvector encoding (`--no-bv-int-encoding`)** for the full pattern, when-to-use criteria, and the bit-iterative wrapping-add example.


## Pragmatic `ignore_abort` for Complex Entry Functions

Complex user-facing entry functions (borrow, liquidate, mint, redeem) that call 10+ sub-functions may use `ignore_abort` when:
- Modeling all abort paths across 10+ transitive calls is impractical
- Focus is on proving meaningful postconditions (state preservation, accounting invariants)
- Sub-function specs already cover their own abort conditions independently

Pair `ignore_abort` with rich `ensures`:
```move
#[spec(
    prove,
    target = market::accrue_all_interests,
    include = specs::reserve_specs,
    include = specs::borrow_dynamics_specs,
    ignore_abort,
    boogie_opt = b"vcsSplitOnEveryAssert useArrayAxioms proverOpt:O:smt.MBQI=false",
)]
fun accrue_all_interests_spec(self: &mut Market, now: u64) {
    requires(forall!<TypeName>(|t| has_valid_revenue_factor(*t, self)));
    let old_market = clone!(self);
    market::accrue_all_interests(self, now);
    ensures(forall!<TypeName>(|t| accrue_preserves_type(*t, old_bs, new_bs)));
    ensures(self.collateral_stats() == old_market.collateral_stats());
}
```

## Axiom Specs (No `prove`)

For functions too complex to prove fully, use `#[spec(target=...)]` WITHOUT `prove`:
```move
// @VERIFY(--/⚠️, TAG=meaningful) Axiom -- prevents prover from inlining complex code
#[spec(target = module::complex_aggregation)]
fun complex_aggregation_spec(self: &mut State, now: u64) {
    let old_state = clone!(self);
    module::complex_aggregation(self, now);
    ensures(self.get_field() >= old_state.get_field());
    ensures(self.other_field() == old_state.other_field());
}
```
Use sparingly. Document why proving is infeasible. Include useful state preservation ensures so callers can rely on them.

## Proving Ensures Through Loops (Entry Point Specs)

When an entry point calls a loop function then modifies state, proving calculation ensures requires careful architecture.

### Step 1: Loop Invariant Preservation

Write `loop_inv` functions in a **separate module** (avoids same-module spec auto-apply poisoning `_Assume`). Use `forall!` wrappers:

```move
// Predicate for single element
#[spec_only, ext(pure)]
fun cash_preserved(t: &TypeName, r: &Storage, old: &Storage): bool {
    get_cash(r, t) == get_cash(old, t)
}
// Quantifier wrapper (forall! must wrap a single function call)
#[spec_only, ext(pure)]
fun all_cash_preserved(r: &Storage, old: &Storage): bool {
    forall!<TypeName>(|t| cash_preserved(t, r, old))
}
// Loop invariant
#[spec_only(loop_inv(target = module::loop_fn)), ext(pure)]
fun loop_inv(i: u64, n: u64, r: &Storage, __old_r: &Storage): bool {
    i <= n && all_cash_preserved(r, __old_r)
}
```

### Step 2: Include Loop Spec as Opaque

In the entry point spec, include the loop spec to make the loop function opaque. Its proven ensures carry to the caller:

```move
#[spec(prove, ignore_abort, no_opaque, target = entry::withdraw,
    include = specs::loop_specs::refresh_interest_spec,  // loop → opaque
    include = specs::loop_specs::is_safe_spec,           // another loop → opaque
    include = specs::other::reward_manager_spec,         // heavy function → opaque
)]
```

### Step 3: Progressive VC Reduction

If Z3 crashes, include more functions as opaque:
1. Loop functions (always include)
2. Safety checks with loops (is_obligation_safe)
3. Heavy computations (liquidity mining, ADL, limiters)
4. Leaf computations (burn_ctokens) — only if needed

## Protocol Invariant Pattern

For DeFi invariants (e.g., exchange rate >= 1.0):

1. Define `ext(pure)` predicate: `valid_exchange_rate_invar(reserve)`
2. Add `requires(invariant)` + `ensures(invariant)` to EVERY state-modifying function
3. Verify each spec passes — failed ensures may reveal real protocol bugs
4. For entry points: propagate `requires` up the call chain
5. Check counterexamples: if they show unreachable states, add auxiliary `requires` (e.g., `cash_covers_reserve`) and prove THAT is maintained too

**Analyzing failed ensures:**
- If `ensures(result.value() > 0)` fails: check for rounding-to-zero attacks
- If invariant fails with extreme values: check if state is reachable
- If unreachable: add filtering `requires`, prove all operations maintain it
- If reachable: report as protocol finding

## Known Prover Bugs

### Nested Table Access in Ensures
Specs with ensures that access fields through nested tables timeout even with `boogie_opt`. Mark as abort coverage only (no ensures).

## Examples

See `examples/` directory:

- **`01_basic_open_position.md`** — Multiple includes, abort coverage, struct field ensures
- **`02_precondition_next_sqrt_price.md`** — Complex tick math, branch-specific asserts, `requires` for domain constraints
- **`03_precondition_wit_table_add.md`** — Conditional asserts, witness pattern, `no_opaque`
- **`04_precondition_risk_model_change.md`** — Repetitive validation, fixed-point bounds, epoch overflow
- **`05_precondition_enable_manage.md`** — Dynamic field existence, boolean toggle
- **`06_postcondition_get_price.md`** — Oracle validation, fixed-point conversion in ensures
- **`07_postcondition_withdraw_fee.md`** — Ghost variables for transfer verification, balance arithmetic
- **`08_loop_is_empty.md`** — Loop invariant with quantifiers, conditional ensures
- **`09_loop_settle.md`** — Complex loop with multiple pure helpers, `clone!`, frame conditions
- **`10_loop_reserve_validation.md`** — Counter-based loop, `requires` as sole mechanism

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

Improve the spec `staking_pool_specs::sui_amount_spec` targeting `staking_pool::sui_amount`. Current state: stub. Add abort coverage (asserts) and remove ignore_abort.

## Assistant

I'll start by loading the spec_improve skill and examining the current state of the spec and target function.

### Code `toolu_01RH2LMoFXSehGsnLAUJdedC`

```python
load_skill('spec_improve')
from foxy.skills.project_env.api import get_project_root, get_packages
from foxy.skills.spec_knowledge.api import get_knowledge

project = get_project_root()
packages = get_packages()
foxy_inspect(f"project={project}")
foxy_inspect(f"packages={packages}")
```

> 145 in, 190 out, 105847 cache read

### Result `toolu_01RH2LMoFXSehGsnLAUJdedC`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_3mdyuop3.txt","to_foxy_logs":"project=/Users/cos/asymptotic/agent/clients/mysten/sui\npackages=['/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system', '/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs']"} -->
```
project=/Users/cos/asymptotic/agent/clients/mysten/sui
packages=['/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system', '/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs']
```

---

## Assistant

### Code `toolu_01DSAq5zqR21Kbki3F3XXApo`

```python
k = get_knowledge(project)
spec = k.item("staking_pool_specs::sui_amount_spec")

foxy_inspect("=== SPEC SOURCE ===")
foxy_inspect(spec.source)
foxy_inspect("=== TARGET SOURCE ===")
foxy_inspect(spec.target["source_code"] if spec.target else "None")
foxy_inspect("=== ATTRIBUTES ===")
foxy_inspect(spec.attributes)
```

> 219 in, 211 out, 105847 cache read, 335 cache create

### Result `toolu_01DSAq5zqR21Kbki3F3XXApo`

<!-- {"error":"'source_code'","type":"KeyError"} -->
```
⚠ KeyError: 'source_code'
```

---

## Assistant

### Code `toolu_013D8uU2ruz5oCVDVaKv1C5W`

```python
foxy_inspect("=== SPEC SOURCE ===")
foxy_inspect(spec.source)
foxy_inspect("=== TARGET KEYS ===")
foxy_inspect(list(spec.target.keys()) if spec.target else "None")
foxy_inspect("=== ATTRIBUTES ===")
foxy_inspect(spec.attributes)
```

> 31 in, 173 out, 106182 cache read, 430 cache create

### Result `toolu_013D8uU2ruz5oCVDVaKv1C5W`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_j24hm69d.txt","to_foxy_logs":"=== SPEC SOURCE ===\n#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]\nfun sui_amount_spec(\n    exchange_rate: &PoolTokenExchangeRate,\n): u64 {\n    staking_pool::sui_amount(exchange_rate)\n}\n=== TARGET KEYS ===\n['qualified_name', 'module', 'function_name', 'file_path', 'line_start', 'line_end', 'visibility', 'is_entry', 'function_type', 'attributes', 'source']\n=== ATTRIBUTES ===\n{'prove': True, 'skip': False, 'skip_reason': None, 'target': 'staking_pool::sui_amount', 'no_opaque': False, 'timeout': None, 'spec_only': False, 'focus': False, 'ignore_abort': True, 'uninterpreted': [], 'include': [], 'boogie_opt': [], 'extra_bpl': []}"} -->
```
=== SPEC SOURCE ===
#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}
=== TARGET KEYS ===
['qualified_name', 'module', 'function_name', 'file_path', 'line_start', 'line_end', 'visibility', 'is_entry', 'function_type', 'attributes', 'source']
=== ATTRIBUTES ===
{'prove': True, 'skip': False, 'skip_reason': None, 'target': 'staking_pool::sui_amount', 'no_opaque': False, 'timeout': None, 'spec_only': False, 'focus': False, 'ignore_abort': True, 'uninterpreted': [], 'include': [], 'boogie_opt': [], 'extra_bpl': []}
```

---

## Assistant

### Code `toolu_012hPEQakNkgDvFcemUFgNzm`

```python
foxy_inspect("=== TARGET SOURCE ===")
foxy_inspect(spec.target["source"])
foxy_inspect("=== SPEC FILE PATH ===")
foxy_inspect(spec.path("source") if hasattr(spec, "path") else "n/a")
foxy_inspect(spec.target["file_path"])
```

> 432 in, 163 out, 106612 cache read, 204 cache create

### Result `toolu_012hPEQakNkgDvFcemUFgNzm`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_63u0nf_j.txt","to_foxy_logs":"=== TARGET SOURCE ===\npublic fun sui_amount(exchange_rate: &PoolTokenExchangeRate): u64 {\n    exchange_rate.sui_amount\n}\n=== SPEC FILE PATH ===\n/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/sui_amount_spec/spec.source.move\n/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/sources/staking_pool.move"} -->
```
=== TARGET SOURCE ===
public fun sui_amount(exchange_rate: &PoolTokenExchangeRate): u64 {
    exchange_rate.sui_amount
}
=== SPEC FILE PATH ===
/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/sui_amount_spec/spec.source.move
/Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/sources/staking_pool.move
```

---

## Assistant

The target is a trivial getter — it just returns a field `exchange_rate.sui_amount`. There are no abort paths: no arithmetic, no division, no table access, no casts, no assertions. The function cannot abort. So abort coverage = empty asserts, and I can safely remove `ignore_abort`.

Let me find the spec file and view the actual function in context.

### Code `toolu_01BpGLawtxQjYE1jHQfZapgJ`

```python
matches = grep("fun sui_amount_spec", include="*.move")
foxy_inspect(matches)
```

> 314 in, 200 out, 106816 cache read, 595 cache create

### Result `toolu_01BpGLawtxQjYE1jHQfZapgJ`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_rkoq7t5f.txt","to_foxy_logs":"crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.move:217:fun sui_amount_spec("} -->
```
crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.move:217:fun sui_amount_spec(
```

---

## Assistant

### Code `toolu_0169yRnT9tkRbzT8ehTmVtg7`

```python
spec_file = "crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.move"
snippet = read(spec_file, 210, 225)
foxy_inspect(snippet)
```

> 143 in, 118 out, 107411 cache read, 514 cache create

### Result `toolu_0169yRnT9tkRbzT8ehTmVtg7`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_1ybdxtn2.txt","to_foxy_logs":"fun staked_sui_amount_spec(\n    staked_sui: &StakedSui,\n): u64 {\n    staking_pool::staked_sui_amount(staked_sui)\n}\n\n#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]\nfun sui_amount_spec(\n    exchange_rate: &PoolTokenExchangeRate,\n): u64 {\n    staking_pool::sui_amount(exchange_rate)\n}\n\n#[spec(prove, target=staking_pool::sui_balance, ignore_abort)]\nfun sui_balance_spec(\n    pool: &StakingPool,\n"} -->
```
fun staked_sui_amount_spec(
    staked_sui: &StakedSui,
): u64 {
    staking_pool::staked_sui_amount(staked_sui)
}

#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}

#[spec(prove, target=staking_pool::sui_balance, ignore_abort)]
fun sui_balance_spec(
    pool: &StakingPool,

```

---

## Assistant

Now I'll check for any existing `@VERIFY` annotation above this spec, then make the edit. The function is a pure getter that cannot abort, so removing `ignore_abort` requires no asserts.

### Code `toolu_01Vyq8rEo1XBCN35APhHugY2`

```python
snippet = read(spec_file, 213, 222)
foxy_inspect(snippet)
```

> 326 in, 143 out, 107925 cache read, 261 cache create

### Result `toolu_01Vyq8rEo1XBCN35APhHugY2`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_79ose6sr.txt","to_foxy_logs":"    staking_pool::staked_sui_amount(staked_sui)\n}\n\n#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]\nfun sui_amount_spec(\n    exchange_rate: &PoolTokenExchangeRate,\n): u64 {\n    staking_pool::sui_amount(exchange_rate)\n}\n\n"} -->
```
    staking_pool::staked_sui_amount(staked_sui)
}

#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}


```

---

## Assistant

No `@VERIFY` annotation present. I'll remove `ignore_abort` and add a `@VERIFY` annotation marking the asserts stage. Since the getter cannot abort, no asserts are needed.

### Code `toolu_015Cu6QgqfuvYa8cc7THuNic`

```python
old = '''#[spec(prove, target=staking_pool::sui_amount, ignore_abort)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}'''

new = '''// @VERIFY(\U0001f6e1\ufe0f/\U0001f300)
#[spec(prove, target=staking_pool::sui_amount)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}'''

result = edit(spec_file, old, new)
foxy_inspect(read(spec_file, 215, 224))
```

> 227 in, 379 out, 108186 cache read, 469 cache create

### Result `toolu_015Cu6QgqfuvYa8cc7THuNic`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_ltjn46v1.txt","to_foxy_logs":"\n// @VERIFY(\ud83d\udee1\ufe0f/\ud83c\udf00)\n#[spec(prove, target=staking_pool::sui_amount)]\nfun sui_amount_spec(\n    exchange_rate: &PoolTokenExchangeRate,\n): u64 {\n    staking_pool::sui_amount(exchange_rate)\n}\n\n#[spec(prove, target=staking_pool::sui_balance, ignore_abort)]\n"} -->
```

// @VERIFY(🛡️/🌀)
#[spec(prove, target=staking_pool::sui_amount)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}

#[spec(prove, target=staking_pool::sui_balance, ignore_abort)]

```

---

## Assistant

Now I'll run the prover to verify. Crucially, since I removed `ignore_abort`, I must **not** use `--skip-spec-no-abort` — that's the phase that actually verifies abort-freedom.

### Code `toolu_01Nn51mqyA3XdiQaksHZUqWc`

```python
specs_pkg = packages[1]
result = await run_sui_prover(
    specs_pkg,
    args=["--functions", "sui_amount_spec", "--timeout", "60", "--force-timeout", "--cloud"],
    timeout=240,
    label="prove-sui_amount_spec",
)
foxy_inspect(result.get("response", "")[-2500:])
```

> 245 in, 242 out, 108655 cache read, 606 cache create

### Result `toolu_01Nn51mqyA3XdiQaksHZUqWc`

<!-- {"error":"'NoneType' object has no attribute 'get'","type":"AttributeError"} -->
```
⚠ AttributeError: 'NoneType' object has no attribute 'get'
```

---

## Assistant

### Code `toolu_01Sw24AH8EeE12rWZEpPSEif`

```python
foxy_inspect(type(result))
foxy_inspect(result)
```

> 43 in, 70 out, 109261 cache read, 487 cache create

### Result `toolu_01Sw24AH8EeE12rWZEpPSEif`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_aezfu28h.txt","to_foxy_logs":"<class 'NoneType'>\nNone"} -->
```
<class 'NoneType'>
None
```

---

## Assistant

The fork returned `None` — likely launched in non-blocking mode or the result wasn't captured. Let me check the task registry.

### Code `toolu_01G9TgufS1c7FMPipPKPwKSH`

```python
for t in list_tasks():
    foxy_inspect(f"#{t['id']} {t['name']} [{t['status']}] {t.get('elapsed', 0)}s")
    if t.get("response"):
        foxy_inspect(f"  {t['response'][:500]}")
```

> 109 in, 180 out, 109748 cache read, 113 cache create

### Result `toolu_01G9TgufS1c7FMPipPKPwKSH`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_xv4ktiaa.txt","to_foxy_logs":"#1 prove-from-scratch/improve/attempt-1 (1 items) [running] 2937s\n#85 prove-split_spec [error] 508s\n#86 prove-split_spec [done] 508s\n#87 sui-prover: specs [done] 505s\n#88 prove-split-fss [error] 340s\n#89 prove-split-fss [error] 340s\n#90 sui-prover: specs [done] 335s\n#91 sui-prover: specs [done] 327s\n#92 prove-stake_activation_epoch [error] 195s\n#93 prove-stake_activation_epoch [done] 195s\n#94 sui-prover: specs [done] 192s\n#95 prove-staked_sui_amount_spec [error] 100s\n#96 prove-staked_sui_amount_spec [done] 100s\n#97 sui-prover: specs [done] 95s\n#98 sui-prover: specs [done] 88s\n#99 prove-sui_amount_spec [error] 6s\n#100 prove-sui_amount_spec [running] 6s\n#101 sui-prover: specs [done] 3s"} -->
```
#1 prove-from-scratch/improve/attempt-1 (1 items) [running] 2937s
#85 prove-split_spec [error] 508s
#86 prove-split_spec [done] 508s
#87 sui-prover: specs [done] 505s
#88 prove-split-fss [error] 340s
#89 prove-split-fss [error] 340s
#90 sui-prover: specs [done] 335s
#91 sui-prover: specs [done] 327s
#92 prove-stake_activation_epoch [error] 195s
#93 prove-stake_activation_epoch [done] 195s
#94 sui-prover: specs [done] 192s
#95 prove-staked_sui_amount_spec [error] 100s
#96 prove-staked_sui_amount_spec [done] 100s
#97 sui-prover: specs [done] 95s
#98 sui-prover: specs [done] 88s
#99 prove-sui_amount_spec [error] 6s
#100 prove-sui_amount_spec [running] 6s
#101 sui-prover: specs [done] 3s
```

---

## Assistant

There's a running task #100. The `await` apparently returned before completion. Let me wait for it.

### Code `toolu_01HtZ4rjDgaZNC2QC4toPT7M`

```python
import time
for _ in range(40):
    r = get_task_result(100)
    if r and r.get("status") != "running":
        break
    time.sleep(5)
foxy_inspect(r.get("response", "")[-2500:] if r else "no result")
```

> 475 in, 158 out, 109861 cache read, 289 cache create

### Result `toolu_01HtZ4rjDgaZNC2QC4toPT7M`

<!-- {"error":"'NoneType' object is not subscriptable","type":"TypeError"} -->
```
⚠ TypeError: 'NoneType' object is not subscriptable
```

---

## Assistant

### Code `toolu_011gpnsAxo7938U5g945cQjE`

```python
r100 = get_task_result(100)
foxy_inspect(f"type={type(r100)}")
foxy_inspect(r100 if r100 else "None")
log = get_task_log(100, tail=60)
foxy_inspect(log if log else "no log")
```

> 40 in, 132 out, 110150 cache read, 633 cache create

### Result `toolu_011gpnsAxo7938U5g945cQjE`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_uwlx6avf.txt","to_foxy_logs":"type=<class 'dict'>\n{'status': 'success', 'result': {'summary': 'PASS sui_amount_spec', 'details': ['PASS staking_pool_specs::sui_amount_spec (_Check)', 'PASS staking_pool_specs::sui_amount_spec (_Assume)', 'PASS staking_pool_specs::sui_amount_spec (_SpecNoAbortCheck)'], 'totals': {'passed': 1, 'failed': 0, 'timeout': 0}, 'raw_verdict': 'Verification successful'}, 'response': None, 'stop_reason': 'tool_use', 'turns': 2, 'tokens_used': 950, 'usage': {'input_tokens': 561, 'output_tokens': 389, 'cache_read_input_tokens': 277847, 'cache_creation_input_tokens': 3732}, 'total_usage': {'input_tokens': 561, 'output_tokens': 389, 'cache_read_input_tokens': 277847, 'cache_creation_input_tokens': 3732}, 'time_elapsed': 12.2567880153656, 'error': None, 'cost_usd': 0.10486709999999999, 'harness': 'foxy', 'session_dir': '/Users/cos/asymptotic/agent/clients/mysten/sui/.foxy/sessions/d1_default_agent_prove-sui_amount_spec_20260616_000746_3209'}\n\n### Result `toolu_01YGQ7nwoB3SzjVqgyPv6BgH`\n\n<!-- {\"executed\":true,\"stdout\":\"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_tuv7g9zv.txt\",\"to_foxy_logs\":\"\\ud83d\\udd04 staking_pool_specs::sui_amount_spec_Check\\n\\ud83d\\udd04 staking_pool_specs::sui_amount_spec_Assume\\n\\ud83d\\udd04 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck\\n\\u2705 staking_pool_specs::sui_amount_spec_Assume\\n\\u2705 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck\\n\\u2705 staking_pool_specs::sui_amount_spec_Check\\nVerification successful\"} -->\n```\n\ud83d\udd04 staking_pool_specs::sui_amount_spec_Check\n\ud83d\udd04 staking_pool_specs::sui_amount_spec_Assume\n\ud83d\udd04 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck\n\u2705 staking_pool_specs::sui_amount_spec_Assume\n\u2705 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck\n\u2705 staking_pool_specs::sui_amount_spec_Check\nVerification successful\n```\n\n---\n\n## Assistant\n\n### Code `toolu_014mHAJGPNq9vPKE5PEBmEX3`\n\n```python\nfinal_result({\n    \"summary\": \"PASS sui_amount_spec\",\n    \"details\": [\n        \"PASS staking_pool_specs::sui_amount_spec (_Check)\",\n        \"PASS staking_pool_specs::sui_amount_spec (_Assume)\",\n        \"PASS staking_pool_specs::sui_amount_spec (_SpecNoAbortCheck)\",\n    ],\n    \"totals\": {\"passed\": 1, \"failed\": 0, \"timeout\": 0},\n    \"raw_verdict\": \"Verification successful\",\n})\n```\n\n> 220 in, 183 out, 140516 cache read, 547 cache create\n\n### Result `toolu_014mHAJGPNq9vPKE5PEBmEX3`\n\n<!-- {\"executed\":true,\"return_value\":{\"summary\":\"PASS sui_amount_spec\",\"details\":[\"PASS staking_pool_specs::sui_amount_spec (_Check)\",\"PASS staking_pool_specs::sui_amount_spec (_Assume)\",\"PASS staking_pool_specs::sui_amount_spec (_SpecNoAbortCheck)\"],\"totals\":{\"passed\":1,\"failed\":0,\"timeout\":0},\"raw_verdict\":\"Verification successful\"}} -->\n```\n\u2713 ok\n```\n\n---\n\n# Session Complete\n\n| | |\n|---|---|\n| status | success |\n| stop_reason | tool_use |\n| model | claude-sonnet-4-6 |\n| prompt | 0caeb6195b49adc01b3b8ff2c837b6258cf32089a96a36329d4c93897404b5cb |\n| parent | /Users/cos/asymptotic/agent/clients/mysten/sui/.foxy/sessions/spec_improve_20260616_000705_4231 |\n| turns | 2 |\n| tool_calls | 2 |\n| time | 12.25s |\n| ended_at | 2026-06-16T00:07:58.570748 |\n| cost | $0.1049 |\n| tokens | 561 in / 389 out |\n| cache | 277847 read / 3732 create |"} -->
````
type=<class 'dict'>
{'status': 'success', 'result': {'summary': 'PASS sui_amount_spec', 'details': ['PASS staking_pool_specs::sui_amount_spec (_Check)', 'PASS staking_pool_specs::sui_amount_spec (_Assume)', 'PASS staking_pool_specs::sui_amount_spec (_SpecNoAbortCheck)'], 'totals': {'passed': 1, 'failed': 0, 'timeout': 0}, 'raw_verdict': 'Verification successful'}, 'response': None, 'stop_reason': 'tool_use', 'turns': 2, 'tokens_used': 950, 'usage': {'input_tokens': 561, 'output_tokens': 389, 'cache_read_input_tokens': 277847, 'cache_creation_input_tokens': 3732}, 'total_usage': {'input_tokens': 561, 'output_tokens': 389, 'cache_read_input_tokens': 277847, 'cache_creation_input_tokens': 3732}, 'time_elapsed': 12.2567880153656, 'error': None, 'cost_usd': 0.10486709999999999, 'harness': 'foxy', 'session_dir': '/Users/cos/asymptotic/agent/clients/mysten/sui/.foxy/sessions/d1_default_agent_prove-sui_amount_spec_20260616_000746_3209'}

### Result `toolu_01YGQ7nwoB3SzjVqgyPv6BgH`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_tuv7g9zv.txt","to_foxy_logs":"\ud83d\udd04 staking_pool_specs::sui_amount_spec_Check\n\ud83d\udd04 staking_pool_specs::sui_amount_spec_Assume\n\ud83d\udd04 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck\n\u2705 staking_pool_specs::sui_amount_spec_Assume\n\u2705 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck\n\u2705 staking_pool_specs::sui_amount_spec_Check\nVerification successful"} -->
```
🔄 staking_pool_specs::sui_amount_spec_Check
🔄 staking_pool_specs::sui_amount_spec_Assume
🔄 staking_pool_specs::sui_amount_spec_SpecNoAbortCheck
✅ staking_pool_specs::sui_amount_spec_Assume
✅ staking_pool_specs::sui_amount_spec_SpecNoAbortCheck
✅ staking_pool_specs::sui_amount_spec_Check
Verification successful
```

---

## Assistant

### Code `toolu_014mHAJGPNq9vPKE5PEBmEX3`

```python
final_result({
    "summary": "PASS sui_amount_spec",
    "details": [
        "PASS staking_pool_specs::sui_amount_spec (_Check)",
        "PASS staking_pool_specs::sui_amount_spec (_Assume)",
        "PASS staking_pool_specs::sui_amount_spec (_SpecNoAbortCheck)",
    ],
    "totals": {"passed": 1, "failed": 0, "timeout": 0},
    "raw_verdict": "Verification successful",
})
```

> 220 in, 183 out, 140516 cache read, 547 cache create

### Result `toolu_014mHAJGPNq9vPKE5PEBmEX3`

<!-- {"executed":true,"return_value":{"summary":"PASS sui_amount_spec","details":["PASS staking_pool_specs::sui_amount_spec (_Check)","PASS staking_pool_specs::sui_amount_spec (_Assume)","PASS staking_pool_specs::sui_amount_spec (_SpecNoAbortCheck)"],"totals":{"passed":1,"failed":0,"timeout":0},"raw_verdict":"Verification successful"}} -->
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
| prompt | 0caeb6195b49adc01b3b8ff2c837b6258cf32089a96a36329d4c93897404b5cb |
| parent | /Users/cos/asymptotic/agent/clients/mysten/sui/.foxy/sessions/spec_improve_20260616_000705_4231 |
| turns | 2 |
| tool_calls | 2 |
| time | 12.25s |
| ended_at | 2026-06-16T00:07:58.570748 |
| cost | $0.1049 |
| tokens | 561 in / 389 out |
| cache | 277847 read / 3732 create |
````

---

## Assistant

Verification passed — all three phases including `_SpecNoAbortCheck` (the abort-freedom check that confirms removing `ignore_abort` is sound):

- ✅ `_Check`
- ✅ `_Assume`
- ✅ `_SpecNoAbortCheck`

Let me finalize the `@VERIFY` annotation to mark it done.

### Code `toolu_01DS9NtGNRmCaHq8HmbejwdX`

```python
read(spec_file, 215, 217)
edit(spec_file, "// @VERIFY(\U0001f6e1\ufe0f/\U0001f300)\n#[spec(prove, target=staking_pool::sui_amount)]",
     "// @VERIFY(\U0001f6e1\ufe0f/\u2705)\n#[spec(prove, target=staking_pool::sui_amount)]")
foxy_inspect(read(spec_file, 215, 223))
```

> 1968 in, 326 out, 110783 cache read, 172 cache create

### Result `toolu_01DS9NtGNRmCaHq8HmbejwdX`

<!-- {"executed":true,"stdout":"/var/folders/n2/rx0nqjvn0px8cv2gw3x6f1ww0000gn/T/foxy_stdout_0vqi5y5y.txt","to_foxy_logs":"\n// @VERIFY(\ud83d\udee1\ufe0f/\u2705)\n#[spec(prove, target=staking_pool::sui_amount)]\nfun sui_amount_spec(\n    exchange_rate: &PoolTokenExchangeRate,\n): u64 {\n    staking_pool::sui_amount(exchange_rate)\n}\n\n"} -->
```

// @VERIFY(🛡️/✅)
#[spec(prove, target=staking_pool::sui_amount)]
fun sui_amount_spec(
    exchange_rate: &PoolTokenExchangeRate,
): u64 {
    staking_pool::sui_amount(exchange_rate)
}


```

---

## Assistant

Done. The spec `staking_pool_specs::sui_amount_spec` is improved and verified.

**What changed:**
- Removed `ignore_abort` from the `#[spec(...)]` attribute.
- Added zero asserts — the target `staking_pool::sui_amount` is a pure field getter (`exchange_rate.sui_amount`) with no abort paths (no arithmetic, division, casts, table access, or `assert!` calls), so abort coverage is trivially empty.
- Marked the spec `// @VERIFY(🛡️/✅)` (abort conditions fully modeled and passing).

**Verification (run *without* `--skip-spec-no-abort`, so abort-freedom was genuinely checked):**
- ✅ `_Check`
- ✅ `_Assume`
- ✅ `_SpecNoAbortCheck` — confirms the function never aborts, validating the `ignore_abort` removal.

The spec went from `stub` → `asserts` stage (abort coverage complete, no `ignore_abort`).

Want me to add `ensures` semantics next (e.g. `ensures(result == staking_pool::get_sui_amount_field(exchange_rate))`-style postcondition), or move on to the neighbouring `sui_balance_spec` / `staked_sui_amount_spec` stubs that still carry `ignore_abort`?

> 213 in, 456 out, 110955 cache read, 2294 cache create

---

# Session Complete

| | |
|---|---|
| status | success |
| stop_reason | end_turn |
| model | claude-opus-4-8 |
| prompt | 8ba04c18cf4fc356e2c4244a9dd3ca559adbf7cac30c49e3dfbf75a8d43bdb42 |
| turns | 15 |
| tool_calls | 14 |
| time | 75.76s |
| ended_at | 2026-06-16T00:08:21.185274 |
| cost | $0.9616 |
| tokens | 4930 in / 3141 out |
| cache | 1624239 read / 7402 create |
| sub-agent tokens | 561 in / 389 out |
| total tokens | 5491 in / 3530 out |
| sub-agent cache | 277847 read / 3732 create |
| total cache | 1902086 read / 11134 create |


## Knowledge Output

- field: spec/improve/staking_pool_specs::sui_amount_spec
- data: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/sui_amount_spec/spec.improve.yaml
- meta: /Users/cos/asymptotic/agent/clients/mysten/sui/crates/sui-framework/packages/sui-system/specs/sources/staking_pool_specs.foxy/sui_amount_spec/improve.meta.yaml
