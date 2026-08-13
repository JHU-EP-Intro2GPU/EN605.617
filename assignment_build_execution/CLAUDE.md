# CLAUDE.md

Context for future Claude sessions working on this toolkit. Read this
before making changes.

## What this is

A bash + Python toolkit (`run_assignments.sh` + `yaml_helper.py`) that
executes a project's `build.sh`/`run.sh` using arguments described in an
`assignment_config.yaml` file. It has two modes, auto-detected by cwd:

- **Single-project mode**: cwd contains `.git` → run build/run directly.
- **Parent/collection mode**: cwd does NOT contain `.git` → treat
  immediate subfolders as git projects, recurse (re-invoke this same
  script) into each one, and aggregate results.

See `README.md` for user-facing usage; this file is about the
implementation.

## Architecture

- **`run_assignments.sh`** — all orchestration logic: mode detection,
  logging, git pull, stage execution, status aggregation. This is the
  only file a user runs directly.
- **`yaml_helper.py`** — a stateless CLI: `yaml_helper.py <config> <folder|build|run> [--count | --index N]`.
  It's the *only* place YAML is parsed. It never touches the filesystem
  beyond reading the config file. Exit codes are meaningful and load-bearing
  (see below) — the bash side branches on them. `build`/`run` each expand
  to a *list* of independent configurations (see "Multi-config model"
  below) — the CLI is queried in two steps (`--count` then `--index N`
  per config) rather than dumping everything in one call, specifically
  so bash never has to hold more than one config's worth of NUL-joined
  args in a single variable (see "Why two-step --count/--index" below).
- **`pyproject.toml`** — declares the `pyyaml` dependency so `uv run`
  can execute `yaml_helper.py` in an isolated env without the user
  manually managing a venv. `run_assignments.sh` prefers `uv run
  --project "$SCRIPT_DIR" python3 ...` when `uv` is on PATH, and falls
  back to bare `python3` (assumes PyYAML preinstalled) otherwise. Both
  paths are exercised in `run_assignments.sh` — don't remove the
  fallback, some environments won't have `uv`.

## Multi-config model

`build`/`run` in the YAML can each be:
1. a single dict (`{flag: value}`) → **one** configuration
2. a single flat list of scalars → **one** configuration, positional args
3. a list where every item is itself a dict or list → **N independent
   configurations**, run one after another, each reported separately.
   A failure in one never stops the rest.

`yaml_helper.py`'s `normalize_configs()` is where forms (1)/(2)/(3) all
get collapsed into "a list of configs" (length 1 for forms 1/2, length
N for form 3). Disambiguating (2) from (3) — both are YAML lists — is
done by checking whether *every* item is a dict/list (→ form 3) or *no*
item is (→ form 2, the whole list is one config's positional args). A
list mixing bare scalars with dicts/lists raises `ValueError` (surfaced
as exit code 2) rather than guessing — don't try to make this "smarter"
by e.g. treating scalar items as flags of their own; the ambiguity is
real and the user needs to disambiguate by wrapping (see example
config's `[--lint-only]`, `[input.txt]`).

## Contract between the two files (don't break this silently)

`yaml_helper.py <config> folder` — unchanged: exit 0, folder path (or
`.`) on stdout.

`yaml_helper.py <config> <build|run> --count` — prints the number of
independent configurations in that section (0 if present-but-empty,
e.g. `build: []`).

`yaml_helper.py <config> <build|run> --index N` (0-based) — prints that
one configuration's args as NUL-separated (`\0`) tokens, terminated by
a trailing `\0` if non-empty (empty output if the config has zero args).

Exit codes, consumed by `run_assignments.sh` (in
`run_all_configs_for_stage()`):
- `0` — success (see per-flag output above).
- `3` — the requested `build`/`run` section doesn't exist in the YAML
  at all. NOT an error — that stage simply isn't configured, and the
  bash side treats it as "skip, info log only." Note this is distinct
  from the section existing but being empty (`build: []`), which is
  exit `0` with `--count` printing `0`.
- `2` — real parse/read error (bad YAML, missing PyYAML, unreadable
  file, invalid mixed-list shape, or `--index` out of range).
- `1` — bad CLI usage (wrong arg count / unknown section name / missing
  `--count`/`--index` when querying build/run / non-integer `--index`).

If you change these exit-code semantics, update the `rc` handling in
`run_all_configs_for_stage()` in `run_assignments.sh` — there is now
exactly ONE such block, shared by both `build` and `run` (unlike the
old single-config version, this one IS deduplicated into a shared
function since the same per-config loop logic applies to both stages).

### Why two-step `--count`/`--index` (not one big dump)

An earlier version of this tool had `yaml_helper.py` dump ALL of a
section's args in one call, NUL-joined, read into a bash array via
`readarray -d '' arr < file`. That still works fine for a *single*
config. It does NOT extend cleanly to *multiple* configs: bash strings/
array elements are C-strings under the hood and cannot contain embedded
NUL bytes, so there's no way to hold "N configs, each NUL-joined
internally" as one delimited blob and split it into per-config strings
in bash — the moment you try to store one "record" (one config's worth
of NUL-joined args) in a bash variable, if you'd used NUL as also the
between-configs separator, everything after the first internal NUL is
lost when it round-trips through a variable/command-substitution.

The fix is to never let bash hold more than one config's args in a
single variable: `run_all_configs_for_stage()` calls the helper once
per config index, reading each config's NUL-joined output straight from
a temp file into a bash array with `readarray -d ''` (same safe pattern
as before, just looped). Don't "optimize" this into one call that
returns everything — it would reintroduce the embedded-NUL problem the
moment any config has more than a trivial number of args, and the
per-config `uv run`/`python3` startup overhead is not worth chasing
without profiling data showing it matters.

## Key design decisions (and why)

- **Build failure does not block run, and one bad config does not block
  the rest of that stage's configs.** Two independent levels of
  "keep going on failure": `run_single_project()` always runs the full
  `build` stage AND the full `run` stage regardless of how the other
  went (via two unconditional calls to `run_all_configs_for_stage()`);
  and *within* `run_all_configs_for_stage()`'s `for` loop over config
  indices, `run_stage()`'s non-zero return is only used to decide
  whether to increment `STAGE_PASSED` — it never breaks/returns out of
  the loop. If you change either of these, update the README's
  "Notes / limitations" section too.
- **Failed stage output → BOLD + UPPERCASE + red.** Done by piping the
  captured stdout/stderr through `tr '[:lower:]' '[:upper:]'` and
  wrapping in `${BOLD}${RED}...${NC}`. This happens only on non-zero
  exit from the stage script, never on the info/success/warn log lines
  themselves.
- **Screen output vs. file output diverge only in color codes.** `log()`
  echoes the colored string to the terminal and a color-stripped copy
  (via a `sed -E 's/\x1b\[[0-9;]*m//g'` filter) to `assignment_output.txt`.
  There is deliberately no separate "plain" and "colored" message
  authored per call site — one message, one strip filter — to avoid the
  two drifting out of sync.
- **`.assignment_status.env` is the parent/child handoff mechanism.**
  Single-project mode always writes this file (even on early-return
  paths like "no config found") with `PROJECT_NAME`, `HAD_CONFIG`,
  `HAD_BUILD`, `BUILD_TOTAL`, `BUILD_PASSED`, `HAD_RUN`, `RUN_TOTAL`,
  `RUN_PASSED` as shell-sourceable `KEY=VALUE` lines — `*_TOTAL`/
  `*_PASSED` replaced the old single `BUILD_OK`/`RUN_OK` booleans so the
  parent can report "2/3 configs passed" instead of a single pass/fail
  bit. "Stage OK" is now derived (`PASSED == TOTAL`), not stored
  directly — computed both in `run_all_configs_for_stage()`'s own log
  summary line and again in `run_parent()`'s aggregation block; these
  two computations must stay consistent if you touch either. Parent
  mode `source`s the file in a subshell after each child invocation,
  builds one line of `assignments_results.txt`, then deletes it. If you
  add/rename status fields, update `write_status_file()`, the
  `source`-and-build-line block in `run_parent()`, AND the format
  described in the README.
- **git pull only on `main`/`master`, and only ever a warning on
  failure.** Never make a failed `git pull` (e.g. "no upstream
  configured", detached HEAD, merge conflict) abort the whole run — the
  user explicitly wants build/run to still be attempted.
- **`uv run` fallback logic must stay resilient offline.** In sandboxes
  or CI without network/PyPI access, `uv run` will hard-fail trying to
  resolve `pyyaml` (403/no solution errors) rather than silently falling
  back to system python. This is expected uv behavior, not a bug in this
  script — don't try to "fix" it by adding a try-uv-then-fallback-to-python3
  retry inside the script; that would mask real dependency problems in
  the normal (networked) case. The PATH-based choice (uv present at
  script start → committed to uv for the whole run) is intentional.

## Testing this locally

There's no test suite checked in (this is a small utility, not a
library). When making changes, smoke-test manually with throwaway git
repos, e.g.:

```bash
mkdir -p /tmp/demo/projA/src && cd /tmp/demo/projA
git init -q -b main
git -c user.email=a@a.com -c user.name=a commit -q --allow-empty -m init
printf '#!/usr/bin/env bash\necho ok\n' > src/build.sh
printf '#!/usr/bin/env bash\necho ok\n' > src/run.sh
chmod +x src/build.sh src/run.sh
cp /path/to/assignment_config.yaml .
/path/to/run_assignments.sh
cat assignment_output.txt
cat .assignment_status.env
```

Cover at minimum: build success, build failure (non-zero exit + stderr),
missing config file, non-main branch (verify git pull is skipped),
parent/collection mode with a mix of the above across 2-3 subfolders
(verify `assignments_results.txt` lines match each subfolder's actual
outcome) — AND, for the multi-config feature specifically: a `build` or
`run` list with 3+ configs where a *middle* one fails, verifying (a) the
configs before and after it still execute, (b) `[i/N]` labels and the
per-stage summary line are correct, and (c) `BUILD_TOTAL`/`BUILD_PASSED`
in `.assignment_status.env` (and the `assignments_results.txt` line)
reflect the actual pass count, not just 0/1.

## Things NOT to do

- Don't add `set -e` to `run_assignments.sh`. Failures (build.sh exiting
  non-zero, git pull failing, missing config) are expected, handled
  outcomes, not script bugs — `set -e` would abort the whole run on the
  first one instead of recording it and continuing.
- Don't parse YAML in bash (e.g. with `grep`/`awk` hacks) even for
  "simple" cases — `yaml_helper.py` is the single source of truth for
  config parsing and its dict-vs-list branching. Keep parsing logic
  Python-side.
- Don't rename `.assignment_status.env` or change its format without
  grepping for both places that read/write it (`write_status_file`,
  and the `source` block in `run_parent`).
- Don't collapse `run_all_configs_for_stage()`'s per-config loop back
  into a single "dump everything, split in bash" call — see "Why
  two-step --count/--index" above; it's a correctness fix (embedded
  NUL bytes can't survive in a bash variable), not stylistic.
- Don't let a config-list validation error (mixed scalars/dicts) fail
  silently or get treated as "0 configs" — it must surface as a loud
  parse error (exit 2, `log_error`), since silently skipping could look
  identical to "nothing configured" and hide a typo'd config from the
  user.
