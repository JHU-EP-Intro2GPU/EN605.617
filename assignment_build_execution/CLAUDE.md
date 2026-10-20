# CLAUDE.md

Context for future Claude sessions working on this toolkit. Read this
before making changes.

## What this is

A bash + Python toolkit (`run_assignments.sh` + `yaml_helper.py`) that
executes a project's `build.sh`/`run.sh` using arguments and stdin lines
described in an `assignment_config.yaml` file. It has two modes,
auto-detected by cwd:

- **Single-project mode**: cwd contains `.git` → run build/run directly.
- **Parent/collection mode**: cwd does NOT contain `.git` → treat
  immediate subfolders as git projects, recurse (re-invoke this same
  script) into each one, and aggregate results.

See `README.md` for user-facing usage; this file is about the
implementation.

## Architecture

- **`run_assignments.sh`** — all orchestration logic: mode detection,
  logging, git pull, stage execution, stdin plumbing, status aggregation.
  The only file a user runs directly.
- **`yaml_helper.py`** — a stateless CLI. It's the *only* place YAML is
  parsed, and it never writes anything. Usage:
  `yaml_helper.py <config> <folder|build|run> [--count | --index N | --stdin-index N]`.
  Exit codes are load-bearing (see below).
- **`pyproject.toml`** — declares the `pyyaml` dependency so `uv run`
  can execute `yaml_helper.py` in an isolated env. `run_assignments.sh`
  prefers `uv run --project "$SCRIPT_DIR" python3 ...` when `uv` is on PATH,
  and falls back to bare `python3` (assumes PyYAML preinstalled) otherwise.
  Don't remove the fallback; some environments won't have `uv`.
- **`tests/run_tests.sh`** — end-to-end suite. See "Testing" below.
- **`examples/`** — two example projects (`greeter/` for stdin, `build-matrix/`
  for multi-config builds). They have no `.git` (nested repos can't be
  committed); the tests copy them and `git init` the copies.

## Configuration model

`build`/`run` can each be:
1. a single dict → **one** configuration. Its reserved keys `stdin` and
   `args` are not flags; every other key is a flag.
2. a single flat list of scalars → **one** configuration, positional args.
   Cannot carry stdin.
3. a list where every item is itself a dict or list → **N independent
   configurations**, run one after another, each reported separately. A
   failure in one never stops the rest.

Per configuration, `yaml_helper.py`'s `parse_config()` returns
`(argv_tokens, stdin_lines)`:
- flags first (`-k v` / `--key v`), then the `args` list, so
  `{mode: quick, args: [x]}` → `--mode quick x`;
- `stdin` is a string (one line) or a list (one line per value);
  `_plain_values()` rejects mappings and nested lists;
- booleans render as lowercase `true`/`false` (`_to_text`). This was
  `True`/`False` before, which contradicted the documented behavior.

`normalize_configs()` collapses forms 1/2/3 into a list of raw configs.
Disambiguating (2) from (3) (both YAML lists) is by "every item is dict/list"
(→ form 3) vs "no item is" (→ form 2). A mix raises `ValueError`. Don't make
this smarter by guessing: wrap the values in their own list, as the examples do.

## Contract between the two files (don't break this silently)

`yaml_helper.py <config> folder` → exit 0, folder path (or `.`) on stdout.

`yaml_helper.py <config> <build|run> --count` → number of configurations
(0 if present but empty, e.g. `build: []`).

`yaml_helper.py <config> <build|run> --index N` (0-based) → that config's
argv tokens, each followed by `\0` (empty output if none).

`yaml_helper.py <config> <build|run> --stdin-index N` (0-based) → that
config's stdin lines, each followed by `\0` (empty output if none).

Every configuration is parsed and validated before any query is answered,
so `--count` also fails on a malformed config. Bash never sees a half-valid
section.

Exit codes, consumed by `run_all_configs_for_stage()` in `run_assignments.sh`:
- `0` — success.
- `3` — the `build`/`run` section doesn't exist. NOT an error: that stage is
  skipped with an info line. Distinct from present-but-empty (`[]`), which is
  exit `0` with `--count` printing `0`.
- `2` — real parse/read error (bad YAML, missing PyYAML, unreadable file,
  invalid mixed-list shape, invalid `args`/`stdin` shape, or an out-of-range
  `--index` / `--stdin-index`).
- `1` — bad CLI usage.

If you change these semantics, update the `rc` handling in
`run_all_configs_for_stage()`. There is exactly one such block, shared by
both stages.

### Why per-config `--index` / `--stdin-index` (not one big dump)

Bash strings can't contain NUL bytes. Packing N configs' NUL-joined args into
one blob and splitting it in bash can't work, because the NUL that separates
configs is the same byte used inside a config. So the runner asks for one
config at a time and reads each reply straight from a temp file with
`readarray -d ''`. Stdin is handled the same way: `--stdin-index` output is
written to a temp file, never held in a bash variable, so blank lines and
embedded newlines survive. Don't "optimize" these into a single call.

## How stdin reaches the script

In `run_stage()`, the `stdin_file` argument (6th positional) is either empty
or a file with one line per stdin value.
- Non-empty file: the script runs with `< $stdin_file`, the lines are logged
  as `⏎ <line>` under an info line, and the count is shown.
- Empty or absent: the script runs with `< /dev/null`. A script that prompts
  without a configured answer gets EOF instead of blocking the run. This is
  deliberate: the old behavior inherited the terminal's stdin, which could hang
  a scheduled or unattended run. Don't remove it.

`run_all_configs_for_stage()` builds the file with `printf '%s\n' "${items[@]}"`,
guarded by a length check. An unguarded `printf '%s\n'` with zero arguments
prints one newline, which would silently give every script one blank line of
input.

## Key design decisions (and why)

- **Build failure does not block run, and one bad config does not block the rest
  of its stage.** `run_single_project()` always runs both stages. Inside
  `run_all_configs_for_stage()`, `run_stage()`'s non-zero return only affects
  `STAGE_PASSED`; it never breaks the loop. If you change either, update the
  README's "Notes / limitations".
- **A malformed section is reported and skipped; the other section still runs.**
  The count call fails with exit 2, the stage logs an error and returns, and
  `run_single_project()` continues.
- **Failed stage output → BOLD + UPPERCASE + red.** Output is piped through
  `tr '[:lower:]' '[:upper:]'`. This happens only on a non-zero exit, never on
  the info/success/warn lines. Tests and any future expectations must match the
  uppercase text.
- **Screen output and file output differ only in color codes.** `log()` echoes
  the colored string and writes a color-stripped copy (via
  `sed -E 's/\x1b\[[0-9;]*m//g'`) to `assignment_output.txt`. Don't author
  separate plain and colored messages per call site.
- **`.assignment_status.env` is the parent/child handoff.** Single-project
  mode always writes it, even on early returns. Parent mode sources it in a
  subshell, builds one line of `assignments_results.txt`, and deletes it. If you
  add or rename fields, update `write_status_file()`, the `source` block in
  `run_parent()`, and the README.
- **git pull only on `main`/`master`, and only ever a warning on failure.**
  A failed pull must never abort the run.
- **uv fallback stays resilient offline.** With uv on PATH, the runner commits
  to `uv run` for the whole run. Offline it fails with a resolution error, which
  is expected and is not masked: don't add a try-uv-then-python3 retry inside the
  runner. The test harness handles this case itself (see "Testing").

## Testing

`tests/run_tests.sh` is the test suite. Run it from anywhere. It needs bash
4.4+ (checked at startup), `git`, and python3. PyYAML is needed for the python3
path; uv is optional.

- Each test builds throwaway git projects under `mktemp -d`, runs the runner,
  and asserts on `assignment_output.txt`, `.assignment_status.env`,
  `assignments_results.txt`, and `calls.log`. Fixture scripts append to
  `calls.log` as `$0 $*`, so each test can check exactly which executions ran,
  and in what order.
- Use `write_file` / `write_script` for fixtures. They read content from stdin,
  so heredocs keep the closing marker at column 0.
- `run_runner` passes `< /dev/null` and calls `run_limited 120`, so a regression
  that reintroduces a blocking prompt fails the test instead of hanging.
  `run_limited` uses `timeout` or `gtimeout` if present. Otherwise it uses a
  small watchdog, because macOS has neither by default. Never call `timeout`
  directly in the harness: on macOS that produced exit 127 and nearly every
  test failed. `RUNNER_TESTS_FORCE_WATCHDOG=1` forces the watchdog path for
  testing.
- Helper checks call `"${HELPER_PY[@]}"`, which is the same interpreter the
  runner uses (`uv run ...` or `python3`). Calling bare `python3` there breaks
  when uv is the only interpreter with PyYAML.
- `uv_can_provide_pyyaml` decides whether the uv path can run. If uv is present
  but can't supply PyYAML, the suite prints a `SKIP` line and runs only the
  python3 path. It must say the uv path is untested. Don't turn that into a
  silent pass.
- When uv is usable, the suite runs twice: once with uv, once with uv removed
  hidden. `make_path_without_uv` builds the child's PATH: directories without
  uv are kept exactly as they are, and a directory that contains uv is replaced
  by a directory of symlinks to its other executables. Two things must hold:
  (1) don't symlink a venv's python3 into another directory, because Python
  finds `pyvenv.cfg` from the directory of the executable it was started through,
  and the venv (with PyYAML) is lost; (2) don't remove a directory that holds uv
  as a whole, because a directory shared by uv and bash or python3 (for example,
  `/opt/homebrew/bin`) would then hide those tools too, and on macOS the runner
  would be started by `/bin/bash` 3.2. The runner is always started with
  `$RUNNER_BASH`, resolved before PATH changes.
- Last verified: all checks passed on the python3 path. The uv path has not
  been exercised in this sandbox (no PyPI access); verify it on a networked
  machine before relying on it.

When you add a configuration form or a stage behavior, add a test for it in the
same change. Include a negative test (a rejected shape) if the change adds
validation.

## Things NOT to do

- Don't add `set -e` to `run_assignments.sh`. Failures (build.sh exiting
  non-zero, git pull failing, missing config) are expected outcomes, and
  `set -e` would abort the run.
- Don't parse YAML in bash (`grep`/`awk` hacks included). `yaml_helper.py` is
  the single source of truth for config parsing.
- Don't rename `.assignment_status.env` or change its format without grepping for
  both the writer and the `source` block in `run_parent`.
- Don't collapse the per-config `--index` / `--stdin-index` loop into one
  "dump everything" call. See "Why per-config" above.
- Don't let scripts inherit the runner's stdin. Give them the configured file or
  `/dev/null`.
- Don't let a config validation error fail silently or turn into "0 configs".
  Nothing is run, the error is named, and the exit is 2.
- Don't add a stdin-related feature that puts input into a bash variable or an
  argv element. Secrets and blank lines both need the file route.
- Don't use the `stdin` or `args` keys as flag names. They're reserved, and the
  README says so.