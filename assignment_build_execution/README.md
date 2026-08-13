# Assignment Runner

A small toolkit that runs a project's `build.sh` and/or `run.sh` using
arguments described in a YAML config file (`assignment_config.yaml`).
It works on a single git project, or on a folder containing many git
projects at once, and produces both live colored terminal output and
plain-text log files.

## Files

| File                     | Purpose                                                             |
|---------------------------|----------------------------------------------------------------------|
| `run_assignments.sh`      | Main script — run this.                                             |
| `yaml_helper.py`           | Internal helper `run_assignments.sh` shells out to for YAML parsing. |
| `pyproject.toml`           | `uv`-managed dependency spec (PyYAML) for `yaml_helper.py`.          |
| `assignment_config.yaml`   | Example / template config — copy this into each project you target. |

## Requirements

- `bash` 4.4+
- `python3`
- PyYAML — handled automatically if you have [`uv`](https://docs.astral.sh/uv/)
  installed (recommended); otherwise install manually with
  `pip3 install pyyaml`.

`run_assignments.sh` auto-detects `uv`: if it's on your `PATH`, the
script runs `yaml_helper.py` via `uv run --project <script dir>`, which
transparently creates an isolated virtual environment satisfying
`pyproject.toml` on first use (no manual setup needed). If `uv` isn't
found, it falls back to your system `python3`, which must already have
PyYAML installed.

## Setup

Copy `run_assignments.sh`, `yaml_helper.py`, and `pyproject.toml` anywhere
on disk (they need to stay together, in the same directory). Then place a
`assignment_config.yaml` at the root of each git project you want to run
against (see the example file in this repo for the format).

Optionally make the script easy to invoke from anywhere:

```bash
chmod +x run_assignments.sh yaml_helper.py
ln -s "$(pwd)/run_assignments.sh" /usr/local/bin/run_assignments
```

## `assignment_config.yaml` format

```yaml
folder: src        # path (relative to this file) containing build.sh / run.sh

build:              # optional — omit to skip the build stage
  target: release    # dict form -> ./build.sh --target release -O 2 --verbose true
  O: 2                # 1-char key -> single-dash flag: -O 2
  verbose: true

run:                # optional — omit to skip the run stage
  - input.txt        # list form -> ./run.sh input.txt 42 --dry-run
  - "42"
  - --dry-run
```

- `folder` — required. Where `build.sh`/`run.sh` live, relative to the
  config file's directory.
- `build` / `run` — each is optional and independent. Each may be
  written as **any** of:
  1. a **dictionary** of `flag: value` pairs — a 1-character key becomes
     `-k value`, a longer key becomes `--key value` — for **one**
     execution; or
  2. a **flat list** of values passed through as positional arguments,
     in order, for **one** execution; or
  3. a **list where every item is itself a dict or a list** — this runs
     **multiple independent executions**, one per item, one after
     another, using whichever of forms (1)/(2) that item is written as:

     ```yaml
     build:
       - target: debug        # execution 1 (dict form)
         verbose: true
       - target: release       # execution 2 (dict form)
         O: 2
       - [--lint-only]          # execution 3 (list form)

     run:
       - [input.txt]            # execution 1 (list form)
       - mode: quick             # execution 2 (dict form)
       - mode: thorough          # execution 3 (dict form)
         retries: 3
     ```

     Each execution is fully independent: **if one fails, the rest
     still run** — a failure never stops the remaining configurations
     in the list, and each is reported individually (see "Multiple
     configurations" below).

  Forms (2) and (3) are both YAML lists, so the rule that tells them
  apart is: a list of *bare scalar values* (strings/numbers/booleans)
  is form (2) — one execution, positional args. A list where *every*
  item is a dict or a list is form (3) — multiple executions. Mixing
  bare scalars with dicts/lists in the same top-level list is invalid
  and the script will error out with a clear message rather than
  guessing what you meant.

## Usage

### Single project

Run from the root of a git project (a directory containing `.git`):

```bash
cd my-project
/path/to/run_assignments.sh                       # looks for assignment_config.yaml
/path/to/run_assignments.sh other_config.yaml      # or a custom filename
```

This will:

1. Check the current git branch; if it's `main` or `master`, run
   `git pull` (otherwise it warns and skips the pull — it will not pull
   on a feature branch).
2. Run every configuration in `build` (if a `build` section exists in
   the config), one after another.
3. Run every configuration in `run` (if a `run` section exists in the
   config), one after another.
4. Write all output (with ANSI colors stripped) to `assignment_output.txt`
   in the project root, while also printing colored/emoji output live to
   the terminal.
5. Write `.assignment_status.env` — a small machine-readable status file
   used internally when this script is invoked from parent (collection)
   mode. Safe to ignore/delete in single-project use.

If a configuration's script exits non-zero, its captured output is
reprinted in **BOLD, UPPERCASE, RED** so failures are unmistakable in
both the terminal and `assignment_output.txt` — but that failure does
**not** stop the rest of the configurations in the same `build`/`run`
list, and does not stop the other stage from running.

#### Multiple configurations

If `build` or `run` is a list-of-configurations (form 3 above), each is
logged with a `[i/N]` label, e.g. `BUILD [2/3]`, and a one-line summary
is printed after the last one in that stage:

```
✅ BUILD summary: 3/3 configurations succeeded
❌ RUN summary: 2/3 configurations succeeded
```

The `[i/N]` label and summary line are only shown when there's more
than one configuration — a single dict/list `build`/`run` still logs
exactly as before (just `BUILD`, no bracket).

### Collection of projects

Run from a folder that is **not** itself a git project (i.e. it contains
no `.git`), but contains one or more git project subfolders:

```bash
cd all-my-projects/
/path/to/run_assignments.sh
```

This will recurse into every immediate subfolder that contains `.git`,
run the same single-project flow there (looking for
`assignment_config.yaml` at that subfolder's root — warning, not
failing, if it's missing), and then write a summary to
`assignments_results.txt` in the parent folder, one line per project.
The build/run status shows how many of that stage's configurations
passed out of the total:

```
projA | config: yes | build: OK (3/3)         | run: OK (3/3)
projB | config: yes | build: FAILED (2/3)      | run: OK (1/1)
projC | config: no  | build: none              | run: none
```

`build: none` / `run: none` means that section wasn't in the config at
all; `build: none configured` (shown when the section exists but is
empty, e.g. `build: []`) means the section was present but defined zero
configurations to run.

## Colors & symbols

| Style              | Meaning                                   |
|---------------------|--------------------------------------------|
| 🚀 blue/bold        | Section header                             |
| ℹ️ blue             | Informational message                      |
| ✅ green            | Success                                    |
| ⚠️ yellow           | Warning (missing config, skipped git pull, non-fatal git pull failure) |
| ❌ red              | Error / failure                            |
| **BOLD CAPS red**   | Captured output of a failed build/run stage |

## Notes / limitations

- A project's `build` failing does **not** prevent its `run` stage from
  executing — they're independent per the config. Likewise, one failed
  configuration within a `build` or `run` list does **not** stop the
  remaining configurations in that same list from running. Check
  `assignments_results.txt` / `assignment_output.txt` for the real
  per-configuration outcome.
- Collection mode only looks one level deep (immediate subfolders with
  `.git`); it does not recurse into nested collections.
- `git pull` is intentionally skipped on any branch other than `main`/
  `master`, and a failed pull (e.g. no upstream configured) is a warning,
  not a fatal error — the build/run stages still proceed.
- A `build`/`run` list mixing bare scalars with dicts/lists (ambiguous
  between "one execution, positional args" and "multiple executions")
  is rejected with an error rather than guessed at — wrap positional
  values in their own `[ ... ]` list item if you meant them as one
  configuration among several (see the `assignment_config.yaml`
  example).
