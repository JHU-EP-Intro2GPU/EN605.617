# Assignment Runner

A small toolkit that runs a project's `build.sh` and/or `run.sh` using
arguments and input described in a YAML config file (`assignment_config.yaml`).
It works on a single git project, or on a folder containing many git
projects at once. It produces live colored terminal output and plain-text
log files. Scripts that ask for input can be driven by supplying the
answers in the config.

## Files

| File / folder             | Purpose                                                              |
|----------------------------|----------------------------------------------------------------------|
| `run_assignments.sh`       | Main script — run this.                                              |
| `yaml_helper.py`           | Internal helper `run_assignments.sh` shells out to for YAML parsing. |
| `pyproject.toml`           | `uv`-managed dependency spec (PyYAML) for `yaml_helper.py`.          |
| `assignment_config.yaml`   | Example / template config — copy this into each project you target.  |
| `tests/run_tests.sh`       | Test suite (see "Testing" below).                                    |
| `examples/`                | Example projects: an interactive `run.sh` and a multi-build project. |

## Requirements

- `bash` 4.4+. macOS ships bash 3.2 at `/bin/bash`, so on a Mac install a
  newer one with `brew install bash` (the scripts check the version and say so
  if it's too old).
- `python3`
- PyYAML — handled automatically if you have [`uv`](https://docs.astral.sh/uv/)
  installed (recommended); otherwise install manually with
  `pip3 install pyyaml`.

`run_assignments.sh` auto-detects `uv`: if it's on your `PATH`, the
script runs `yaml_helper.py` via `uv run --project <script dir>`, which
transparently creates an isolated virtual environment (`.venv/`, next to the
script) satisfying `pyproject.toml` on first use. It needs network access, or
a warm uv cache, the first time. If `uv` isn't found, it falls back to your
system `python3`, which must already have PyYAML installed.

## Setup

Copy `run_assignments.sh`, `yaml_helper.py`, and `pyproject.toml` anywhere
on disk (they need to stay together, in the same directory). Then place an
`assignment_config.yaml` at the root of each git project you want to run
against (see the template in this repo for the format).

Optionally make the script easy to invoke from anywhere:

```bash
chmod +x run_assignments.sh yaml_helper.py
ln -s "$(pwd)/run_assignments.sh" /usr/local/bin/run_assignments
```

## `assignment_config.yaml` format

The file holds one or more **modules**. A module is one buildable/runnable part
of the project, and it carries the full configuration for that part: a `name`,
and optionally `folder`, `build` and `run`.

```yaml
modules:
  - name: current     # the default module (used when no -m is given)
    folder: src        # where build.sh / run.sh live (relative to the project root)
    build:              # optional — omit to skip the build stage
      target: release    # dict form -> ./build.sh --target release -O 2 --verbose true
      O: 2                # 1-char key -> single-dash flag: -O 2
      verbose: true
    run:                # optional — omit to skip the run stage
      - input.txt        # list form -> ./run.sh input.txt 42 --dry-run
      - "42"
      - --dry-run

  - name: tools         # a second module, selected with -m tools
    folder: lib/tools    # optional; defaults to the name ("tools")
    run:
      - [--check]
```

### Modules

- `name` — required, non-empty, unique within the file. It names the module for
  `-m`/`--module`. A module named **`current`** is the default: it is used when
  no `-m` is given.
- `folder` — optional. Where `build.sh`/`run.sh` live, relative to the project
  root. If it is missing, the folder is the module's `name`. So
  `name: tools` runs in `tools/`.
- `build` / `run` — optional, as described next.
- `modules` may be a list of module objects, or a single module object.

Choosing a module:

| Command                       | Module used                                              |
|-------------------------------|----------------------------------------------------------|
| `run_assignments.sh -m tools` | `tools` (an error if there is no module of that name)    |
| `run_assignments.sh`          | `current`, if a module has that name                      |
| no `-m` and no `current`      | nothing runs: the error lists the available modules      |

The module name is never guessed from the project's folder name. A project
called `tools` does not use the `tools` module by default; name a module
`current` for that, or pass `-m`.

A file with only one module is still a module list, so name that one `current`
or pass `-m` to select it.

Modules cannot be mixed with top-level keys. A file must use either `modules`,
or the single-configuration shape below, not both. Module names must be unique.
`-m` on a file without a `modules` key is an error.

### Single configuration (no `modules` key)

A project with only one part can put the keys at the top level. This is the
same as a single module with no name, and `-m` is not accepted with it:

```yaml
folder: src        # path (relative to this file) containing build.sh / run.sh

build:
  target: release
  O: 2

run:
  - input.txt
  - "42"
  - --dry-run
```

`folder` defaults to `.` (the project root) when it is omitted.

### Build and run sections

- `build` / `run` — each is optional and independent. Each may be
  written as **any** of:
  1. a **dictionary** of keys — one execution. Its keys are flags, plus two
     reserved keys described below; or
  2. a **flat list** of values passed through as positional arguments, in
     order, for **one** execution; or
  3. a **list where every item is itself a dict or a list** — this runs
     **multiple independent executions**, one per item, one after another,
     using whichever of forms (1)/(2) that item is written as:

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
       - stdin: [Ada, blue]      # execution 3 (dict form, input only)
       - args: [input.txt]       # execution 4 (positional args plus input)
         stdin: [Ada, blue]
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

### Keys inside a dictionary configuration

| Key          | Meaning                                                                    |
|--------------|----------------------------------------------------------------------------|
| `stdin`      | Input for the script, one line per value (see below).                      |
| `args`       | A list of positional values, appended after any flags.                     |
| anything else | A flag. A 1-character key becomes `-k value`; a longer key becomes `--key value`. `true`/`false` are written lowercase. |

`stdin` and `args` are reserved, so a flag with either name can't be written
in dictionary form. A flag value that is a list is not supported; write the
values as separate flags or as `args`.

Because form (2) has no room for keys, a configuration that needs both
positional values and input must use form (1) with an `args` key, as in
`- args: [input.txt]` plus `stdin: [...]` above.

## Input for scripts that ask questions

Some `run.sh` scripts read answers from standard input. Give each configuration
the answers it needs with `stdin:`:

```yaml
run:
  - stdin: [Ada, blue]                 # run.sh receives "Ada", then "blue"
  - args: [--shout]                     # flag or positional values...
    stdin: [Grace, green]               # ...and its input
  - stdin: "Linus\nblack"              # a string: one line per "\n"
```

- `stdin` may be a list (one line per value) or a string (one line per
  `\n`). Values are written exactly as given, so a YAML number such as
  `42` becomes the line `42`.
- Each configuration gets its own input. Nothing carries over from one
  configuration to the next.
- The supplied lines are echoed into the log, indented and marked with
  `⏎`, so you can see what the script was given. Be aware that this means
  passwords or other secrets placed in `stdin` appear in `assignment_output.txt`.
- **A configuration with no `stdin` gets an empty input**: the script's first
  `read` sees end-of-file instead of waiting for a keyboard. A prompt the
  configuration didn't anticipate therefore fails cleanly rather than hanging
  the run.
- Output from an interactive script appears in the log after the script
  finishes, not as it runs (as for every script).

## Usage

### Single project

Run from the root of a git project (a directory containing `.git`):

```bash
cd my-project
/path/to/run_assignments.sh                       # runs the "current" module
/path/to/run_assignments.sh -m tools               # runs the module named "tools"
/path/to/run_assignments.sh --module=tools         # same, with =
/path/to/run_assignments.sh other_config.yaml      # or a custom filename
/path/to/run_assignments.sh -m tools assignment_config.yaml /path/to/my-project   # or from anywhere
/path/to/run_assignments.sh --help                 # usage
```

Arguments may come in any order. `-m` / `--module` takes a module name. Any
other argument that is an existing directory is the project directory, and
any other argument is the config file name.

This will:

1. Choose the module (see "Modules" above). If it can't be chosen, the error
   names the modules that are available, and nothing is pulled or run.
2. Check the current git branch; if it's `main` or `master`, run
   `git pull` (otherwise it warns and skips the pull — it will not pull
   on a feature branch).
3. Run every configuration in `build` (if a `build` section exists in
   the module), one after another.
4. Run every configuration in `run` (if a `run` section exists in the
   module), one after another.
5. Write all output (with ANSI colors stripped) to `assignment_output.txt`
   in the project root, while also printing colored/emoji output live to
   the terminal.
6. Write `.assignment_status.env` — a small machine-readable status file
   used internally when this script is invoked from parent (collection)
   mode. Safe to ignore/delete in single-project use.

If a configuration's script exits non-zero, its captured output is
reprinted in **BOLD, UPPERCASE, RED** so failures are unmistakable in
both the terminal and `assignment_output.txt` — but that failure does
**not** stop the rest of the configurations in the same `build`/`run`
list, and does not stop the other stage from running.

If a section is malformed (for example, bare values mixed with dictionaries in
one list), the error is reported and that section is skipped. The other
section still runs.

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

Each project gets a time limit. The default is **10 minutes**, and `-t` /
`--timeout` sets it in whole minutes:

```bash
/path/to/run_assignments.sh -t 30          # 30 minutes per project
/path/to/run_assignments.sh --timeout=5    # 5 minutes per project
```

A project that is still running at its limit is stopped, together with the
build and run scripts it started, and is recorded as `TIMED OUT`. The next
project then starts. The limit covers the whole project pass (git pull, build,
and run). It applies only to collection runs: a single-project run ignores `-t`
and says so.

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
| ℹ️ blue             | Informational message (including the stdin lines supplied to a script) |
| ✅ green            | Success                                    |
| ⚠️ yellow           | Warning (missing config, skipped git pull, non-fatal git pull failure) |
| ❌ red              | Error / failure                            |
| **BOLD CAPS red**   | Captured output of a failed build/run stage |

## Testing

```bash
tests/run_tests.sh          # or: zsh tests/run_tests.sh
```

The test harness runs under bash 4.4+ or zsh 5.9+. The runner itself is a bash
script, so bash must be installed either way.

The suite builds throwaway git projects in a temporary directory, runs the
runner against them, and checks the logs, status files, and which scripts
actually executed. It covers:

- every configuration form (dict, flat list, list of configurations) for both
  stages, and the flags/positional-argument rendering;
- `stdin` in each form, including strings with several lines, `args` combined
  with `stdin`, configurations that each get their own input, and scripts run
  without input;
- a failing configuration in the middle of a list, with the rest still running;
- a build failure not blocking the run stage, and a missing `run.sh`;
- empty and absent sections, custom `folder`, missing config files;
- git pull on `main` (failure is a warning) and on a feature branch (skipped);
- the project-directory argument, collection mode with a mix of projects, and the
  example projects in `examples/`;
- module selection: `-m` / `--module=` / missing value, the `current` default and
  its absence, unknown modules, `-m` on a legacy file, modules mixed with top-level
  keys, duplicate or unnamed modules, `-m` forwarded to each project in collection
  mode, and the `examples/multi-module` project;
- `yaml_helper.py` output and exit codes directly, and rejection of invalid
  configurations (mixed lists, non-list `args`, mapping-valued `stdin`).

If `uv` is installed, the suite runs twice: once using `uv run`, and again with
`uv` removed from `PATH` to exercise the python3 fallback. If `uv` is installed
but can't provide PyYAML (for example, no network access), the suite says so
explicitly and runs only the python3 path, so the uv path is then **not** tested.

Last verified: 77/77 checks on the python3 path. The uv path was not exercised
in that run because the sandbox had no PyPI access.

## Examples

See `examples/README.md`. Briefly:

- `examples/greeter/` — a `run.sh` that asks for a name and a color. Its
  configuration supplies input in each form, and one run uses `--shout`.
- `examples/build-matrix/` — four build configurations, one of which fails on
  purpose, so you can see the failure report and the configurations that follow it.
- `examples/multi-module/` — two modules, `alpha` and `beta`, in one file. Run
  it with `-m alpha` or `-m beta`.

## Notes / limitations

- A project's `build` failing does **not** prevent its `run` stage from
  executing — they're independent per the config. Likewise, one failed
  configuration within a `build` or `run` list does **not** stop the
  remaining configurations in that same list from running. Check
  `assignments_results.txt` / `assignment_output.txt` for the real
  per-configuration outcome.
- Input is supplied up front; the runner can't pass the keyboard to a script
  or answer a prompt that depends on the script's output. Anything the script
  asks must be known in advance and written into `stdin`.
- Scripts that read from `/dev/tty` directly (rather than standard input) bypass
  `stdin` and will wait for a keyboard, which the runner doesn't provide.
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
- Module selection never guesses. With no `-m` and no module named `current`,
  nothing runs, and the error lists the modules. An unknown `-m` name is an
  error, and so is `-m` on a file without a `modules` key.
- A module's `folder` defaults to its `name`. A module named `current` therefore
  needs an explicit `folder` if its scripts are not in a folder called `current`.
- Boolean flag values are written as lowercase `true` / `false`, matching the
  YAML and the examples above. Earlier versions wrote Python's `True` / `False`.
