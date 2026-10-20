#!/usr/bin/env bash
#
# tests/run_tests.sh - tests for run_assignments.sh and yaml_helper.py
#
# Each test builds throwaway git projects in a temporary directory, runs
# run_assignments.sh against them, and checks the screen/file output, the
# status file, and which scripts actually executed (via calls.log). The
# example projects in examples/ are copied and checked too.
#
# Usage:  tests/run_tests.sh
#
# Requirements: bash 4.4+ or zsh 5.9+, git, python3. PyYAML is needed for the python3
# path; uv is optional. macOS ships bash 3.2 at /bin/bash, so on a Mac
# install a newer bash first (brew install bash) and run the suite with it.
#
# If uv is installed and can provide PyYAML, the suite runs twice: once
# through uv, and once with uv removed from PATH (the python3 fallback).

set -uo pipefail

# Runs under bash 4.4+ or zsh 5.9+. The runner itself (run_assignments.sh) is
# always run by bash, because it is a bash script.
if [ -n "${ZSH_VERSION:-}" ]; then
    zsh_major="${ZSH_VERSION%%.*}"
    zsh_minor="${ZSH_VERSION#*.}"
    zsh_minor="${zsh_minor%%[!0-9]*}"
    if [ "$zsh_major" -lt 5 ] || { [ "$zsh_major" -eq 5 ] && [ "$zsh_minor" -lt 9 ]; }; then
        echo "tests/run_tests.sh needs zsh 5.9 or newer; this is ${ZSH_VERSION}." >&2
        exit 1
    fi
elif [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
    echo "tests/run_tests.sh needs bash 4.4 or newer; this is ${BASH_VERSION}." >&2
    echo "On macOS: brew install bash, then run: $(brew --prefix 2>/dev/null || echo /opt/homebrew)/bin/bash tests/run_tests.sh" >&2
    exit 1
fi

# Path to this file, under either shell.
if [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_PATH="${(%):-%x}"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
TESTS_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ROOT_DIR="$(cd "$TESTS_DIR/.." && pwd)"
RUNNER="$ROOT_DIR/run_assignments.sh"
HELPER="$ROOT_DIR/yaml_helper.py"
EXAMPLES_DIR="$ROOT_DIR/examples"

PASS=0
FAIL=0
FAILED=()
WORK=""

# Output of the most recent runner invocation (with colors) and its exit code.
OUT=""
RC=0

# Absolute path of the bash that runs the runner. Resolved now, before the
# python3 pass changes PATH, so the runner is never run by a different bash.
RUNNER_BASH="${BASH:-$(command -v bash)}"

# Interpreter used to run yaml_helper.py directly. Set to the same command
# the runner uses (uv run ... or python3) by main().
HELPER_PY=(python3)

# ----------------------------------------------------------------------------
# Assertions
# ----------------------------------------------------------------------------

pass() {
    PASS=$((PASS + 1))
    printf '  ok    %s\n' "$1"
}

fail() {
    FAIL=$((FAIL + 1))
    FAILED+=("$1")
    printf '  FAIL  %s\n' "$1"
    [ -n "${2:-}" ] && printf '        %s\n' "$2"
}

assert_contains() {        # assert_contains <desc> <haystack> <needle>
    if printf '%s' "$2" | grep -qF -- "$3"; then pass "$1"
    else fail "$1" "missing: $3"; fi
}

assert_not_contains() {    # assert_not_contains <desc> <haystack> <needle>
    if printf '%s' "$2" | grep -qF -- "$3"; then fail "$1" "unexpected: $3"
    else pass "$1"; fi
}

assert_eq() {              # assert_eq <desc> <actual> <expected>
    if [ "$2" = "$3" ]; then pass "$1"
    else fail "$1" "got: $(printf '%q' "$2")  expected: $(printf '%q' "$3")"; fi
}

assert_true() {            # assert_true <desc> <command...>
    local desc="$1"; shift
    if "$@"; then pass "$desc"; else fail "$desc"; fi
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Run a command with a time limit (seconds first). Uses timeout or gtimeout
# when present. Otherwise falls back to a small watchdog, because macOS has
# neither by default. Setting RUNNER_TESTS_FORCE_WATCHDOG=1 forces the
# watchdog path, so it can be tested on Linux.
run_limited() {            # run_limited <seconds> <command...>
    local secs="$1"; shift
    if [ -z "${RUNNER_TESTS_FORCE_WATCHDOG:-}" ]; then
        if command -v timeout >/dev/null 2>&1; then timeout "$secs" "$@"; return $?; fi
        if command -v gtimeout >/dev/null 2>&1; then gtimeout "$secs" "$@"; return $?; fi
    fi
    "$@" &
    local pid=$!
    # The watchdog's output goes to /dev/null so a command substitution
    # around this call does not wait for it.
    ( sleep "$secs" && kill -9 "$pid" ) >/dev/null 2>&1 &
    local watchdog=$!
    wait "$pid"
    local rc=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return $rc
}

# True if the last runner call was killed for running too long.
not_timed_out() {
    [ "$RC" -ne 124 ] && [ "$RC" -ne 137 ]
}

# Fresh git repository on branch main with one empty commit.
make_project() {           # make_project <dir>
    mkdir -p "$1"
    git -C "$1" init -q
    git -C "$1" symbolic-ref HEAD refs/heads/main
    git -C "$1" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m init
}

write_file() {             # write_file <path>   (content on stdin)
    mkdir -p "$(dirname "$1")"
    cat > "$1"
}

write_script() {           # write_script <path>   (content on stdin)
    write_file "$1"
    chmod +x "$1"
}

# Logs each call to calls.log, echoes its arguments, and fails on --fail.
record_script() {          # record_script <path>
    write_script "$1" <<'SH'
#!/usr/bin/env bash
echo "$0 $*" >> calls.log
echo "ARGC=$# ARGS=[$*]"
for a in "$@"; do
    if [ "$a" = "--fail" ]; then echo "boom: requested failure"; exit 1; fi
done
echo "finished"
SH
}

# Reads a name and a color from stdin (like the greeter example).
greeter_script() {         # greeter_script <path>
    write_script "$1" <<'SH'
#!/usr/bin/env bash
printf 'Name? '
IFS= read -r name || { echo "no name given"; exit 1; }
printf 'Color? '
IFS= read -r color || { echo "no color given"; exit 1; }
echo "HELLO ${name} (${color}) args=[$*]"
SH
}

# run_runner <cwd> [runner args...]  -> sets OUT (with colors) and RC.
run_runner() {
    local dir="$1"; shift
    OUT="$(cd "$dir" && run_limited 120 "$RUNNER_BASH" "$RUNNER" "$@" < /dev/null 2>&1)"
    RC=$?
}

plain_output() {           # plain_output <project_dir>  (color-free log)
    cat "$1/assignment_output.txt" 2>/dev/null
}

calls_log() {              # calls_log <dir>  (what the scripts recorded)
    cat "$1/calls.log" 2>/dev/null
}

status_val() {             # status_val <project_dir> <KEY>
    grep -E "^$2=" "$1/.assignment_status.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'
}

# ----------------------------------------------------------------------------
# Tests: single project, dictionary and flat-list forms
# ----------------------------------------------------------------------------

test_dict_build_and_flat_run() {
    local p="$WORK/dict_and_flat"
    make_project "$p"
    record_script "$p/src/build.sh"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  target: release
  O: 2
  verbose: true
run:
  - input.txt
  - "42"
  - --dry-run
YAML
    run_runner "$p"
    assert_eq "runner exits 0" "$RC" "0"
    assert_eq "dict build and flat run each execute once, in order" \
        "$(calls_log "$p/src")" \
        "$(printf '%s\n%s' './build.sh --target release -O 2 --verbose true' './run.sh input.txt 42 --dry-run')"
    assert_not_contains "single configurations are not labelled [i/N]" "$(plain_output "$p")" "[1/"
    assert_eq "status counts are 1/1 for both stages" \
        "$(status_val "$p" BUILD_PASSED)/$(status_val "$p" BUILD_TOTAL)/$(status_val "$p" RUN_PASSED)/$(status_val "$p" RUN_TOTAL)" \
        "1/1/1/1"
}

test_multi_config_failure_in_middle() {
    local p="$WORK/middle_failure"
    make_project "$p"
    record_script "$p/src/build.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - {target: a}
  - [--fail]
  - target: c
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_eq "configs before and after the failure both run" \
        "$(calls_log "$p/src")" \
        "$(printf '%s\n%s\n%s' './build.sh --target a' './build.sh --fail' './build.sh --target c')"
    assert_contains "configuration 1 reported as completed" "$out" "BUILD [1/3] completed successfully"
    assert_contains "configuration 2 reported as failed with exit code" "$out" "BUILD [2/3] FAILED (exit code 1)"
    assert_contains "failed output is printed in uppercase" "$out" "BOOM: REQUESTED FAILURE"
    assert_contains "configuration 3 reported as completed" "$out" "BUILD [3/3] completed successfully"
    assert_contains "stage summary counts the passes" "$out" "BUILD summary: 2/3 configurations succeeded"
    assert_eq "status file records 2 of 3 passing" \
        "$(status_val "$p" BUILD_PASSED)/$(status_val "$p" BUILD_TOTAL)" "2/3"
    assert_eq "exit status unchanged by a failed configuration" "$RC" "0"
}

test_build_failure_does_not_block_run() {
    local p="$WORK/build_fail_run_ok"
    make_project "$p"
    record_script "$p/src/build.sh"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - [--fail]
run:
  - [input.txt]
YAML
    run_runner "$p"
    assert_eq "run.sh still executes after build.sh fails" \
        "$(calls_log "$p/src")" \
        "$(printf '%s\n%s' './build.sh --fail' './run.sh input.txt')"
}

test_empty_build_list_and_absent_sections() {
    local p="$WORK/empty_and_absent"
    make_project "$p"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build: []
YAML
    run_runner "$p"
    assert_contains "empty build section is reported" "$(plain_output "$p")" "defines no configurations"
    assert_eq "status shows build section present with 0 configs" \
        "$(status_val "$p" HAD_BUILD)/$(status_val "$p" BUILD_TOTAL)" "1/0"
    assert_contains "absent run section is reported as skipped" \
        "$(plain_output "$p")" "No 'run' section configured"
    assert_eq "absent run section recorded as not configured" "$(status_val "$p" HAD_RUN)" "0"
}

test_custom_folder() {
    local p="$WORK/custom_folder"
    make_project "$p"
    record_script "$p/scripts/tools/build.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: scripts/tools
build:
  x: 1
YAML
    run_runner "$p"
    assert_eq "scripts run from the configured folder" \
        "$(calls_log "$p/scripts/tools")" "./build.sh -x 1"
}

test_missing_script_is_reported() {
    local p="$WORK/missing_script"
    make_project "$p"
    record_script "$p/src/build.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - [--x]
run:
  - [input.txt]
YAML
    run_runner "$p"
    assert_contains "missing run.sh is reported" "$(plain_output "$p")" "'run.sh' was not found"
    assert_eq "status records the run configuration as not passed" \
        "$(status_val "$p" RUN_PASSED)/$(status_val "$p" RUN_TOTAL)" "0/1"
}

test_missing_config_is_a_warning() {
    local p="$WORK/no_config"
    make_project "$p"
    run_runner "$p"
    assert_eq "missing config does not fail the runner" "$RC" "0"
    assert_contains "missing config is reported" "$(plain_output "$p")" "not found for project"
    assert_eq "status records no config" "$(status_val "$p" HAD_CONFIG)" "0"
}

test_git_pull_on_main_failure_is_only_a_warning() {
    local p="$WORK/pull_main"
    make_project "$p"
    record_script "$p/src/build.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - {target: a}
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_contains "pull is attempted on main" "$out" "running git pull"
    assert_contains "failed pull (no remote) is a warning" "$out" "git pull failed"
    assert_eq "build still runs after a failed pull" "$(calls_log "$p/src")" "./build.sh --target a"
}

test_git_pull_skipped_on_feature_branch() {
    local p="$WORK/pull_feature"
    make_project "$p"
    git -C "$p" checkout -q -b feature
    record_script "$p/src/build.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - {target: a}
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_contains "pull skipped on a feature branch" "$out" "skipping git pull"
    assert_not_contains "no pull attempted on a feature branch" "$out" "running git pull"
}

test_project_dir_argument() {
    local p="$WORK/by_dir"
    make_project "$p"
    record_script "$p/src/build.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - {target: a}
YAML
    # Run from outside the project, passing the project directory.
    run_runner "$WORK" "$p"
    assert_eq "project directory argument runs the project" \
        "$(calls_log "$p/src")" "./build.sh --target a"
    assert_true "output written inside the project" [ -f "$p/assignment_output.txt" ]
    assert_true "current directory left untouched" [ ! -e "$WORK/assignment_output.txt" ]

    # Same, but with an absolute path to the config file.
    rm -f "$p/src/calls.log"
    run_runner "$WORK" "$p/assignment_config.yaml" "$p"
    assert_eq "absolute config path plus project directory also works" \
        "$(calls_log "$p/src")" "./build.sh --target a"
}

# ----------------------------------------------------------------------------
# Tests: stdin (input for scripts that read from standard input)
# ----------------------------------------------------------------------------

test_stdin_dict_form() {
    local p="$WORK/stdin_dict"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  stdin: [Ada, blue]
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_contains "script receives the configured input" "$out" "HELLO Ada (blue) args=[]"
    assert_contains "the supplied input is logged" "$out" "supplying 2 line(s) of stdin"
    assert_eq "run passes" "$(status_val "$p" RUN_PASSED)/$(status_val "$p" RUN_TOTAL)" "1/1"
}

test_stdin_with_args_list_form() {
    local p="$WORK/stdin_args"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - args: [input.txt, "42"]
    stdin: [Ada, blue]
YAML
    run_runner "$p"
    assert_contains "positional args and input both reach the script" \
        "$(plain_output "$p")" "HELLO Ada (blue) args=[input.txt 42]"
}

test_stdin_multiline_string() {
    local p="$WORK/stdin_string"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  stdin: "Ada\nblue"
YAML
    run_runner "$p"
    assert_contains "a multi-line string feeds one line per row" \
        "$(plain_output "$p")" "HELLO Ada (blue) args=[]"
}

test_flags_args_and_stdin_together() {
    local p="$WORK/stdin_flags"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - mode: quick
    args: [input.txt]
    stdin: [Ada, blue]
YAML
    run_runner "$p"
    assert_contains "flags come before positional args, then input" \
        "$(plain_output "$p")" "HELLO Ada (blue) args=[--mode quick input.txt]"
}

test_stdin_isolated_per_configuration() {
    local p="$WORK/stdin_isolated"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - {stdin: [Ada, blue]}
  - {stdin: [Grace, green]}
  - {stdin: [Linus, black]}
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_eq "each configuration receives only its own input (Ada once)" \
        "$(grep -c 'HELLO Ada' "$p/assignment_output.txt")" "1"
    assert_contains "second configuration gets its own input" "$out" "HELLO Grace (green)"
    assert_contains "third configuration gets its own input" "$out" "HELLO Linus (black)"
    assert_contains "stage summary for the three runs" "$out" "RUN summary: 3/3 configurations succeeded"
}

test_failed_configuration_does_not_stop_stdin_configurations() {
    local p="$WORK/stdin_after_fail"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - args: [no-input-supplied]
  - {stdin: [Ada, blue]}
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_contains "first configuration (no input) fails" "$out" "RUN [1/2] FAILED (exit code 1)"
    assert_contains "second configuration still runs with its input" "$out" "HELLO Ada (blue)"
    assert_contains "summary counts one pass" "$out" "RUN summary: 1/2 configurations succeeded"
}

test_no_stdin_gives_end_of_file_not_a_hang() {
    local p="$WORK/stdin_none"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [input.txt]
YAML
    run_runner "$p"
    assert_true "runner does not hang waiting for a prompt" not_timed_out
    # Failed output is printed upper-cased, hence the capitals below.
    assert_contains "script sees end-of-file" "$(plain_output "$p")" "NO NAME GIVEN"
    assert_not_contains "no stdin lines are logged" "$(plain_output "$p")" "supplying"
}

test_empty_stdin_list_is_same_as_no_stdin() {
    local p="$WORK/stdin_empty"
    make_project "$p"
    greeter_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  stdin: []
YAML
    run_runner "$p"
    assert_contains "empty stdin gives the script end-of-file" "$(plain_output "$p")" "NO NAME GIVEN"
}

# ----------------------------------------------------------------------------
# Tests: collection mode and the example projects
# ----------------------------------------------------------------------------

test_collection_mode_summary() {
    local c="$WORK/collection"
    mkdir -p "$c"

    make_project "$c/projA"
    record_script "$c/projA/src/build.sh"
    greeter_script "$c/projA/src/run.sh"
    write_file "$c/projA/assignment_config.yaml" <<'YAML'
folder: src
build:
  - {target: a}
  - {target: b}
run:
  - stdin: [Ada, blue]
YAML

    make_project "$c/projB"                       # no config file at all

    make_project "$c/projC"
    record_script "$c/projC/src/build.sh"
    write_file "$c/projC/assignment_config.yaml" <<'YAML'
folder: src
build:
  - [--fail]
YAML

    make_project "$c/projD"
    mkdir -p "$c/projD/src"                      # the configured folder must exist
    write_file "$c/projD/assignment_config.yaml" <<'YAML'
folder: src
build: []
YAML

    mkdir -p "$c/not_a_repo"                     # not a git project: ignored

    run_runner "$c"
    local results
    results="$(cat "$c/assignments_results.txt" 2>/dev/null)"
    assert_eq "one summary line per git project, in order" "$results" \
        "$(printf '%s\n' \
            'projA | config: yes | build: OK (2/2) | run: OK (1/1)' \
            'projB | config: no | build: none | run: none' \
            'projC | config: yes | build: FAILED (0/1) | run: none' \
            'projD | config: yes | build: none configured | run: none')"
    assert_not_contains "non-git folders are skipped" "$results" "not_a_repo"
    assert_true "status files are cleaned up" [ ! -e "$c/projA/.assignment_status.env" ]
}

test_example_greeter() {
    local p="$WORK/example_greeter"
    cp -R "$EXAMPLES_DIR/greeter" "$p"
    make_project "$p"
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_eq "example exits 0" "$RC" "0"
    assert_contains "Ada's greeting" "$out" "Hello, Ada! Your favorite color is blue."
    assert_contains "Grace's greeting is shouted" "$out" "HELLO, GRACE! YOUR FAVORITE COLOR IS GREEN."
    assert_contains "Linus's multi-line input" "$out" "Hello, Linus! Your favorite color is black."
    assert_contains "run summary is 3/3" "$out" "RUN summary: 3/3 configurations succeeded"
    assert_contains "build {} runs once with no arguments" "$out" "BUILD: executing build.sh"
}

test_example_build_matrix() {
    local p="$WORK/example_matrix"
    cp -R "$EXAMPLES_DIR/build-matrix" "$p"
    make_project "$p"
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_contains "failing configuration is reported" "$out" "BUILD [3/4] FAILED (exit code 3)"
    assert_contains "configuration after the failure still runs" "$out" "BUILD [4/4] completed successfully"
    assert_contains "summary is 3/4" "$out" "BUILD summary: 3/4 configurations succeeded"
    assert_eq "three configurations printed their completion line" \
        "$(grep -c 'build complete' "$p/assignment_output.txt" 2>/dev/null)" "3"
    assert_contains "no run section means no run stage" "$out" "No 'run' section configured"
}

# ----------------------------------------------------------------------------
# Tests: yaml_helper.py directly (output shapes and exit codes)
# ----------------------------------------------------------------------------

test_helper_output_shapes() {
    local d="$WORK/helper"
    mkdir -p "$d"
    write_file "$d/c.yaml" <<'YAML'
build:
  - {target: release, verbose: true, args: [extra one]}
run:
  - {stdin: ["", "line two"]}
  - [input.txt]
YAML
    assert_eq "--index: flags, then args, each NUL-terminated" \
        "$("${HELPER_PY[@]}" "$HELPER" "$d/c.yaml" build --index 0 | tr '\0' '|')" \
        "--target|release|--verbose|true|extra one|"
    assert_eq "--stdin-index keeps blank lines" \
        "$("${HELPER_PY[@]}" "$HELPER" "$d/c.yaml" run --stdin-index 0 | tr '\0' '|')" "|line two|"
    assert_eq "--stdin-index prints nothing when there is no stdin" \
        "$("${HELPER_PY[@]}" "$HELPER" "$d/c.yaml" build --stdin-index 0)" ""
    assert_eq "--count counts configurations" \
        "$("${HELPER_PY[@]}" "$HELPER" "$d/c.yaml" run --count)" "2"
    assert_eq "flat list of values has no stdin" \
        "$("${HELPER_PY[@]}" "$HELPER" "$d/c.yaml" run --stdin-index 1)" ""

    "${HELPER_PY[@]}" "$HELPER" "$d/c.yaml" run --index 9 >/dev/null 2>&1
    assert_eq "out-of-range index exits 2" "$?" "2"

    write_file "$d/nobuild.yaml" <<'YAML'
run: [x]
YAML
    "${HELPER_PY[@]}" "$HELPER" "$d/nobuild.yaml" build --count >/dev/null 2>&1
    assert_eq "missing section exits 3" "$?" "3"
}

# ----------------------------------------------------------------------------
# Tests: invalid configurations are rejected without running anything
# ----------------------------------------------------------------------------

test_mixed_list_rejected_other_section_still_runs() {
    local p="$WORK/mixed_list"
    make_project "$p"
    record_script "$p/src/build.sh"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
build:
  - {target: a}
run:
  - input.txt
  - {mode: quick}
YAML
    run_runner "$p"
    local out; out="$(plain_output "$p")"
    assert_contains "mixed list is reported as invalid" "$out" "cannot mix bare values"
    assert_eq "build section still runs" "$(calls_log "$p/src")" "./build.sh --target a"
    assert_eq "run section recorded as not run" "$(status_val "$p" HAD_RUN)" "0"
}

test_args_must_be_a_list() {
    local p="$WORK/bad_args"
    make_project "$p"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - args: input.txt
    stdin: [Ada]
YAML
    run_runner "$p"
    assert_contains "non-list args is reported" "$(plain_output "$p")" "'args' must be a list"
    assert_true "script was not executed" [ ! -e "$p/src/calls.log" ]
}

test_stdin_must_be_plain_values() {
    local p="$WORK/bad_stdin"
    make_project "$p"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  stdin:
    user: Ada
YAML
    run_runner "$p"
    assert_contains "mapping as stdin is reported" "$(plain_output "$p")" \
        "'stdin' must be a string or a list of values, not a mapping"
    assert_true "script was not executed" [ ! -e "$p/src/calls.log" ]
}

# ----------------------------------------------------------------------------
# Tests: modules (choosing with -m / --module, the "current" default, errors)
# ----------------------------------------------------------------------------

# Writes a three-module configuration: alpha (folder defaults to its name),
# current (the default, folder lib/current) and beta (folder lib/beta).
two_module_config() {      # two_module_config <path>
    write_file "$1" <<'YAML'
modules:
  - name: alpha
    run:
      - [one]
  - name: current
    folder: lib/current
    build:
      - [--fast]
  - name: beta
    folder: lib/beta
    run:
      - mode: quick
YAML
}

# Runs yaml_helper.py with the same interpreter as the runner. Sets
# HELPER_OUT (stdout, trailing newline removed) and HELPER_RC (exit code).
helper_out() {
    HELPER_OUT="$("${HELPER_PY[@]}" "$HELPER" "$@" 2>/dev/null)"
    HELPER_RC=$?
}

test_module_flag_selects_module() {
    local p="$WORK/module_flag"
    make_project "$p"
    record_script "$p/alpha/run.sh"
    record_script "$p/lib/current/build.sh"
    record_script "$p/lib/beta/run.sh"
    two_module_config "$p/assignment_config.yaml"
    run_runner "$p" --module beta
    assert_eq "runner exits 0" "$RC" "0"
    assert_eq "only beta's run.sh executed, with its flags" \
        "$(calls_log "$p/lib/beta")" "./run.sh --mode quick"
    assert_eq "alpha did not run" "$(calls_log "$p/alpha")" ""
    assert_eq "current's build did not run" "$(calls_log "$p/lib/current")" ""
    assert_contains "log names the chosen module" "$(plain_output "$p")" "Using module 'beta'"
}

test_current_module_is_the_default() {
    local p="$WORK/current_default"
    make_project "$p"
    record_script "$p/lib/current/build.sh"
    record_script "$p/alpha/run.sh"
    record_script "$p/lib/beta/run.sh"
    two_module_config "$p/assignment_config.yaml"
    run_runner "$p"
    assert_eq "runner exits 0" "$RC" "0"
    assert_eq "current's build ran with no -m" "$(calls_log "$p/lib/current")" "./build.sh --fast"
    assert_eq "alpha and beta did not run" \
        "$(calls_log "$p/alpha")$(calls_log "$p/lib/beta")" ""
}

test_project_folder_name_is_not_a_default() {
    # The project folder is named "alpha", and a module is named alpha. The
    # run must still stop: only a module named "current" is the default.
    local p="$WORK/alpha"
    make_project "$p"
    record_script "$p/alpha/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
modules:
  - name: alpha
    run:
      - [one]
  - name: beta
    run:
      - [two]
YAML
    run_runner "$p"
    assert_eq "runner exits 1 when no default can be chosen" "$RC" "1"
    assert_contains "error explains the missing default" "$(plain_output "$p")" \
        "no module named 'current' to use by default"
    assert_contains "error lists the modules to choose from" "$(plain_output "$p")" \
        "use --module with one of: alpha, beta"
    assert_eq "no script ran" "$(calls_log "$p/alpha")" ""
}

test_unknown_module_is_an_error() {
    local p="$WORK/unknown_module"
    make_project "$p"
    record_script "$p/alpha/run.sh"
    two_module_config "$p/assignment_config.yaml"
    run_runner "$p" -m gamma
    assert_eq "runner exits 1" "$RC" "1"
    assert_contains "error names the missing module and lists the others" \
        "$(plain_output "$p")" "no module named 'gamma'; available modules: alpha, current, beta"
    assert_eq "no script ran" "$(calls_log "$p/alpha")" ""
}

test_module_flag_forms_and_bad_usage() {
    local p="$WORK/flag_forms"
    make_project "$p"
    record_script "$p/lib/beta/run.sh"
    two_module_config "$p/assignment_config.yaml"

    run_runner "$p" --module=beta
    assert_eq "--module=NAME works" "$RC" "0"
    assert_eq "--module=beta ran beta" "$(calls_log "$p/lib/beta")" "./run.sh --mode quick"

    rm -f "$p/lib/beta/calls.log"
    run_runner "$p" -m beta
    assert_eq "-m NAME works" "$RC" "0"
    assert_eq "-m beta ran beta" "$(calls_log "$p/lib/beta")" "./run.sh --mode quick"

    run_runner "$p" -m
    assert_eq "-m with no value is rejected" "$RC" "1"
    assert_contains "says -m needs a module name" "$OUT" "needs a module name"

    run_runner "$p" --bogus
    assert_eq "unknown option is rejected" "$RC" "1"
    assert_contains "names the unknown option" "$OUT" "Unknown option '--bogus'"
}

test_module_flag_with_legacy_config_is_rejected() {
    local p="$WORK/flag_legacy"
    make_project "$p"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [input.txt]
YAML
    run_runner "$p" -m anything
    assert_eq "runner exits 1" "$RC" "1"
    assert_contains "explains there is no modules key" "$(plain_output "$p")" \
        "has no 'modules' key"
    assert_eq "nothing ran" "$(calls_log "$p/src")" ""
}

test_modules_mixed_with_top_level_keys_is_rejected() {
    local p="$WORK/mixed_keys"
    make_project "$p"
    record_script "$p/alpha/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
run:
  - [top]
modules:
  - name: alpha
    run:
      - [one]
YAML
    run_runner "$p" -m alpha
    assert_eq "runner exits 1" "$RC" "1"
    assert_contains "explains the mix is not allowed" "$(plain_output "$p")" "not both"
    assert_eq "nothing ran" "$(calls_log "$p/alpha")" ""
}

test_duplicate_and_unnamed_modules_are_rejected() {
    local p="$WORK/bad_modules"
    make_project "$p"
    write_file "$p/assignment_config.yaml" <<'YAML'
modules:
  - name: current
    run:
      - [one]
  - name: current
    run:
      - [two]
YAML
    run_runner "$p"
    assert_eq "duplicate names: runner exits 1" "$RC" "1"
    assert_contains "duplicate names are named" "$(plain_output "$p")" "duplicate module name(s): current"

    write_file "$p/assignment_config.yaml" <<'YAML'
modules:
  - folder: src
    run:
      - [one]
YAML
    run_runner "$p"
    assert_eq "unnamed module: runner exits 1" "$RC" "1"
    assert_contains "unnamed module is reported" "$(plain_output "$p")" "needs a non-empty 'name'"
}

test_collection_forwards_module_flag() {
    local parent="$WORK/collection_modules"
    mkdir -p "$parent"
    local proj
    for proj in p1 p2; do
        make_project "$parent/$proj"
        two_module_config "$parent/$proj/assignment_config.yaml"
        record_script "$parent/$proj/alpha/run.sh"
        record_script "$parent/$proj/lib/beta/run.sh"
    done
    run_runner "$parent" --module beta
    assert_eq "collection run exits 0" "$RC" "0"
    assert_eq "p1 ran beta" "$(calls_log "$parent/p1/lib/beta")" "./run.sh --mode quick"
    assert_eq "p2 ran beta" "$(calls_log "$parent/p2/lib/beta")" "./run.sh --mode quick"
    assert_eq "alpha did not run in p1 or p2" \
        "$(calls_log "$parent/p1/alpha")$(calls_log "$parent/p2/alpha")" ""
    assert_contains "results list p1 with its config" "$(cat "$parent/assignments_results.txt")" \
        "p1 | config: yes"
}

test_helper_module_queries() {
    local y="$WORK/helper_modules.yaml"
    two_module_config "$y"

    helper_out "$y" select-module
    assert_eq "select-module defaults to current" "$HELPER_OUT" "current"
    assert_eq "select-module default exits 0" "$HELPER_RC" "0"

    helper_out "$y" select-module --module beta
    assert_eq "select-module --module beta" "$HELPER_OUT" "beta"

    helper_out "$y" select-module --module gamma
    assert_eq "select-module with unknown module exits 2" "$HELPER_RC" "2"

    helper_out "$y" folder --module alpha
    assert_eq "folder defaults to the module name" "$HELPER_OUT" "alpha"

    helper_out "$y" folder --module beta
    assert_eq "folder uses the module's folder key" "$HELPER_OUT" "lib/beta"

    helper_out "$y" build --module current --count
    assert_eq "build --count for a module" "$HELPER_OUT" "1"

    helper_out "$y" run --module beta --count
    assert_eq "run --count for a module" "$HELPER_OUT" "1"

    helper_out "$y" run --count
    assert_eq "current has no run section: exit 3" "$HELPER_RC" "3"

    local top="$WORK/helper_top.yaml"
    write_file "$top" <<'YAML'
folder: src
build:
  - [--x]
YAML
    helper_out "$top" select-module
    assert_eq "top-level config selects nothing" "$HELPER_OUT" ""
    assert_eq "top-level select-module exits 0" "$HELPER_RC" "0"

    helper_out "$top" select-module --module alpha
    assert_eq "--module on a top-level config exits 2" "$HELPER_RC" "2"
}

test_example_multi_module() {
    local p="$WORK/example_multi_module"
    cp -R "$EXAMPLES_DIR/multi-module" "$p"
    make_project "$p"

    run_runner "$p"
    assert_eq "without -m and no current module: exits 1" "$RC" "1"
    assert_contains "asks for --module" "$(plain_output "$p")" "use --module with one of: alpha, beta"

    run_runner "$p" -m alpha
    local out; out="$(plain_output "$p")"
    assert_eq "-m alpha exits 0" "$RC" "0"
    assert_contains "alpha's run got its stdin and args" "$out" "alpha run: args=[one] hello Ada"
    assert_contains "alpha's build ran both configurations" "$out" "BUILD summary: 2/2 configurations succeeded"
    assert_not_contains "beta did not run" "$out" "beta run"

    run_runner "$p" -m beta
    out="$(plain_output "$p")"
    assert_eq "-m beta exits 0" "$RC" "0"
    assert_contains "beta's run executed" "$out" "beta run: --mode quick"
    assert_not_contains "alpha did not run" "$out" "alpha run"
}

# ----------------------------------------------------------------------------
# Tests: per-project time limit in collection runs (-t / --timeout)
# ----------------------------------------------------------------------------

test_timeout_option_and_defaults() {
    local parent="$WORK/timeout_options"
    mkdir -p "$parent"
    make_project "$parent/p1"
    record_script "$parent/p1/src/run.sh"
    write_file "$parent/p1/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [x]
YAML

    run_runner "$parent"
    assert_eq "default collection run exits 0" "$RC" "0"
    assert_contains "default time limit is 10 minutes" "$OUT" "Time limit per project: 10 minute(s)"

    run_runner "$parent" -t 3
    assert_contains "-t sets the limit in minutes" "$OUT" "Time limit per project: 3 minute(s)"

    run_runner "$parent" --timeout=5
    assert_contains "--timeout=N sets the limit" "$OUT" "Time limit per project: 5 minute(s)"

    run_runner "$parent" -t 0
    assert_eq "-t 0 is rejected" "$RC" "1"
    assert_contains "says it must be greater than 0" "$OUT" "greater than 0"

    run_runner "$parent" -t abc
    assert_eq "-t with a non-number is rejected" "$RC" "1"

    run_runner "$parent" -t 1.5
    assert_eq "-t with a fraction is rejected" "$RC" "1"

    run_runner "$parent" -t
    assert_eq "-t with no value is rejected" "$RC" "1"
    assert_contains "says -t needs a number" "$OUT" "needs a number of minutes"
}

test_timeout_ignored_with_warning_for_single_project() {
    local p="$WORK/timeout_single"
    make_project "$p"
    record_script "$p/src/run.sh"
    write_file "$p/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [x]
YAML
    run_runner "$p" -t 2
    assert_eq "single project still runs" "$(calls_log "$p/src")" "./run.sh x"
    assert_contains "warns that -t is for collection runs" "$OUT" "applies to collection runs only"
}

test_timeout_stops_hung_project_and_continues() {
    local parent="$WORK/timeout_hung"
    mkdir -p "$parent"
    make_project "$parent/hung"
    write_script "$parent/hung/src/run.sh" <<'SH'
#!/usr/bin/env bash
echo "started" >> calls.log
sleep 61
echo "should not get here"
SH
    write_file "$parent/hung/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [x]
YAML
    make_project "$parent/ok"
    record_script "$parent/ok/src/run.sh"
    write_file "$parent/ok/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [x]
YAML

    export RUNNER_TIMEOUT_SECONDS=2
    local started=$SECONDS
    run_runner "$parent"
    local elapsed=$((SECONDS - started))
    unset RUNNER_TIMEOUT_SECONDS

    assert_eq "collection run exits 0 after a timeout" "$RC" "0"
    assert_contains "hung project is recorded as timed out" \
        "$(cat "$parent/assignments_results.txt")" "hung | TIMED OUT"
    assert_eq "the hung project started its script" "$(calls_log "$parent/hung/src")" "started"
    assert_contains "the next project still ran" "$(cat "$parent/assignments_results.txt")" \
        "ok | config: yes"
    assert_eq "the project after the timeout ran" "$(calls_log "$parent/ok/src")" "./run.sh x"
    assert_true "the run stopped well before the script's 61 seconds ($elapsed s)" \
        test "$elapsed" -lt 30
    sleep 1
    assert_eq "no sleep from the stopped script is left running" \
        "$(ps -eo args 2>/dev/null | grep -c '[s]leep 61')" "0"
}

test_fast_project_is_not_timed_out() {
    local parent="$WORK/timeout_fast"
    mkdir -p "$parent"
    make_project "$parent/quick"
    record_script "$parent/quick/src/run.sh"
    write_file "$parent/quick/assignment_config.yaml" <<'YAML'
folder: src
run:
  - [x]
YAML
    export RUNNER_TIMEOUT_SECONDS=30
    run_runner "$parent"
    unset RUNNER_TIMEOUT_SECONDS
    assert_eq "collection run exits 0" "$RC" "0"
    assert_contains "fast project completes normally" "$(cat "$parent/assignments_results.txt")" \
        "quick | config: yes | build: none | run: OK (1/1)"
    assert_not_contains "no timeout recorded" "$(cat "$parent/assignments_results.txt")" "TIMED OUT"
}

# ----------------------------------------------------------------------------
# Test list and runner
# ----------------------------------------------------------------------------

TESTS=(
    test_dict_build_and_flat_run
    test_multi_config_failure_in_middle
    test_build_failure_does_not_block_run
    test_empty_build_list_and_absent_sections
    test_custom_folder
    test_missing_script_is_reported
    test_missing_config_is_a_warning
    test_git_pull_on_main_failure_is_only_a_warning
    test_git_pull_skipped_on_feature_branch
    test_project_dir_argument
    test_stdin_dict_form
    test_stdin_with_args_list_form
    test_stdin_multiline_string
    test_flags_args_and_stdin_together
    test_stdin_isolated_per_configuration
    test_failed_configuration_does_not_stop_stdin_configurations
    test_no_stdin_gives_end_of_file_not_a_hang
    test_empty_stdin_list_is_same_as_no_stdin
    test_collection_mode_summary
    test_example_greeter
    test_example_build_matrix
    test_helper_output_shapes
    test_mixed_list_rejected_other_section_still_runs
    test_args_must_be_a_list
    test_stdin_must_be_plain_values
    test_module_flag_selects_module
    test_current_module_is_the_default
    test_project_folder_name_is_not_a_default
    test_unknown_module_is_an_error
    test_module_flag_forms_and_bad_usage
    test_module_flag_with_legacy_config_is_rejected
    test_modules_mixed_with_top_level_keys_is_rejected
    test_duplicate_and_unnamed_modules_are_rejected
    test_collection_forwards_module_flag
    test_helper_module_queries
    test_example_multi_module
    test_timeout_option_and_defaults
    test_timeout_ignored_with_warning_for_single_project
    test_timeout_stops_hung_project_and_continues
    test_fast_project_is_not_timed_out
)

run_all_tests() {
    WORK="$(mktemp -d)"
    local t
    for t in "${TESTS[@]}"; do
        echo "$t"
        "$t"
    done
    rm -rf "$WORK"
}

# Runs the suite and prints the summary. Returns 0 only if every check passed.
run_suite_and_report() {
    run_all_tests
    echo
    echo "passed: $PASS   failed: $FAIL"
    if [ "$FAIL" -gt 0 ]; then
        local name
        for name in "${FAILED[@]}"; do echo "  - $name"; done
        return 1
    fi
    return 0
}

# True if `uv run` can supply PyYAML for this project (needs network or a
# warm uv cache on first use).
uv_can_provide_pyyaml() {
    command -v uv >/dev/null 2>&1 || return 1
    uv run --project "$ROOT_DIR" python3 -c "import yaml" >/dev/null 2>&1
}

python3_has_yaml() {
    python3 -c "import yaml" >/dev/null 2>&1
}

# Builds the PATH for a child run in which uv cannot be found. Directories
# without uv are kept exactly as they are. That matters for virtualenv bin
# directories: Python finds pyvenv.cfg from the directory of the executable it
# was started through, so a symlink to a venv's python3 in another directory
# silently loses the venv (and PyYAML with it). A directory that does contain
# uv is replaced by a directory of symlinks to its other executables, so bash,
# python3 and git there still resolve. The shim directories are created under
# $1, which the caller removes afterwards. Prints the new PATH.
# PATH is split with a here-string, not word splitting, because zsh does not
# split unquoted parameter expansions.
make_path_without_uv() {
    local root="$1" entry n=0 result="" shim src name
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        if [ -x "$entry/uv" ]; then
            n=$((n + 1))
            shim="$root/$n"
            mkdir -p "$shim"
            while IFS= read -r src; do
                name="$(basename "$src")"
                [ "$name" = "uv" ] && continue
                [ -e "$shim/$name" ] || ln -s "$src" "$shim/$name"
            done < <(find -L "$entry" -maxdepth 1 -type f -perm -100 2>/dev/null)
            result="${result:+$result:}$shim"
        else
            result="${result:+$result:}$entry"
        fi
    done <<< "$(printf '%s\n' "$PATH" | tr ':' '\n')"
    printf '%s' "$result"
}

# Runs the whole suite in a child process with uv hidden, so the runner and
# the helper both use the plain python3 path. The shell is resolved to an
# absolute path first, before PATH changes.
run_python3_pass() {
    local self_shell root new_path rc
    if [ -n "${ZSH_VERSION:-}" ]; then
        self_shell="$(command -v zsh)"
    else
        self_shell="${BASH:-$(command -v bash)}"
    fi
    root="$(mktemp -d)"
    new_path="$(make_path_without_uv "$root")"
    PATH="$new_path" RUNNER_TESTS_CHILD=1 "$self_shell" "$TESTS_DIR/run_tests.sh"
    rc=$?
    rm -rf "$root"
    return $rc
}

main() {
    echo "Testing: $RUNNER"
    echo "shell:   ${ZSH_VERSION:+zsh $ZSH_VERSION}${BASH_VERSION:+bash $BASH_VERSION}"
    echo

    # Child process: uv is hidden and python3 is the only path.
    if [ -n "${RUNNER_TESTS_CHILD:-}" ]; then
        HELPER_PY=(python3)
        run_suite_and_report
        return $?
    fi

    local uv_usable=0
    if command -v uv >/dev/null 2>&1; then
        if uv_can_provide_pyyaml; then
            uv_usable=1
        else
            echo "SKIP  uv is installed but cannot provide PyYAML here (no network or PyPI access?)."
            echo "      The uv code path is NOT tested in this run."
            echo
        fi
    fi

    if [ "$uv_usable" -eq 1 ]; then
        local overall=0
        HELPER_PY=(uv run --project "$ROOT_DIR" python3)
        echo "== Pass 1 of 2: through uv run =="
        run_suite_and_report || overall=1
        echo
        if python3_has_yaml; then
            echo "== Pass 2 of 2: python3 fallback =="
            run_python3_pass || overall=1
        else
            echo "SKIP  python3 has no PyYAML, so the python3 fallback pass was not run."
            echo "      Install it (pip3 install pyyaml) to test that path too."
        fi
        return $overall
    fi

    # No usable uv: test the python3 path.
    if ! python3_has_yaml; then
        echo "NOT RUN  python3 has no PyYAML, so the suite cannot run."
        echo "         Install it (pip3 install pyyaml) or make uv usable, then rerun."
        return 2
    fi
    if command -v uv >/dev/null 2>&1; then
        # uv is installed but unusable: hide it so the runner uses python3.
        run_python3_pass
        return $?
    fi
    HELPER_PY=(python3)
    run_suite_and_report
    return $?
}

main "$@"
exit $?
