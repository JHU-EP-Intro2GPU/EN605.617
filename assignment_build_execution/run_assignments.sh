#!/usr/bin/env bash
#
# run_assignments.sh
#
# Runs a project's build.sh / run.sh as configured by an
# assignment_config.yaml file.
#
# Usage:
#   ./run_assignments.sh [-m|--module NAME] [config_file.yaml] [project_dir]
#   ./run_assignments.sh --help
#
#   config_file.yaml defaults to "assignment_config.yaml".
#   -m, --module NAME   Run the module called NAME in the config file. Also
#                       accepted as --module=NAME.
#   Positional arguments are recognized by type (order does not matter): an
#   existing directory is treated as project_dir; anything else is the config
#   file.
#
# Behavior:
#   - If a project_dir is passed, that directory is used as the project
#     root and single-project mode runs there (build/run via
#     run_single_project), regardless of the current working directory.
#
#   - If run inside a git project (a directory containing .git), it will:
#       1. choose the configuration to use (see "modules" in the README)
#       2. git reset --hard && git pull (only if currently on main/master)
#       3. run build.sh (if a "build" section exists in the configuration)
#       4. run run.sh   (if a "run" section exists in the configuration)
#     ...using the folder / flags / stdin described in the configuration,
#     logging everything to screen (in color) and to assignment_output.txt.
#     A configuration's stdin lines are fed to the script; a configuration
#     without stdin gets an empty input, so the script sees end-of-file
#     instead of waiting on the terminal.
#
#   - If run inside a folder that is NOT itself a git project (i.e. a
#     folder that contains a collection of git project subfolders), it
#     will recurse into each subfolder that IS a git project, run this
#     same script there (passing --module along), and summarize all
#     results into assignments_results.txt.
#
# Requires: bash 4.4+, python3, PyYAML (pip3 install pyyaml), or uv.

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
    echo "run_assignments.sh needs bash 4.4 or newer; this is ${BASH_VERSION}." >&2
    echo "On macOS, /bin/bash is version 3.2. Install a newer one with: brew install bash" >&2
    exit 1
fi

set -uo pipefail

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
SCRIPT_FULL_PATH="$SCRIPT_DIR/$(basename "$SCRIPT_SOURCE")"
HELPER="$SCRIPT_DIR/yaml_helper.py"

# Parsed in Main: optional config file, project directory, and module name.
CONFIG_FILE="assignment_config.yaml"
TARGET_DIR=""
MODULE=""

# Time limit for each project in a collection run, in whole minutes.
TIMEOUT_MINUTES=10
TIMEOUT_GIVEN=0

# Extra arguments for yaml_helper.py queries: --module NAME once a module is
# chosen, and empty for a top-level configuration.
MODULE_ARGS=()

# Colors / styles
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

OUTPUT_FILE="assignment_output.txt"
STATUS_FILE=".assignment_status.env"

# Prefer `uv run` (uses pyproject.toml alongside this script to manage the
# PyYAML dependency in an isolated environment). Falls back to a plain
# python3 that already has PyYAML installed if uv isn't available.
if command -v uv >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PY_RUNNER=(uv run --project "$SCRIPT_DIR" python3)
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}❌ python3 is required but was not found on PATH.${NC}"
        exit 1
    fi
    if ! python3 -c "import yaml" >/dev/null 2>&1; then
        echo -e "${RED}❌ PyYAML is required. Either install uv (https://docs.astral.sh/uv/), or run: pip3 install pyyaml${NC}"
        exit 1
    fi
    PY_RUNNER=(python3)
fi

# ----------------------------------------------------------------------------
# Logging helpers — write colored text to the screen AND a plain-text
# (color-stripped) copy to assignment_output.txt
# ----------------------------------------------------------------------------

_strip_colors() {
    sed -E $'s/\x1b\\[[0-9;]*m//g'
}

log() {
    # $1 = message, with color codes already embedded
    echo -e "$1"
    echo -e "$1" | _strip_colors >> "$OUTPUT_FILE"
}

log_header()  { log "${BOLD}${BLUE}🚀 $*${NC}"; }
log_info()    { log "${BLUE}ℹ️  $*${NC}"; }
log_success() { log "${GREEN}✅ $*${NC}"; }
log_warn()    { log "${YELLOW}⚠️  $*${NC}"; }
log_error()   { log "${RED}❌ $*${NC}"; }

usage() {
    cat <<'EOF'
Usage: run_assignments.sh [-m|--module NAME] [config_file.yaml] [project_dir]

  -m, --module NAME   Run the module NAME from the config file (--module=NAME also works).
                      Without it, the module named after the project folder is used.
  config_file.yaml    Defaults to assignment_config.yaml.
  project_dir         Run in this project instead of the current directory.
  -t, --timeout MIN   Time limit for each project in a collection run, in whole
                      minutes (default 10). A project that runs longer is stopped,
                      recorded as TIMED OUT, and the next project starts.
  -h, --help          Show this help.
EOF
}

# ----------------------------------------------------------------------------
# git reset --hard && git pull — pull only if currently on main/master
# ----------------------------------------------------------------------------

do_git_reset_pull() {
    if [ ! -d .git ]; then
        return
    fi

    local branch
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    local reset_hard
    reset_hard="$(git reset --hard)"
    log "$reset_hard"
    if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
        log_info "On branch '${branch}' — running git pull..."
        local pull_out
        pull_out="$(git pull 2>&1)"
        if [ $? -eq 0 ]; then
            log_success "git pull complete"
            log "$pull_out"
        else
            log_warn "git pull failed:"
            log "$pull_out"
        fi
    else
        log_warn "Current branch is '${branch:-unknown}', not main/master — skipping git pull."
    fi
}

# ----------------------------------------------------------------------------
# Run a single execution of a stage (one build or run configuration)
#   $1 = stage label ("BUILD" / "RUN")
#   $2 = script filename ("build.sh" / "run.sh")
#   $3 = working directory to execute it from
#   $4 = this configuration's 1-based index (e.g. "1")
#   $5 = total number of configurations for this stage (e.g. "3")
#   $6 = file holding this configuration's stdin lines ("" or an empty
#        file means no input: the script gets /dev/null)
#   $@ (remaining) = args to pass to the script for this configuration
# Returns 0 on success, non-zero on failure. Never aborts the caller —
# each configuration is independent, so the caller is expected to keep
# looping over the rest even when this returns non-zero.
# ----------------------------------------------------------------------------

run_stage() {
    local stage_name="$1"
    local script_name="$2"
    local work_dir="$3"
    local cfg_index="$4"
    local cfg_total="$5"
    local stdin_file="$6"
    shift 6
    local args=("$@")

    local label="$stage_name"
    if [ "$cfg_total" -gt 1 ]; then
        label="${stage_name} [${cfg_index}/${cfg_total}]"
    fi

    local script_path="$work_dir/$script_name"

    if [ ! -f "$script_path" ]; then
        log_error "${label}: stage is configured but '${script_name}' was not found in '${work_dir}'"
        return 2
    fi

    log_header "${label}: executing ${script_name} ${args[*]:-}"

    # Standard input for the script. Lines are fed from the configured stdin
    # file; with no stdin the script reads /dev/null (end-of-file), so a
    # prompt the configuration did not anticipate fails instead of hanging.
    local stdin_src="/dev/null"
    if [ -n "$stdin_file" ] && [ -s "$stdin_file" ]; then
        stdin_src="$stdin_file"
        log_info "${label}: supplying $(wc -l < "$stdin_file" | tr -d ' ') line(s) of stdin:"
        local input_line
        while IFS= read -r input_line; do
            log "${BLUE}     ⏎ ${input_line}${NC}"
        done < "$stdin_file"
    fi

    local tmp_out
    tmp_out="$(mktemp)"

    ( cd "$work_dir" && bash "./${script_name}" "${args[@]}" ) < "$stdin_src" > "$tmp_out" 2>&1
    local status=$?

    local content
    content="$(cat "$tmp_out")"
    rm -f "$tmp_out"

    if [ $status -eq 0 ]; then
        log_success "${label} completed successfully"
        [ -n "$content" ] && log "${GREEN}${content}${NC}"
    else
        log_error "${label} FAILED (exit code ${status})"
        local caps
        caps="$(printf '%s' "$content" | tr '[:lower:]' '[:upper:]')"
        [ -n "$caps" ] && log "${BOLD}${RED}${caps}${NC}"
    fi

    return $status
}

# ----------------------------------------------------------------------------
# Run every independent configuration for a stage (build or run), in
# order, continuing regardless of individual failures.
#   $1 = stage label ("BUILD" / "RUN")
#   $2 = script filename ("build.sh" / "run.sh")
#   $3 = work_dir
#   $4 = config_file
#   $5 = section ("build" / "run")
# Uses the global MODULE_ARGS to select the module.
# Sets (via global out-vars, since bash can't return multiple values
# cleanly): STAGE_HAD=1/0, STAGE_TOTAL=<n>, STAGE_PASSED=<n>
# ----------------------------------------------------------------------------

run_all_configs_for_stage() {
    local stage_name="$1"      # "BUILD" / "RUN"
    local script_name="$2"     # "build.sh" / "run.sh"
    local work_dir="$3"
    local config_file="$4"
    local section="$5"         # "build" / "run"

    STAGE_HAD=0
    STAGE_TOTAL=0
    STAGE_PASSED=0

    local count_file yerr
    count_file="$(mktemp)"
    yerr="$(mktemp)"
    "${PY_RUNNER[@]}" "$HELPER" "$config_file" "$section" "${MODULE_ARGS[@]}" --count > "$count_file" 2>"$yerr"
    local rc=$?

    if [ $rc -eq 3 ]; then
        log_info "No '${section}' section configured — skipping ${stage_name,,} stage."
        rm -f "$count_file" "$yerr"
        return
    elif [ $rc -ne 0 ]; then
        log_error "Failed to parse the '${section}' section of '${config_file}': $(cat "$yerr")"
        rm -f "$count_file" "$yerr"
        return
    fi
    rm -f "$yerr"

    STAGE_HAD=1
    STAGE_TOTAL="$(cat "$count_file")"
    rm -f "$count_file"

    if [ "$STAGE_TOTAL" -eq 0 ]; then
        log_info "'${section}' section is present but defines no configurations — nothing to ${stage_name,,}."
        return
    fi

    local i
    for (( i=0; i<STAGE_TOTAL; i++ )); do
        # Command-line arguments for configuration i (NUL-separated tokens).
        local args_file
        args_file="$(mktemp)"
        "${PY_RUNNER[@]}" "$HELPER" "$config_file" "$section" "${MODULE_ARGS[@]}" --index "$i" > "$args_file"
        local args=()
        readarray -d '' args < "$args_file"
        rm -f "$args_file"

        # Stdin lines for configuration i (NUL-separated). They are written
        # to a file rather than kept in a bash variable, so blank lines and
        # embedded newlines survive intact.
        local items_file stdin_items=() stdin_file
        items_file="$(mktemp)"
        "${PY_RUNNER[@]}" "$HELPER" "$config_file" "$section" "${MODULE_ARGS[@]}" --stdin-index "$i" > "$items_file"
        readarray -d '' stdin_items < "$items_file"
        rm -f "$items_file"

        stdin_file="$(mktemp)"
        if [ "${#stdin_items[@]}" -gt 0 ]; then
            printf '%s\n' "${stdin_items[@]}" > "$stdin_file"
        fi

        if run_stage "$stage_name" "$script_name" "$work_dir" "$((i + 1))" "$STAGE_TOTAL" \
                "$stdin_file" "${args[@]}"; then
            STAGE_PASSED=$((STAGE_PASSED + 1))
        fi
        rm -f "$stdin_file"
        # Deliberately no early exit here: each configuration is
        # independent, so a failure never stops the remaining ones.
    done

    if [ "$STAGE_TOTAL" -gt 1 ]; then
        if [ "$STAGE_PASSED" -eq "$STAGE_TOTAL" ]; then
            log_success "${stage_name} summary: ${STAGE_PASSED}/${STAGE_TOTAL} configurations succeeded"
        else
            log_error "${stage_name} summary: ${STAGE_PASSED}/${STAGE_TOTAL} configurations succeeded"
        fi
    fi
}

# ----------------------------------------------------------------------------
# Write the machine-readable status file a parent invocation will read
# ----------------------------------------------------------------------------

write_status_file() {
    local project_name="$1" had_config="$2"
    local had_build="$3" build_total="$4" build_passed="$5"
    local had_run="$6" run_total="$7" run_passed="$8"
    cat > "$STATUS_FILE" <<EOF
PROJECT_NAME="${project_name}"
HAD_CONFIG=${had_config}
HAD_BUILD=${had_build}
BUILD_TOTAL=${build_total}
BUILD_PASSED=${build_passed}
HAD_RUN=${had_run}
RUN_TOTAL=${run_total}
RUN_PASSED=${run_passed}
EOF
}

# ----------------------------------------------------------------------------
# Single project mode — this directory contains .git
# ----------------------------------------------------------------------------

run_single_project() {
    local config_file="$1"
    local project_name
    project_name="$(basename "$PWD")"

    : > "$OUTPUT_FILE"

    local had_config=0
    local had_build=0 build_total=0 build_passed=0
    local had_run=0 run_total=0 run_passed=0

    log_header "Project: ${project_name}"

    if [ ! -f "$config_file" ]; then
        log_warn "Configuration file '${config_file}' not found for project '${project_name}'. Skipping."
        write_status_file "$project_name" "$had_config" \
            "$had_build" "$build_total" "$build_passed" \
            "$had_run" "$run_total" "$run_passed"
        return 0
    fi
    had_config=1

    # Choose the module (or the top-level configuration) before touching git,
    # so an invalid choice never pulls or runs anything.
    local selected yerr
    yerr="$(mktemp)"
    selected="$("${PY_RUNNER[@]}" "$HELPER" "$config_file" select-module \
        "${MODULE_ARGS[@]}" 2>"$yerr")"
    if [ $? -ne 0 ]; then
        log_error "Could not choose a configuration in '${config_file}': $(cat "$yerr")"
        rm -f "$yerr"
        write_status_file "$project_name" "$had_config" \
            "$had_build" "$build_total" "$build_passed" \
            "$had_run" "$run_total" "$run_passed"
        return 1
    fi
    rm -f "$yerr"

    if [ -n "$selected" ]; then
        MODULE_ARGS=(--module "$selected")
        log_info "Using module '${selected}' from '${config_file}'."
    fi

    do_git_reset_pull

    local folder
    yerr="$(mktemp)"
    folder="$("${PY_RUNNER[@]}" "$HELPER" "$config_file" folder "${MODULE_ARGS[@]}" 2>"$yerr")"
    if [ $? -ne 0 ]; then
        log_error "Failed to parse '${config_file}': $(cat "$yerr")"
        rm -f "$yerr"
        write_status_file "$project_name" "$had_config" \
            "$had_build" "$build_total" "$build_passed" \
            "$had_run" "$run_total" "$run_passed"
        return 1
    fi
    rm -f "$yerr"
    [ -z "$folder" ] && folder="."

    local work_dir
    work_dir="$(cd "$folder" 2>/dev/null && pwd)"
    if [ -z "$work_dir" ]; then
        log_error "Configured folder '${folder}' does not exist in project '${project_name}'"
        write_status_file "$project_name" "$had_config" \
            "$had_build" "$build_total" "$build_passed" \
            "$had_run" "$run_total" "$run_passed"
        return 1
    fi

    # ---- BUILD (every configured build runs, independently) ----
    run_all_configs_for_stage "BUILD" "build.sh" "$work_dir" "$config_file" "build"
    had_build=$STAGE_HAD
    build_total=$STAGE_TOTAL
    build_passed=$STAGE_PASSED

    # ---- RUN (every configured run runs, independently) ----
    run_all_configs_for_stage "RUN" "run.sh" "$work_dir" "$config_file" "run"
    had_run=$STAGE_HAD
    run_total=$STAGE_TOTAL
    run_passed=$STAGE_PASSED

    write_status_file "$project_name" "$had_config" \
        "$had_build" "$build_total" "$build_passed" \
        "$had_run" "$run_total" "$run_passed"
    log_info "Full output written to ${OUTPUT_FILE}"
}
# ----------------------------------------------------------------------------
# Parent mode — this directory is NOT a git project; treat subfolders as
# a collection of git projects
# ----------------------------------------------------------------------------

#   Seconds allowed for one project. RUNNER_TIMEOUT_SECONDS overrides the
#   minutes setting; the test suite uses it to avoid waiting minutes.
timeout_seconds() {
    if [ -n "${RUNNER_TIMEOUT_SECONDS:-}" ]; then
        echo "$RUNNER_TIMEOUT_SECONDS"
    else
        echo $((TIMEOUT_MINUTES * 60))
    fi
}

# Runs one project's single-project pass in a subshell with a time limit.
#   $1 = project directory   $2 = config file name
# Returns the pass's exit status, or 124 if the time limit was reached. Both
# the pass and its watchdog run as their own process groups, so on timeout the
# whole group is stopped, including any build.sh / run.sh it started.
run_child_with_timeout() {
    local dir="$1" config="$2"
    local limit marker child_pid wd_pid rc
    limit="$(timeout_seconds)"
    marker="$(mktemp -u)"

    set -m   # job control: each background job gets its own process group
    ( cd "$dir" && exec bash "$SCRIPT_FULL_PATH" "$config" "${MODULE_ARGS[@]}" ) &
    child_pid=$!
    (
        sleep "$limit"
        touch "$marker"
        kill -TERM -- -"$child_pid" 2>/dev/null
        sleep 5
        kill -KILL -- -"$child_pid" 2>/dev/null
    ) &
    wd_pid=$!
    set +m

    wait "$child_pid"
    rc=$?

    kill -TERM -- -"$wd_pid" 2>/dev/null
    wait "$wd_pid" 2>/dev/null

    if [ -e "$marker" ]; then
        rm -f "$marker"
        return 124
    fi
    return $rc
}

run_parent() {
    local config_file="$1"
    local results_file="assignments_results.txt"
    : > "$results_file"

    echo -e "${BOLD}${BLUE}🚀 Collection folder detected — scanning subfolders for git projects...${NC}"
    echo -e "${BLUE}ℹ️  Time limit per project: ${TIMEOUT_MINUTES} minute(s)${NC}"

    local dir
    for dir in */ ; do
        [ -d "${dir}.git" ] || continue
        dir="${dir%/}"

        echo -e "${BOLD}${BLUE}📦 Entering project '${dir}'${NC}"
        rm -f "${dir}/${STATUS_FILE}"
        run_child_with_timeout "$dir" "$config_file"
        if [ $? -eq 124 ]; then
            echo -e "${RED}⏱  Project '${dir}' TIMED OUT after ${TIMEOUT_MINUTES} minute(s) — stopped${NC}"
            echo "${dir} | TIMED OUT after ${TIMEOUT_MINUTES} minute(s)" >> "$results_file"
            rm -f "${dir}/${STATUS_FILE}"
            continue
        fi

        local line="${dir} | config: "
        if [ -f "${dir}/${STATUS_FILE}" ]; then
            # shellcheck disable=SC1090
            (
                source "${dir}/${STATUS_FILE}"
                out="${PROJECT_NAME}"
                if [ "${HAD_CONFIG}" = "1" ]; then
                    out="${out} | config: yes"
                else
                    out="${out} | config: no"
                fi
                if [ "${HAD_BUILD}" = "1" ]; then
                    if [ "${BUILD_TOTAL}" = "0" ]; then
                        out="${out} | build: none configured"
                    elif [ "${BUILD_PASSED}" = "${BUILD_TOTAL}" ]; then
                        out="${out} | build: OK (${BUILD_PASSED}/${BUILD_TOTAL})"
                    else
                        out="${out} | build: FAILED (${BUILD_PASSED}/${BUILD_TOTAL})"
                    fi
                else
                    out="${out} | build: none"
                fi
                if [ "${HAD_RUN}" = "1" ]; then
                    if [ "${RUN_TOTAL}" = "0" ]; then
                        out="${out} | run: none configured"
                    elif [ "${RUN_PASSED}" = "${RUN_TOTAL}" ]; then
                        out="${out} | run: OK (${RUN_PASSED}/${RUN_TOTAL})"
                    else
                        out="${out} | run: FAILED (${RUN_PASSED}/${RUN_TOTAL})"
                    fi
                else
                    out="${out} | run: none"
                fi
                echo "${out}"
            ) >> "$results_file"
            rm -f "${dir}/${STATUS_FILE}"
        else
            echo "${dir} | config: no | build: none | run: none" >> "$results_file"
        fi
    done

    echo -e "${BOLD}${GREEN}✅ All projects processed. Results written to ${results_file}${NC}"
    echo
    cat "$results_file"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--module)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}❌ $1 needs a module name, e.g. $1 alpha${NC}" >&2
                exit 1
            fi
            MODULE="$2"
            shift 2
            ;;
        --module=*)
            MODULE="${1#--module=}"
            if [ -z "$MODULE" ]; then
                echo -e "${RED}❌ --module needs a module name, e.g. --module=alpha${NC}" >&2
                exit 1
            fi
            shift
            ;;
        -t|--timeout)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}❌ $1 needs a number of minutes, e.g. $1 30${NC}" >&2
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$((10#$2))" -eq 0 ]; then
                echo -e "${RED}❌ $1 must be a whole number of minutes, greater than 0 (got '$2')${NC}" >&2
                exit 1
            fi
            TIMEOUT_MINUTES=$((10#$2))
            TIMEOUT_GIVEN=1
            shift 2
            ;;
        --timeout=*)
            value="${1#--timeout=}"
            if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$((10#$value))" -eq 0 ]; then
                echo -e "${RED}❌ --timeout must be a whole number of minutes, greater than 0 (got '$value')${NC}" >&2
                exit 1
            fi
            TIMEOUT_MINUTES=$((10#$value))
            TIMEOUT_GIVEN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}❌ Unknown option '$1'${NC}" >&2
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ -n "$MODULE" ]; then
    MODULE_ARGS=(--module "$MODULE")
fi

for arg in "${POSITIONAL[@]}"; do
    if [ -d "$arg" ]; then
        TARGET_DIR="$arg"
    else
        CONFIG_FILE="$arg"
    fi
done

if [ -n "$TARGET_DIR" ]; then
    TARGET_DIR="$(cd "$TARGET_DIR" && pwd)" || exit 1
    # A config that already exists relative to the original cwd should
    # keep working after we cd into the project; otherwise look for it
    # inside TARGET_DIR (the usual assignment_config.yaml case).
    if [ -f "$CONFIG_FILE" ]; then
        CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
    fi
    cd "$TARGET_DIR" || exit 1
    [ "$TIMEOUT_GIVEN" -eq 1 ] && echo -e "${YELLOW}⚠️  -t applies to collection runs only; it is ignored for a single project.${NC}"
    run_single_project "$CONFIG_FILE"
elif [ -d .git ]; then
    [ "$TIMEOUT_GIVEN" -eq 1 ] && echo -e "${YELLOW}⚠️  -t applies to collection runs only; it is ignored for a single project.${NC}"
    run_single_project "$CONFIG_FILE"
else
    run_parent "$CONFIG_FILE"
fi
