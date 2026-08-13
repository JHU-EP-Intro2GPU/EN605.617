#!/usr/bin/env bash
#
# run_assignments.sh
#
# Runs a project's build.sh / run.sh as configured by an
# assignment_config.yaml file.
#
# Usage:
#   ./run_assignments.sh [config_file.yaml]
#
#   config_file.yaml defaults to "assignment_config.yaml"
#
# Behavior:
#   - If run inside a git project (a directory containing .git), it will:
#       1. git pull (only if currently on main/master)
#       2. run build.sh (if a "build" section exists in the config)
#       3. run run.sh   (if a "run" section exists in the config)
#     ...using the folder / flags described in the YAML config, logging
#     everything to screen (in color) and to assignment_output.txt.
#
#   - If run inside a folder that is NOT itself a git project (i.e. a
#     folder that contains a collection of git project subfolders), it
#     will recurse into each subfolder that IS a git project, run this
#     same script there, and summarize all results into
#     assignments_results.txt.
#
# Requires: bash 4.4+, python3, PyYAML (pip3 install pyyaml)

set -uo pipefail

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
SCRIPT_FULL_PATH="$SCRIPT_DIR/$(basename "$SCRIPT_SOURCE")"
HELPER="$SCRIPT_DIR/yaml_helper.py"

CONFIG_FILE="${1:-assignment_config.yaml}"

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

# ----------------------------------------------------------------------------
# git pull — only if currently on main/master
# ----------------------------------------------------------------------------

do_git_pull() {
    if [ ! -d .git ]; then
        return
    fi

    local branch
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

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
    shift 5
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

    local tmp_out
    tmp_out="$(mktemp)"

    ( cd "$work_dir" && bash "./${script_name}" "${args[@]}" ) > "$tmp_out" 2>&1
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
    "${PY_RUNNER[@]}" "$HELPER" "$config_file" "$section" --count > "$count_file" 2>"$yerr"
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
        local args_file
        args_file="$(mktemp)"
        "${PY_RUNNER[@]}" "$HELPER" "$config_file" "$section" --index "$i" > "$args_file"
        local args=()
        readarray -d '' args < "$args_file"
        rm -f "$args_file"

        if run_stage "$stage_name" "$script_name" "$work_dir" "$((i + 1))" "$STAGE_TOTAL" "${args[@]}"; then
            STAGE_PASSED=$((STAGE_PASSED + 1))
        fi
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

    do_git_pull

    local folder yerr
    yerr="$(mktemp)"
    folder="$("${PY_RUNNER[@]}" "$HELPER" "$config_file" folder 2>"$yerr")"
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

run_parent() {
    local config_file="$1"
    local results_file="assignments_results.txt"
    : > "$results_file"

    echo -e "${BOLD}${BLUE}🚀 Collection folder detected — scanning subfolders for git projects...${NC}"

    local dir
    for dir in */ ; do
        [ -d "${dir}.git" ] || continue
        dir="${dir%/}"

        echo -e "${BOLD}${BLUE}📦 Entering project '${dir}'${NC}"
        ( cd "$dir" && bash "$SCRIPT_FULL_PATH" "$config_file" )

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

if [ -d .git ]; then
    run_single_project "$CONFIG_FILE"
else
    run_parent "$CONFIG_FILE"
fi
