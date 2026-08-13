#!/usr/bin/env python3
"""
yaml_helper.py - companion to run_assignments.sh

Reads assignment_config.yaml and prints information about its
"folder", "build", or "run" top-level keys.

    yaml_helper.py <yaml_file> folder
        Prints the "folder" value (raw string, defaults to ".").

    yaml_helper.py <yaml_file> <build|run> --count
        Prints the number of independent configurations found in that
        section (0 if the section is present but empty).

    yaml_helper.py <yaml_file> <build|run> --index N
        Prints a NUL-separated ("\0") list of command-line argument
        tokens for the Nth configuration (0-based) in that section.

A "build" or "run" section may be written in any of these forms:

  1) A single dictionary of {flag: value}
         build:
           target: release
           O: 2
     -> exactly ONE configuration: ./build.sh --target release -O 2

  2) A single flat list of positional values
         run:
           - input.txt
           - 42
     -> exactly ONE configuration: ./run.sh input.txt 42

  3) A list where every item is itself a dictionary or a list
         build:
           - target: debug
           - target: release
             O: 2
           - [--lint-only]
     -> MULTIPLE independent configurations, one per item, each run
        one after another regardless of whether earlier ones failed.

Form (3) is what distinguishes "a list of configurations" from form
(2)'s "a list of positional values for one configuration": if every
item in the top-level list is itself a dict/list, it's treated as a
list of separate configs. Mixing bare scalars with dict/list items in
the same top-level list is invalid (ambiguous) and is rejected with an
error.

Exit codes:
  0 - success, output written to stdout
  3 - the requested section ("build"/"run") does not exist in the YAML
      (not an error - that stage just isn't configured)
  2 - the YAML file could not be read/parsed, the section's shape is
      invalid (e.g. a mixed list), or --index was out of range
  1 - bad usage
"""
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is not installed. Install it with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


def build_args(config):
    """Turn a single config (dict, list, scalar, or None) into a list
    of command-line argument tokens."""
    args = []
    if isinstance(config, dict):
        for key, value in config.items():
            key_str = str(key)
            flag = f"-{key_str}" if len(key_str) == 1 else f"--{key_str}"
            args.append(flag)
            if value is not None and value != "":
                args.append(str(value))
    elif isinstance(config, list):
        args.extend(str(v) for v in config)
    elif config is None:
        pass
    else:
        # a bare scalar config, e.g. "build: some-single-arg"
        args.append(str(config))
    return args


def normalize_configs(section_data):
    """Turn a raw build/run section value into a list of independent
    configs, each of which is itself a dict, list, or scalar suitable
    for build_args(). Raises ValueError on an ambiguous/invalid shape."""
    if section_data is None:
        return []

    if isinstance(section_data, dict):
        return [section_data]

    if isinstance(section_data, list):
        if len(section_data) == 0:
            return []
        is_config_item = [isinstance(item, (dict, list)) for item in section_data]
        if all(is_config_item):
            # every item is itself a dict/list -> a list of independent configs
            return list(section_data)
        if not any(is_config_item):
            # every item is a bare scalar -> one config, positional args
            return [section_data]
        raise ValueError(
            "cannot mix bare scalar values with dict/list items in the same "
            "build/run list — use either a flat list of values (one "
            "configuration) or a list of dicts/lists (multiple "
            "configurations), not both"
        )

    # a bare scalar section, e.g. "build: some-single-arg"
    return [section_data]


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <yaml_file> <folder|build|run> [--count | --index N]", file=sys.stderr)
        sys.exit(1)

    yaml_file, section = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]

    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read/parse '{yaml_file}': {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(data, dict):
        print(f"Top level of '{yaml_file}' must be a mapping (folder/build/run)", file=sys.stderr)
        sys.exit(2)

    if section == "folder":
        sys.stdout.write(str(data.get("folder", ".")))
        return

    if section not in ("build", "run"):
        print(f"Unknown section '{section}'", file=sys.stderr)
        sys.exit(1)

    if section not in data:
        sys.exit(3)

    try:
        configs = normalize_configs(data[section])
    except ValueError as exc:
        print(f"Invalid '{section}' section in '{yaml_file}': {exc}", file=sys.stderr)
        sys.exit(2)

    if "--count" in extra:
        sys.stdout.write(str(len(configs)))
        return

    if "--index" in extra:
        idx_pos = extra.index("--index")
        try:
            idx = int(extra[idx_pos + 1])
        except (IndexError, ValueError):
            print("--index requires an integer argument", file=sys.stderr)
            sys.exit(1)
        if idx < 0 or idx >= len(configs):
            print(f"index {idx} out of range (section has {len(configs)} configuration(s))", file=sys.stderr)
            sys.exit(2)
        args = build_args(configs[idx])
        if args:
            sys.stdout.write("\0".join(args) + "\0")
        return

    print("Specify --count or --index N when querying 'build'/'run'", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
