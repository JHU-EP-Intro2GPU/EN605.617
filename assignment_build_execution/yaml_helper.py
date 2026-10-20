#!/usr/bin/env python3
"""
yaml_helper.py - companion to run_assignments.sh

Reads assignment_config.yaml and prints information about its "folder",
"build", or "run" top-level keys.

    yaml_helper.py <yaml_file> folder
        Prints the "folder" value (raw string, defaults to ".").

    yaml_helper.py <yaml_file> <build|run> --count
        Prints the number of independent configurations in that section
        (0 if the section is present but empty).

    yaml_helper.py <yaml_file> <build|run> --index N
        Prints the command-line tokens for the Nth configuration (0-based),
        each followed by a NUL byte. Prints nothing if there are none.

    yaml_helper.py <yaml_file> <build|run> --stdin-index N
        Prints the standard-input lines for the Nth configuration (0-based),
        each followed by a NUL byte. Prints nothing if it has no input.

A "build" or "run" section may be written in any of these forms:

  1) A single dictionary (one configuration)
         build:
           target: release
           O: 2
     -> ./build.sh --target release -O 2

  2) A single flat list of positional values (one configuration)
         run:
           - input.txt
           - 42
     -> ./run.sh input.txt 42

  3) A list where every item is a dictionary or a list (one configuration
     per item, run one after another)
         run:
           - stdin: [Ada, blue]
           - args: [--shout]
             stdin: [Grace, green]
           - [input.txt]

Keys inside a dictionary configuration:
  stdin  Lines fed to the script's standard input. Either a string (one
         line; may contain \n for several lines) or a list of values (one
         line each). Without it the script gets an empty input (EOF).
  args   A list of positional values, appended after any flags. Use it to
         combine positional values with stdin in one dictionary.
  any other key
         A flag: a 1-character key becomes "-k value", a longer key becomes
         "--key value". true/false values are written in lowercase.

"stdin" and "args" are reserved, so a flag with either name cannot be written
in dictionary form.

A flat list of values (form 2) cannot carry stdin. Use form 1 or form 3 with
an "args" key when a configuration needs both positional values and input.

Mixing bare values with dictionaries/lists in the same top-level list is
ambiguous and is rejected.

Exit codes:
  0 - success, output written to stdout
  3 - the requested section ("build"/"run") does not exist in the YAML
      (not an error - that stage just isn't configured)
  2 - the YAML file could not be read/parsed, the section's shape is
      invalid, or the requested index is out of range
  1 - bad usage
"""
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is not installed. Install it with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


def _to_text(value):
    """Render a YAML scalar as a string (booleans as lowercase true/false)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _plain_values(value, what):
    """Return `value` as a list of strings, for stdin lines or positional args.

    None -> [], a scalar -> [its text], a list -> one string per item (a null
    item becomes an empty string). Mappings and nested lists are rejected.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        raise ValueError(f"{what} must be a string or a list of values, not a mapping")
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, (dict, list)):
                raise ValueError(f"{what} must contain only plain values, not mappings or lists")
            out.append("" if item is None else _to_text(item))
        return out
    return [_to_text(value)]


def parse_config(config):
    """Split one configuration into (argv_tokens, stdin_lines).

    Raises ValueError if the configuration is malformed.
    """
    if config is None:
        return [], []

    if isinstance(config, dict):
        flags = []
        positional = []
        stdin_lines = []
        for key, value in config.items():
            key_str = str(key)
            if key_str == "stdin":
                stdin_lines = _plain_values(value, "'stdin'")
            elif key_str == "args":
                if not isinstance(value, list):
                    raise ValueError("'args' must be a list of positional values")
                positional = _plain_values(value, "'args'")
            else:
                flags.append(f"-{key_str}" if len(key_str) == 1 else f"--{key_str}")
                if value is not None and value != "":
                    flags.append(_to_text(value))
        return flags + positional, stdin_lines

    if isinstance(config, list):
        return _plain_values(config, "a configuration list"), []

    # a bare scalar config, e.g. "build: some-single-arg"
    return [_to_text(config)], []


def normalize_configs(section_data):
    """Turn a raw build/run section value into a list of independent configs,
    each of which is a dict, list, or scalar. Raises ValueError on an
    ambiguous/invalid shape."""
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
            "cannot mix bare values with dict/list items in the same build/run "
            "list - use either a flat list of values (one configuration) or a "
            "list of dicts/lists (multiple configurations), not both"
        )

    # a bare scalar section, e.g. "build: some-single-arg"
    return [section_data]


def _index_arg(extra, flag, count):
    pos = extra.index(flag)
    try:
        idx = int(extra[pos + 1])
    except (IndexError, ValueError):
        print(f"{flag} requires an integer argument", file=sys.stderr)
        sys.exit(1)
    if idx < 0 or idx >= count:
        print(f"{flag} {idx} is out of range (section has {count} configuration(s))", file=sys.stderr)
        sys.exit(2)
    return idx


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <yaml_file> <folder|build|run> [--count | --index N | --stdin-index N]",
              file=sys.stderr)
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

    # Validate every configuration up front, so a malformed one is reported
    # no matter which query the caller makes.
    parsed = []
    for number, config in enumerate(configs, start=1):
        try:
            parsed.append(parse_config(config))
        except ValueError as exc:
            print(f"Invalid '{section}' section in '{yaml_file}', configuration {number}: {exc}",
                  file=sys.stderr)
            sys.exit(2)

    if "--count" in extra:
        sys.stdout.write(str(len(parsed)))
        return

    if "--index" in extra:
        argv, _ = parsed[_index_arg(extra, "--index", len(parsed))]
        if argv:
            sys.stdout.write("\0".join(argv) + "\0")
        return

    if "--stdin-index" in extra:
        _, lines = parsed[_index_arg(extra, "--stdin-index", len(parsed))]
        if lines:
            sys.stdout.write("".join(line + "\0" for line in lines))
        return

    print("Specify --count, --index N, or --stdin-index N when querying 'build'/'run'", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
