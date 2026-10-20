#!/usr/bin/env python3
"""
yaml_helper.py - companion to run_assignments.sh

A configuration file takes one of two shapes.

  A) A single configuration at the top level:
         folder: src
         build: ...
         run: ...

  B) One or more module objects under a "modules" key. Each module holds the
     same keys as (A), plus a required "name":
         modules:
           - name: alpha          # the top-level folder this module is for
             build: ...
             run: ...
           - name: beta
             folder: lib/beta     # optional; defaults to the name
             run: ...

     A module is chosen with --module NAME. Without --module, the module named
     "current" is used, if there is one. Otherwise no module is chosen and the
     run stops, listing the available modules.

Commands:

    yaml_helper.py <yaml_file> select-module [--module NAME]
        Prints the name of the module to use, or nothing for a top-level
        configuration. Exit 2 if no single module can be chosen.

    yaml_helper.py <yaml_file> folder [--module NAME]
        Prints the folder that build.sh / run.sh live in. Defaults to "."
        for a top-level configuration and to the module name for a module.

    yaml_helper.py <yaml_file> <build|run> [--module NAME] --count
    yaml_helper.py <yaml_file> <build|run> [--module NAME] --index N
    yaml_helper.py <yaml_file> <build|run> [--module NAME] --stdin-index N
        As described below.

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
         line; may contain \\n for several lines) or a list of values (one
         line each). Without it the script gets an empty input (EOF).
  args   A list of positional values, appended after any flags.
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
  3 - the requested section ("build"/"run") does not exist in the selected
      configuration (not an error - that stage just isn't configured)
  2 - the YAML file could not be read/parsed, the configuration or module
      choice is invalid, a section's shape is invalid, or an index is out
      of range
  1 - bad usage
"""
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is not installed. Install it with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


class ConfigError(Exception):
    """A configuration-level problem (module choice or file shape)."""


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


def module_list(data):
    """Return the module objects under data["modules"], checked for shape,
    a non-empty "name", and unique names."""
    value = data["modules"]
    if value is None:
        raise ConfigError("'modules' is empty")
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        if not value:
            raise ConfigError("'modules' is an empty list")
        items = value
    else:
        raise ConfigError("'modules' must be a module object or a list of module objects")

    for number, module in enumerate(items, start=1):
        if not isinstance(module, dict):
            raise ConfigError(f"module {number} must be a mapping with a 'name' key")
        name = module.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"module {number} needs a non-empty 'name'")

    names = [module["name"] for module in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError("duplicate module name(s): " + ", ".join(duplicates))
    return items


DEFAULT_MODULE = "current"


def resolve(data, module):
    """Choose the configuration to use.

    Returns (source, name): `source` is the mapping that holds folder / build /
    run, and `name` is the module name, or "" for a top-level configuration.
    Without `module`, the module named "current" is used. Raises ConfigError if
    no single configuration can be chosen.
    """
    if "modules" not in data:
        if module is not None:
            raise ConfigError(
                f"--module '{module}' was given, but the configuration has no 'modules' key")
        return data, ""

    if any(key in data for key in ("folder", "build", "run")):
        raise ConfigError(
            "use either top-level folder/build/run or a 'modules' key, not both")

    modules = module_list(data)
    available = ", ".join(m["name"] for m in modules)

    wanted = module if module is not None else DEFAULT_MODULE
    for candidate in modules:
        if candidate["name"] == wanted:
            return candidate, wanted

    if module is not None:
        raise ConfigError(f"no module named '{module}'; available modules: {available}")
    raise ConfigError(
        f"no --module given and no module named '{DEFAULT_MODULE}' to use by default; "
        f"use --module with one of: {available}")


def folder_of(source, selected_name):
    """The folder for build.sh / run.sh: the 'folder' key if set, otherwise
    the module name (for a module) or '.' (for a top-level configuration)."""
    value = source.get("folder")
    if value is None or value == "":
        value = selected_name if selected_name else "."
    return str(value)


VALUE_OPTIONS = ("--module", "--index", "--stdin-index")


def parse_options(args):
    """Parse the options after the command. Exits 1 on bad usage."""
    opts = {"module": None, "count": False, "index": None, "stdin-index": None}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--count":
            opts["count"] = True
            i += 1
            continue
        if arg in VALUE_OPTIONS:
            if i + 1 >= len(args) or args[i + 1] == "":
                print(f"{arg} requires a non-empty value", file=sys.stderr)
                sys.exit(1)
            opts[arg[2:]] = args[i + 1]
            i += 2
            continue
        print(f"Unknown option '{arg}'", file=sys.stderr)
        sys.exit(1)
    return opts


def checked_index(text, flag, count):
    """Parse an index option and check it is within 0..count-1."""
    try:
        idx = int(text)
    except (TypeError, ValueError):
        print(f"{flag} requires an integer argument", file=sys.stderr)
        sys.exit(1)
    if idx < 0 or idx >= count:
        print(f"{flag} {idx} is out of range (section has {count} configuration(s))", file=sys.stderr)
        sys.exit(2)
    return idx


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <yaml_file> <select-module|folder|build|run> [options]",
              file=sys.stderr)
        sys.exit(1)

    yaml_file, command = sys.argv[1], sys.argv[2]
    opts = parse_options(sys.argv[3:])

    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read/parse '{yaml_file}': {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(data, dict):
        print(f"Top level of '{yaml_file}' must be a mapping", file=sys.stderr)
        sys.exit(2)

    if command in ("select-module", "folder"):
        try:
            source, name = resolve(data, opts["module"])
        except ConfigError as exc:
            print(f"Invalid configuration '{yaml_file}': {exc}", file=sys.stderr)
            sys.exit(2)
        if command == "select-module":
            sys.stdout.write(name)
        else:
            sys.stdout.write(folder_of(source, name))
        return

    if command not in ("build", "run"):
        print(f"Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)

    try:
        source, _ = resolve(data, opts["module"])
    except ConfigError as exc:
        print(f"Invalid configuration '{yaml_file}': {exc}", file=sys.stderr)
        sys.exit(2)

    if command not in source:
        sys.exit(3)

    try:
        configs = normalize_configs(source[command])
    except ValueError as exc:
        print(f"Invalid '{command}' section in '{yaml_file}': {exc}", file=sys.stderr)
        sys.exit(2)

    # Validate every configuration up front, so a malformed one is reported
    # no matter which query the caller makes.
    parsed = []
    for number, config in enumerate(configs, start=1):
        try:
            parsed.append(parse_config(config))
        except ValueError as exc:
            print(f"Invalid '{command}' section in '{yaml_file}', configuration {number}: {exc}",
                  file=sys.stderr)
            sys.exit(2)

    if opts["count"]:
        sys.stdout.write(str(len(parsed)))
        return

    if opts["index"] is not None:
        argv, _ = parsed[checked_index(opts["index"], "--index", len(parsed))]
        if argv:
            sys.stdout.write("\0".join(argv) + "\0")
        return

    if opts["stdin-index"] is not None:
        _, lines = parsed[checked_index(opts["stdin-index"], "--stdin-index", len(parsed))]
        if lines:
            sys.stdout.write("".join(line + "\0" for line in lines))
        return

    print("Specify --count, --index N, or --stdin-index N when querying 'build'/'run'", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
