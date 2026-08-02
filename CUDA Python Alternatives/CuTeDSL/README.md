CuTeDSL Example Project

This is a minimal example project demonstrating a tiny CuTeDSL-style language and a simple Python interpreter.

Files
- [example/example.cute](example/example.cute)
- [run_example.py](run_example.py)

Quick start

Run the interpreter on the example DSL file:

```bash
python3 run_example.py example/example.cute
```

The interpreter supports these simple commands:
- `say <text>`: print text (supports `${var}` variable interpolation)
- `set <name> = <value>`: set a variable (numbers or quoted strings)
- `add <name> <value>`: add numeric value to a numeric variable
- `sub <name> <value>`: subtract numeric value
- `mul <name> <value>`: multiply numeric value
- `div <name> <value>`: divide numeric value
- `if <left> <op> <right>` / `end`: conditional block supporting `== != > >= < <=`
- `while <left> <op> <right>` / `end`: loop block

Feel free to modify `example/example.cute` and experiment.