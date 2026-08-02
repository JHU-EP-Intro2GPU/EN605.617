#!/usr/bin/env python3
"""Tiny CuTeDSL interpreter example.

Usage:
    python3 run_example.py example/example.cute
"""
import argparse
import re
import sys

VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def replace_vars(s, vars_):
    def repl(m):
        name = m.group(1)
        return str(vars_.get(name, ""))
    return VAR_RE.sub(repl, s)


def parse_value(token, vars_):
    token = token.strip()
    # quoted string
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return replace_vars(token[1:-1], vars_)
    # ${var} reference
    if token.startswith('${') and token.endswith('}'):
        name = token[2:-1]
        return vars_.get(name)
    # bare variable name
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', token):
        return vars_.get(token)
    # number
    try:
        if '.' in token:
            return float(token)
        return int(token)
    except Exception:
        return replace_vars(token, vars_)


def evaluate_condition(left_tok, op, right_tok, vars_):
    left = parse_value(left_tok, vars_)
    right = parse_value(right_tok, vars_)
    try:
        if op == '==':
            return left == right
        if op == '!=':
            return left != right
        if op == '>':
            return float(left) > float(right)
        if op == '<':
            return float(left) < float(right)
        if op == '>=':
            return float(left) >= float(right)
        if op == '<=':
            return float(left) <= float(right)
    except Exception:
        return False
    return False


def find_matching_end(lines, start):
    depth = 0
    for j in range(start + 1, len(lines)):
        l = lines[j].strip()
        if l.startswith('if ') or l.startswith('while '):
            depth += 1
        if l == 'end':
            if depth == 0:
                return j
            depth -= 1
    return None


def execute_lines(lines, vars_):
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        if line.startswith('say ') or line.startswith('print '):
            payload = line.split(None, 1)[1].strip()
            if payload.startswith('"') and payload.endswith('"'):
                payload = payload[1:-1]
            payload = replace_vars(payload, vars_)
            print(payload)
            i += 1
            continue

        if line.startswith('set '):
            m = re.match(r'set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)', line)
            if not m:
                print(f"Invalid set syntax: {line}", file=sys.stderr)
                i += 1
                continue
            name, valtok = m.group(1), m.group(2).strip()
            val = parse_value(valtok, vars_)
            vars_[name] = val
            i += 1
            continue

        for op_cmd in ('add', 'sub', 'mul', 'div'):
            if line.startswith(op_cmd + ' '):
                parts = line.split(None, 2)
                if len(parts) < 3:
                    print(f"Invalid {op_cmd} syntax: {line}", file=sys.stderr)
                    break
                name = parts[1]
                val = parse_value(parts[2], vars_)
                try:
                    cur = vars_.get(name, 0) or 0
                    if cur is None:
                        cur = 0
                    if val is None:
                        val = 0
                    if op_cmd == 'add':
                        vars_[name] = (float(cur) + float(val)) if (isinstance(cur, (int, float)) or isinstance(val, (int, float))) else str(cur) + str(val)
                    elif op_cmd == 'sub':
                        vars_[name] = float(cur) - float(val)
                    elif op_cmd == 'mul':
                        vars_[name] = float(cur) * float(val)
                    elif op_cmd == 'div':
                        vars_[name] = float(cur) / float(val)
                except Exception:
                    print(f"Cannot {op_cmd} non-numeric values: {name} {op_cmd} {val}", file=sys.stderr)
                break
        else:
            # not a math op
            if line.startswith('if '):
                m = re.match(r'if\s+(.+)', line)
                if not m:
                    i += 1
                    continue
                cond = m.group(1).strip()
                m2 = re.match(r'(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)', cond)
                if not m2:
                    i += 1
                    continue
                left, op, right = m2.group(1).strip(), m2.group(2), m2.group(3).strip()
                end_idx = find_matching_end(lines, i)
                if end_idx is None:
                    print(f"Missing end for if starting at line: {i+1}", file=sys.stderr)
                    return
                if evaluate_condition(left, op, right, vars_):
                    block = lines[i+1:end_idx]
                    execute_lines(block, vars_)
                i = end_idx + 1
                continue

            if line.startswith('while '):
                m = re.match(r'while\s+(.+)', line)
                if not m:
                    i += 1
                    continue
                cond = m.group(1).strip()
                m2 = re.match(r'(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)', cond)
                if not m2:
                    i += 1
                    continue
                left, op, right = m2.group(1).strip(), m2.group(2), m2.group(3).strip()
                end_idx = find_matching_end(lines, i)
                if end_idx is None:
                    print(f"Missing end for while starting at line: {i+1}", file=sys.stderr)
                    return
                block = lines[i+1:end_idx]
                # loop
                loop_guard = 0
                while evaluate_condition(left, op, right, vars_):
                    execute_lines(block, vars_)
                    loop_guard += 1
                    if loop_guard > 10000:
                        print('Aborting possible infinite loop', file=sys.stderr)
                        break
                i = end_idx + 1
                continue

            if line == 'end':
                # handled by callers; skip
                i += 1
                continue

            print(f"Unknown command: {line}", file=sys.stderr)
            i += 1


def run(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    vars_ = {}
    execute_lines(lines, vars_)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file', nargs='?', default='example/example.cute')
    args = p.parse_args()
    run(args.file)
