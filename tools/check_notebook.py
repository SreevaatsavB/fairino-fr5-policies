#!/usr/bin/env python3
"""
check_notebook.py — cross-cell name-flow validation for notebooks.

    python tools/check_notebook.py notebooks/*.ipynb
    python tools/check_notebook.py --self-test

Why this exists: ast.parse-per-cell catches syntax errors but can NEVER catch a
name defined in one cell and used in another going missing — which is exactly how
`sub_val` was deleted from the training cell of train_pi0_delta_runpod.ipynb
while its call survived, killing a pod run at step 500 (NameError, 5 h lost).

For each code cell in top-to-bottom order it accumulates every name the cell
defines (assignments, imports, def/class, except-as, comprehension targets,
function parameters, walrus) and flags any loaded name that is neither a builtin
nor defined in this or ANY earlier cell. Run order = document order, the same
assumption "Run All" makes.

Limits (deliberate, this is a linter not an interpreter): a name defined later in
the same cell counts as defined (functions may be declared after their callers);
`del` is ignored; attribute access is not checked. False negatives are possible;
false positives should be near zero — anything flagged is worth reading.

Exit 0 = every notebook clean; 1 = findings; 2 = usage / unreadable file.
"""

import ast
import builtins
import json
import sys
from pathlib import Path

# names the notebook runtime injects
RUNTIME = {"display", "get_ipython", "In", "Out", "exit", "quit"}
BUILTIN = set(dir(builtins)) | RUNTIME


def cell_names(src: str):
    """(defined, used) name sets for one cell. Raises SyntaxError."""
    src = "\n".join(l for l in src.splitlines()
                    if not l.strip().startswith(("!", "%")))
    tree = ast.parse(src)
    defined, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            else:
                used.add(node.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.Lambda):
            for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                defined.add(a.arg)
    return defined, used


def check(path: Path):
    """[(cell_index, message)] findings for one notebook."""
    nb = json.loads(path.read_text())
    findings, seen = [], set()
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            defined, used = cell_names(src)
        except SyntaxError as e:
            findings.append((i, f"syntax error: {e}"))
            continue
        missing = sorted(used - defined - seen - BUILTIN)
        if missing:
            findings.append((i, f"used but never defined in this or any earlier "
                                f"cell: {missing}"))
        seen |= defined
    return findings


def self_test():
    """The check must catch the exact bug class that motivated it."""
    def nb(*cells):
        return {"cells": [{"cell_type": "code", "source": [c]} for c in cells]}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # cell 2 calls a function cell 1 never defines -> MUST flag (the sub_val bug)
        bad = Path(td, "bad.ipynb")
        bad.write_text(json.dumps(nb("def full_val():\n    return 1\n",
                                     "v = sub_val()\n")))
        f = check(bad)
        assert f and "sub_val" in f[0][1], f
        # helper defined in an earlier cell -> clean
        good = Path(td, "good.ipynb")
        good.write_text(json.dumps(nb("def sub_val():\n    return 1\n",
                                      "v = sub_val()\n")))
        assert check(good) == [], check(good)
        # define-after-use within one cell (declared later, called earlier at
        # runtime-safe positions) -> clean; imports, magics, comprehensions, walrus
        tricky = Path(td, "tricky.ipynb")
        tricky.write_text(json.dumps(nb(
            "!pip install x\nimport numpy as np\nfrom pathlib import Path as P\n",
            "def caller():\n    return helper()\ndef helper():\n    return np.pi\n",
            "xs = [P(str(i)) for i in range(3)]\nif (n := len(xs)) > 1:\n    print(n)\n",
            "try:\n    caller()\nexcept Exception as e:\n    print(e)\n")))
        assert check(tricky) == [], check(tricky)
        # syntax error is reported, not swallowed
        broken = Path(td, "broken.ipynb")
        broken.write_text(json.dumps(nb("def f(:\n")))
        f = check(broken)
        assert f and "syntax" in f[0][1], f
    print("self-test: catches the sub_val bug class, no false positives on the "
          "tricky cases, reports syntax errors")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "--self-test":
        self_test()
        return 0
    bad = 0
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"{path}: not found")
            return 2
        findings = check(path)
        if findings:
            bad += 1
            print(f"FAIL {path.name}")
            for i, msg in findings:
                print(f"       cell {i}: {msg}")
        else:
            print(f"ok   {path.name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
