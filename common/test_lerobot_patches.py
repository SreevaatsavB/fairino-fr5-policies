"""
test_lerobot_patches.py — the groot stub must let the pi-family import.

    python common/test_lerobot_patches.py

This is the bug that killed a pod run at the model cell: lerobot's
policies/__init__ eagerly imports groot, whose GR00TN15Config is a @dataclass
with a non-default field after defaulted ones — a TypeError as soon as
transformers' PretrainedConfig is itself a dataclass (5.x). One broken class took
down every pi-family import.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_stub_is_inert_after_import():
    """Calling it once lerobot is already loaded must change nothing."""
    sys.path.insert(0, str(REPO / "common"))
    from lerobot_patches import stub_unused_policies
    first = stub_unused_policies()
    assert isinstance(first, list)
    import lerobot.policies  # noqa: F401
    assert stub_unused_policies() == [], "must no-op once lerobot.policies exists"
    print("  stub is inert after lerobot.policies is imported")


def test_each_policy_imports_in_a_fresh_interpreter():
    """The real check: a clean process, like the notebook's model cell."""
    for pol in ("pi0", "pi05", "pi0_fast"):
        code = (f"import importlib.util;"
                f"s=importlib.util.spec_from_file_location('m',r'{REPO}/policies/{pol}/model.py');"
                f"m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                f"assert hasattr(m,'build_model');print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=str(REPO))
        assert r.returncode == 0, f"{pol} failed to import:\n{r.stderr[-1500:]}"
        assert "ok" in r.stdout, r.stdout
        print(f"  {pol}/model.py imports in a fresh interpreter")


def test_stub_precedes_the_lerobot_import():
    """Order matters — the stub cannot repair an import that already failed."""
    import re
    for pol in ("pi0", "pi05", "pi0_fast"):
        src = (REPO / "policies" / pol / "model.py").read_text()
        # line-anchored: the phrase also appears inside the explanatory comment
        i_stub = re.search(r"^stub_unused_policies\(\)", src, re.M).start()
        i_lero = re.search(r"^from lerobot\.policies\.", src, re.M).start()
        assert i_stub < i_lero, f"{pol}: stub call must come before the lerobot import"
    print("  all three call the stub before importing lerobot.policies")


if __name__ == "__main__":
    for fn in (test_stub_is_inert_after_import,
               test_stub_precedes_the_lerobot_import,
               test_each_policy_imports_in_a_fresh_interpreter):
        print(f"{fn.__name__}:")
        fn()
    print("\nall lerobot-patch checks passed")
