"""Package bootstrap for *src*.

Eager‑import all sub‑modules **except** those listed in `_SKIP`.  Skipping
`main`, `workers`, and `ga_optimization` prevents circular imports and accidental
execution of top‑level code that should only run when invoked directly.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import types
from typing import Set

# Modules to skip during eager import (avoid circulars / heavy deps)
_SKIP: Set[str] = {"main", "workers", "ga_optimization"}
_pkg_name = __name__  # 'src'

for modinfo in pkgutil.walk_packages(__path__, prefix=f"{_pkg_name}."):
    short = modinfo.name.rsplit(".", 1)[-1]
    if short in _SKIP:
        continue
    try:
        importlib.import_module(modinfo.name)
    except Exception as exc:
        print(f"[src.__init__] skipped {modinfo.name}: {exc}", file=sys.stderr)

# export list for `from src import *` (rarely used)
__all__ = [m.rsplit(".", 1)[-1] for m in sys.modules if m.startswith(f"{_pkg_name}.") and m.rsplit(".", 1)[-1] not in _SKIP]
