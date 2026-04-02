"""Wrapper: moment-matched entropy batch for noise_level only. Forwards extra CLI args."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BATCH = Path(__file__).resolve().parent / "recompute_entropy_moment_matched_batch_from_npz.py"
_spec = importlib.util.spec_from_file_location("recompute_entropy_moment_matched_batch_from_npz", _BATCH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    extra = sys.argv[1:]
    sys.argv = [str(Path(__file__).resolve()), "--batch", "--experiments", "noise_level"] + extra
    _mod.main()
