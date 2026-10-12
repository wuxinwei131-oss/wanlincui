"""Process-safe wrapper around QA1's SHA-verified source-fit prediction cache."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import msvcrt
import sys
from contextlib import contextmanager
from pathlib import Path

QA1 = Path(str(method_path('qa1_code')))
sys.path.insert(0, str(QA1))
from tv2_pipeline.cache import CanonicalSourceFitCache, canonical_source_fit_key


@contextmanager
def key_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class LockedCanonicalSourceFitCache:
    """Same QA1 cache semantics; serialize per-key reads/writes across workers."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.inner = CanonicalSourceFitCache(self.root)
        (self.root / "locks").mkdir(exist_ok=True)

    def get(self, spec: dict) -> bytes:
        key = canonical_source_fit_key(spec)
        with key_lock(self.root / "locks" / f"{key}.lock"):
            return self.inner.get(spec)

    def put(self, spec: dict, payload: bytes) -> str:
        key = canonical_source_fit_key(spec)
        with key_lock(self.root / "locks" / f"{key}.lock"):
            try:
                existing = self.inner.get(spec)
            except FileNotFoundError:
                return self.inner.put(spec, payload)
            if existing != payload:
                raise RuntimeError("Same canonical source-fit identity produced nonidentical predictions")
            return key
