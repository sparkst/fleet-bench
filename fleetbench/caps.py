"""Budget caps: load caps.yaml and enforce per-provider call ceilings.

Cap defaults are call counts. ``CapCounter`` is thread-safe so the concurrent
per-provider workers can share one counter. ``max_calls == 0`` means unlimited.
"""

from __future__ import annotations

import threading
from pathlib import Path

import yaml

DEFAULT_CAPS_PATH = Path(__file__).resolve().parent.parent / "caps.yaml"


def load_caps(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CAPS_PATH
    return yaml.safe_load(p.read_text(encoding="utf-8"))


class CapCounter:
    """Tracks calls per provider and enforces max_calls from caps.yaml."""

    def __init__(self, caps: dict):
        self.caps = caps or {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _max_for(self, provider: str) -> int:
        # 'judge' is a top-level budget, not a provider under providers:.
        if provider == "judge":
            return int(self.caps.get("judge", {}).get("max_calls", 0))
        return int(self.caps.get("providers", {}).get(provider, {}).get("max_calls", 0))

    def rpm_for(self, provider: str) -> float:
        return float(self.caps.get("providers", {}).get(provider, {}).get("rpm", 0) or 0)

    def concurrency_for(self, provider: str) -> int:
        return int(self.caps.get("providers", {}).get(provider, {}).get("concurrency", 1) or 1)

    def try_reserve(self, provider: str) -> bool:
        """Reserve one call for ``provider``. Returns False if the cap is hit."""
        with self._lock:
            limit = self._max_for(provider)
            used = self._counts.get(provider, 0)
            if limit and used >= limit:
                return False
            self._counts[provider] = used + 1
            return True

    def count(self, provider: str) -> int:
        with self._lock:
            return self._counts.get(provider, 0)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counts)
