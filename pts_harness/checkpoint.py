"""Crash-safe, resumable run storage.

Two levels, and the second is what makes resume cheap:

* **Durable unit: the query.** When a query finishes, one line goes into
  ``manifest.jsonl``. Resume skips those uids.
* **Memoized unit: the bisection node.** Every probability estimate is
  appended to ``probs.jsonl``, content-addressed by the conditioning prefix.

On resume a half-searched query is *not* restored -- it re-runs from the top,
and every node it revisits is a cache hit costing zero GPU. So no recursion
is ever serialized, and the only wasted work is the cheap tree walk.

Files are append-only and fsynced at wave boundaries (per-line fsync would
dominate runtime). Sessions get their own directory so that a HuggingFace
mirror never rewrites a file it has already uploaded.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from probe_pipeline.artifacts_io import AppendWriter, iter_jsonl

from .probability import ProbEstimate


@dataclass
class RunPaths:
    root: Path
    session: str

    @property
    def session_dir(self) -> Path:
        return self.root / "sessions" / self.session

    @property
    def events(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def probs(self) -> Path:
        return self.session_dir / "probs.jsonl"

    @property
    def manifest(self) -> Path:
        return self.session_dir / "manifest.jsonl"

    @property
    def heartbeat(self) -> Path:
        return self.session_dir / "heartbeat.json"


class RunStore:
    """Append-only store for one PTS run, readable across sessions."""

    def __init__(
        self,
        root: str | Path,
        *,
        session: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = RunPaths(
            Path(root), session or time.strftime("%Y%m%d-%H%M%S")
        )
        self.paths.session_dir.mkdir(parents=True, exist_ok=True)
        self._events: AppendWriter | None = None
        self._probs: AppendWriter | None = None
        self._manifest: AppendWriter | None = None

        if config is not None:
            cfg_path = self.paths.root / "config.json"
            if not cfg_path.exists():
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json.dumps(config, indent=2, default=str))

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "RunStore":
        self._events = AppendWriter(self.paths.events)
        self._probs = AppendWriter(self.paths.probs)
        self._manifest = AppendWriter(self.paths.manifest)
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()
        for w in (self._events, self._probs, self._manifest):
            if w is not None:
                w.close()

    def flush(self, *, fsync: bool = True) -> None:
        for w in (self._events, self._probs, self._manifest):
            if w is not None:
                w.flush(fsync=fsync)

    # -- reading across sessions ----------------------------------------

    def _all_sessions(self, name: str) -> Iterator[Path]:
        sessions = self.paths.root / "sessions"
        if not sessions.exists():
            return
        for d in sorted(sessions.iterdir()):
            p = d / name
            if p.exists():
                yield p

    def completed_ids(self) -> set[str]:
        """Query uids finished in *any* session of this run."""
        done: set[str] = set()
        for path in self._all_sessions("manifest.jsonl"):
            for row in iter_jsonl(path):
                uid = row.get("query_uid")
                if uid is not None:
                    done.add(str(uid))
        return done

    def load_prob_cache(self) -> dict[str, ProbEstimate]:
        """Every estimate from every session. This is what makes resume free."""
        cache: dict[str, ProbEstimate] = {}
        for path in self._all_sessions("probs.jsonl"):
            for row in iter_jsonl(path):
                est = ProbEstimate.from_dict(row)
                cache[est.key] = est
        return cache

    def load_events(self) -> list[dict]:
        """All events, deduped -- an interrupted query re-runs and may repeat."""
        seen: set[tuple] = set()
        out: list[dict] = []
        for path in self._all_sessions("events.jsonl"):
            for row in iter_jsonl(path):
                key = (row.get("query_uid"), row.get("position"), row.get("prefix_len"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(row)
        return out

    # -- writing ---------------------------------------------------------

    def record_probs(self, estimates: Iterable[ProbEstimate]) -> None:
        assert self._probs is not None, "use RunStore as a context manager"
        for est in estimates:
            self._probs.write(est.to_dict())

    def record_events(self, events: Sequence[Any]) -> None:
        assert self._events is not None
        for ev in events:
            self._events.write(ev.to_dict() if hasattr(ev, "to_dict") else ev)

    def complete_query(self, record: dict) -> None:
        assert self._manifest is not None
        self._manifest.write(record)

    def heartbeat(self, payload: dict) -> None:
        """Overwritten each wave; atomic so a reader never sees a half-write."""
        tmp = self.paths.heartbeat.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, self.paths.heartbeat)

    # -- export ----------------------------------------------------------

    def export_events(self, out_path: str | Path) -> Path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.load_events()
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return path
