"""Content-addressed artifact cache backed by local disk and a HuggingFace repo.

Every expensive result in this project -- PTS event files, activation caches,
GSM8K evaluations, probe weights -- goes through here, so that a Colab session
killed halfway through a sweep resumes at the next item instead of recomputing
from scratch, and so that a second session (or the SSH box) can pull what the
first one already produced.

The pattern this generalizes is the ``FORCE_RERUN`` / ``cache.exists()`` block
that was copy-pasted into every steering notebook::

    spec = ArtifactSpec("gsm8k_eval", cfg, tier="medium")
    result = store.load_or_compute(spec, lambda: run_eval(...))

Lookup order is memory -> local disk -> HuggingFace -> compute.

Design notes
------------
* **Importing this module must stay cheap.** No torch, no transformers, no
  huggingface_hub at import time; they are imported lazily inside the codecs
  and the hub methods. Tests run on a laptop without a GPU.
* **Keys are derived from the config that produced the bytes**, so changing a
  parameter invalidates the entry. Two escape hatches keep that from being
  maddening in practice: ``SCHEMA_VERSION`` (bump to invalidate everything
  after a semantics change) and ``VOLATILE_KEYS`` (things that must *not*
  invalidate, like which device you happen to be on).
* **Every artifact gets a sidecar ``.meta.json``** with the full config. A hash
  you cannot invert is a hash you cannot debug.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

# Bump this when the *meaning* of a cached artifact changes even though the
# config that names it does not -- e.g. after fixing a labelling bug. It is
# mixed into every key, so bumping it invalidates the whole cache at once.
SCHEMA_VERSION = 1

# Keys that describe *how* a computation ran rather than *what* it computed.
# Excluded from the hash so that moving between MPS and A100, or changing the
# log level, does not silently orphan every artifact you already have.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "device",
        "dtype_runtime",
        "output_dir",
        "out_dir",
        "cache_dir",
        "num_proc",
        "num_workers",
        "batch_size_runtime",
        "verbose",
        "tqdm",
        "progress",
        "log_level",
        "logger",
        "run_tag",
        "hf_token",
        "token",
        "_meta",
    }
)

Tier = Literal["small", "medium", "large", "huge"]
Codec = Literal["json", "jsonl", "npy", "npz", "safetensors", "torch", "text", "bytes"]

_EXTENSIONS: dict[str, str] = {
    "json": ".json",
    "jsonl": ".jsonl",
    "npy": ".npy",
    "npz": ".npz",
    "safetensors": ".safetensors",
    "torch": ".pt",
    "text": ".txt",
    "bytes": ".bin",
}


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def _canonical(obj: Any) -> Any:
    """Reduce a config value to something JSON-serializable and stable.

    Stability is the whole point: the same logical config must produce the
    same bytes across processes and platforms. Floats go through ``repr`` so
    0.1 does not drift, and arrays are summarized by a digest of their buffer
    rather than inlined.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {
            str(k): _canonical(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            if str(k) not in VOLATILE_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonical(v) for v in obj)

    # numpy / torch without importing either: duck-type on the buffer protocol.
    tobytes = getattr(obj, "tobytes", None)
    if tobytes is not None and hasattr(obj, "shape"):
        digest = hashlib.blake2b(tobytes(), digest_size=16).hexdigest()
        return {"__array__": digest, "shape": list(obj.shape), "dtype": str(obj.dtype)}

    detach = getattr(obj, "detach", None)
    if detach is not None and hasattr(obj, "shape"):  # torch.Tensor
        arr = detach().cpu().numpy()
        digest = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
        return {"__tensor__": digest, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    return {"__repr__": repr(obj)}


def config_hash(config: Mapping[str, Any], *, length: int = 12) -> str:
    """Stable short hash of a config, ignoring :data:`VOLATILE_KEYS`."""
    payload = {"__schema__": SCHEMA_VERSION, "config": _canonical(config)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()[:length]


def sha_of_array(arr: Any, *, length: int = 12) -> str:
    """Digest of an array-like, for putting a vector's identity into a config."""
    return str(_canonical(arr).get("__array__", _canonical(arr).get("__tensor__", "")))[:length]


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSpec:
    """Identifies one cacheable artifact.

    Args:
        name: Logical family, used as the directory. ``"pts_events"``,
            ``"acts"``, ``"gsm8k_eval"``, ``"probe"``.
        config: Everything that affects the bytes. Volatile keys are dropped.
        tier: Size class; controls where it lives and how it is pushed.
            See the module docstring in the project plan (Phase 0b).
        codec: On-disk format.
        ext: Override the extension inferred from ``codec``.
    """

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    tier: Tier = "small"
    codec: Codec = "json"
    ext: str | None = None

    @property
    def hash(self) -> str:
        return config_hash(self.config)

    @property
    def key(self) -> str:
        return f"{self.name}/{self.hash}"

    @property
    def filename(self) -> str:
        return f"{self.hash}{self.ext or _EXTENSIONS[self.codec]}"

    @property
    def relpath(self) -> Path:
        return Path(self.name) / self.filename


def artifact_key(spec: ArtifactSpec) -> str:
    return spec.key


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


def resolve_hf_token(*, required: bool = False) -> str | None:
    """Find a HuggingFace token without ever reading one from notebook source.

    Order: ``HF_TOKEN`` env -> Colab secret -> ``huggingface_hub`` cached login.
    Deliberately never falls back to a literal in the caller, so a token cannot
    end up committed inside an ``.ipynb``.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    # A gitignored .env at the repo root, for local runs.
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                return value.strip().strip("\"'")

    try:  # Colab secret: per-user, never serialized into the notebook JSON
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            return str(token).strip()
    except Exception:
        pass

    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return str(token).strip()
    except Exception:
        pass

    if required:
        raise RuntimeError(
            "No HuggingFace token found. Set HF_TOKEN, add it to Colab Secrets, "
            "or run `hf auth login`. Never paste a token into a notebook cell."
        )
    return None


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def default_cache_root() -> Path:
    """Where medium/large artifacts live. Colab gets local disk, never Drive.

    Drive is avoided on purpose: after a mount dies, writes silently land on
    the ephemeral VM disk and are lost, so a run can appear checkpointed and
    have nothing. The HF repo is the durable copy instead.
    """
    env = os.environ.get("PTRL_CACHE_ROOT")
    if env:
        return Path(env)
    if in_colab():
        return Path("/content/ptrl-cache")
    return Path.home() / ".cache" / "ptrl" / "artifacts"


# --------------------------------------------------------------------------
# codecs
# --------------------------------------------------------------------------


def _write(path: Path, obj: Any, codec: Codec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    if codec == "json":
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    elif codec == "jsonl":
        with tmp.open("w", encoding="utf-8") as fh:
            for row in obj:
                fh.write(json.dumps(row, default=str) + "\n")
    elif codec == "text":
        tmp.write_text(str(obj), encoding="utf-8")
    elif codec == "bytes":
        tmp.write_bytes(obj)
    elif codec == "npy":
        import numpy as np

        np.save(tmp, obj, allow_pickle=False)
        # np.save appends .npy if the name lacks it; normalize back.
        appended = Path(str(tmp) + ".npy")
        if appended.exists():
            appended.replace(tmp)
    elif codec == "npz":
        import numpy as np

        np.savez_compressed(tmp, **obj)
        appended = Path(str(tmp) + ".npz")
        if appended.exists():
            appended.replace(tmp)
    elif codec == "safetensors":
        from safetensors.torch import save_file

        save_file(obj, str(tmp))
    elif codec == "torch":
        import torch

        torch.save(obj, tmp)
    else:  # pragma: no cover - guarded by the Literal
        raise ValueError(f"Unknown codec: {codec}")

    os.replace(tmp, path)  # atomic on POSIX


def _read(path: Path, codec: Codec) -> Any:
    if codec == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if codec == "jsonl":
        return list(iter_jsonl(path))
    if codec == "text":
        return path.read_text(encoding="utf-8")
    if codec == "bytes":
        return path.read_bytes()
    if codec == "npy":
        import numpy as np

        return np.load(path, allow_pickle=False)
    if codec == "npz":
        import numpy as np

        return dict(np.load(path, allow_pickle=False))
    if codec == "safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    if codec == "torch":
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unknown codec: {codec}")  # pragma: no cover


def iter_jsonl(path: Path, *, tolerant: bool = True) -> Iterator[dict]:
    """Read JSONL, tolerating one torn final line.

    A process killed mid-write (Colab session limit) leaves a partial last
    line. That is expected and recoverable. A malformed line *anywhere else*
    is corruption and raises, because silently skipping it would quietly drop
    real results.
    """
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    last = len(lines) - 1
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if tolerant and i == last:
                return  # torn final write
            raise


class AppendWriter:
    """Append-only JSONL writer that fsyncs at explicit checkpoints.

    Per-line fsync would dominate runtime; per-wave fsync is what actually
    survives a session kill.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(row, default=str) + "\n")

    def extend(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self.write(row)

    def flush(self, *, fsync: bool = True) -> None:
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "AppendWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


class ArtifactStore:
    """Local-disk cache with an optional HuggingFace mirror.

    Args:
        local_root: Directory for ``medium``/``large`` artifacts. Defaults to
            :func:`default_cache_root`.
        repo_root: Directory for ``small`` artifacts, which are git-tracked so
            notebooks can fetch them over raw.githubusercontent. Defaults to
            ``<project>/artifacts``.
        repo_id: HuggingFace repo to mirror to, e.g. ``"user/ptrl-artifacts"``.
        push: Whether ``save`` uploads. Turn off for local experimentation.
        offline: Never touch the network, even to read.
    """

    def __init__(
        self,
        local_root: str | Path | None = None,
        repo_root: str | Path | None = None,
        repo_id: str | None = None,
        repo_type: str = "dataset",
        revision: str = "main",
        *,
        push: bool = True,
        offline: bool = False,
        mem_cache: bool = True,
    ) -> None:
        self.local_root = Path(local_root) if local_root else default_cache_root()
        self.repo_root = (
            Path(repo_root)
            if repo_root
            else Path(__file__).resolve().parent.parent / "artifacts"
        )
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.push = push and repo_id is not None
        self.offline = offline
        self._mem: dict[str, Any] | None = {} if mem_cache else None

    # -- paths ------------------------------------------------------------

    def root_for(self, tier: Tier) -> Path:
        return self.repo_root if tier == "small" else self.local_root

    def path(self, spec: ArtifactSpec) -> Path:
        return self.root_for(spec.tier) / spec.relpath

    def meta_path(self, spec: ArtifactSpec) -> Path:
        p = self.path(spec)
        return p.with_name(p.name + ".meta.json")

    # -- presence ---------------------------------------------------------

    def exists(self, spec: ArtifactSpec) -> bool:
        if self._mem is not None and spec.key in self._mem:
            return True
        if self.path(spec).exists():
            return True
        return self._hub_download(spec) is not None

    # -- hub --------------------------------------------------------------

    def _hub_download(self, spec: ArtifactSpec) -> Path | None:
        """Try to pull one artifact from the hub into the local path."""
        if self.offline or not self.repo_id:
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None
        try:
            fetched = hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                filename=str(spec.relpath).replace(os.sep, "/"),
                token=resolve_hf_token(),
            )
        except Exception:
            return None

        dest = self.path(spec)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(fetched, dest)
        return dest

    def _hub_upload(self, spec: ArtifactSpec) -> None:
        if not self.push or self.offline:
            return
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return
        token = resolve_hf_token()
        if token is None:
            return
        api = HfApi(token=token)
        for local in (self.path(spec), self.meta_path(spec)):
            if not local.exists():
                continue
            rel = local.relative_to(self.root_for(spec.tier))
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=str(rel).replace(os.sep, "/"),
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                )
            except Exception:
                # A failed push must never lose a completed computation; the
                # local copy is authoritative and sync() can retry later.
                pass

    # -- read / write -----------------------------------------------------

    def load(self, spec: ArtifactSpec) -> Any:
        if self._mem is not None and spec.key in self._mem:
            return self._mem[spec.key]

        path = self.path(spec)
        if not path.exists():
            if self._hub_download(spec) is None:
                raise FileNotFoundError(f"No artifact for {spec.key} at {path}")

        obj = _read(path, spec.codec)
        if self._mem is not None:
            self._mem[spec.key] = obj
        return obj

    def save(self, spec: ArtifactSpec, obj: Any, *, push: bool | None = None) -> Path:
        path = self.path(spec)
        _write(path, obj, spec.codec)

        meta = {
            "key": spec.key,
            "name": spec.name,
            "tier": spec.tier,
            "codec": spec.codec,
            "schema_version": SCHEMA_VERSION,
            "config": _canonical(spec.config),
            "created_at": _now(),
            "git_sha": _git_sha(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        }
        _write(self.meta_path(spec), meta, "json")

        if self._mem is not None:
            self._mem[spec.key] = obj
        if push is None or push:
            self._hub_upload(spec)
        return path

    def load_or_compute(
        self,
        spec: ArtifactSpec,
        fn: Callable[[], Any],
        *,
        force: bool = False,
    ) -> Any:
        """Return the cached artifact, computing and storing it if absent."""
        if not force:
            try:
                return self.load(spec)
            except FileNotFoundError:
                pass
        obj = fn()
        self.save(spec, obj)
        return obj

    def open_append(self, spec: ArtifactSpec) -> AppendWriter:
        """Append-only writer for streaming artifacts (PTS events, probs)."""
        return AppendWriter(self.path(spec))

    def sync(self) -> int:
        """Push every locally-present artifact to the hub. Returns the count.

        Used at the end of a session, or to recover after pushes failed while
        the network was down.
        """
        if not self.push or self.offline:
            return 0
        pushed = 0
        for root in (self.repo_root, self.local_root):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    self._hub_upload_path(path, root)
                    pushed += 1
        return pushed

    def _hub_upload_path(self, local: Path, root: Path) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return
        token = resolve_hf_token()
        if token is None:
            return
        try:
            HfApi(token=token).upload_file(
                path_or_fileobj=str(local),
                path_in_repo=str(local.relative_to(root)).replace(os.sep, "/"),
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
            )
        except Exception:
            pass


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""
