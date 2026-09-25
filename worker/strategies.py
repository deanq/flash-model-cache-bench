"""Model-acquisition strategies under test.

Each strategy turns a Hugging Face repo id into a local snapshot directory that
`from_pretrained` can load, and reports how long each step took. They differ
only in *where the bytes come from*:

- nocache      fresh download from the Hub into container disk, every cold start
- volume       HF cache lives on the network volume; load straight off the mount
- volumecache  HF cache on container disk, mirrored to the volume by VolumeCache
- modelstore   Runpod cached model, pre-staged by the platform
- globalstore  Runpod Global Store (object-backed, geesefs) mounted at /runpod-volume;
               load straight off the mount
- gsvolumecache  HF cache on container disk, mirrored to Global Store by VolumeCache

Every strategy resolves offline first and only calls the Hub when the files are
not already usable -- that is the best-practice version of each strategy, and
`hub_online` in the result records when a strategy could not avoid the network.

Local caches deliberately live outside /root/.cache: flash-worker tars new files
under /root/.cache onto an attached network volume, which would silently turn
every strategy into a volume-backed one.

This module is imported by the Flash worker as a top-level module (worker/ is
the deploy root), and by the unit tests.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

STRATEGIES = ("nocache", "volume", "volumecache", "modelstore", "globalstore", "gsvolumecache")

VOLUME_ROOT = "/runpod-volume"
LOCAL_ROOT = "/opt/hfbench"                       # container disk, outside /root/.cache
VOLUME_HF_CACHE = f"{VOLUME_ROOT}/hfbench/hub"    # `volume` strategy's HF cache
MODELSTORE_HUB = f"{VOLUME_ROOT}/huggingface-cache/hub"
GLOBALSTORE_MODELS = f"{VOLUME_ROOT}/hfbench/models"   # plain files, not an HF cache (see acquire_globalstore)
COMPLETE_MARKER = ".hfbench-complete"
DEFAULT_REVISION = "main"
POPULATION_MIN_BYTES = 1 << 20


@dataclass
class Acquired:
    """Where the model ended up and what it cost to get it there."""
    path: str
    timings: dict[str, float] = field(default_factory=dict)
    bytes_total: int = 0
    hub_online: bool = False          # had to talk to the Hub (metadata and/or bytes)
    notes: dict[str, object] = field(default_factory=dict)
    shared_storage_files: list[str] = field(default_factory=list)  # read from the volume at load


def repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _find_repo_dir(hub_root: Path, repo_id: str) -> Path | None:
    """HF cache dir for repo_id; case-insensitive because Model Store lowercases refs."""
    exact = hub_root / repo_dir_name(repo_id)
    if exact.is_dir():
        return exact
    if not hub_root.is_dir():
        return None
    wanted = repo_dir_name(repo_id).lower()
    matches = [p for p in hub_root.iterdir() if p.is_dir() and p.name.lower() == wanted]
    return matches[0] if len(matches) == 1 else None


def resolve_snapshot(hub_root: str | Path, repo_id: str, revision: str = DEFAULT_REVISION) -> Path | None:
    """Offline snapshot lookup in an HF-layout cache: refs/<rev> if present, else the
    single snapshot. Returns None when nothing loadable is there (no config.json)."""
    repo = _find_repo_dir(Path(hub_root), repo_id)
    if repo is None:
        return None
    ref = repo / "refs" / revision
    snap: Path | None = None
    if ref.is_file():
        snap = repo / "snapshots" / ref.read_text().strip()
    else:
        snaps = sorted(p for p in (repo / "snapshots").glob("*") if p.is_dir())
        snap = snaps[0] if len(snaps) == 1 else None
    if snap is None or not (snap / "config.json").exists():
        return None
    return snap


def snapshot_files(snap: Path) -> list[Path]:
    """Real files behind a snapshot (symlinks resolved to blobs)."""
    return [p.resolve() for p in snap.rglob("*") if p.is_file()]


def tree_bytes(root: str | Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def _download(repo_id: str, cache_dir: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, cache_dir=cache_dir, token=token,
                                  allow_patterns=ALLOW_PATTERNS))


# Weights + tokenizer + configs; skip duplicate formats (.bin/.pth/.gguf/onnx).
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.py"]


def _resolve_or_download(repo_id: str, hub_root: str, token: str | None, t: dict) -> tuple[Path, bool]:
    t0 = time.monotonic()
    snap = resolve_snapshot(hub_root, repo_id)
    if snap is not None:
        t["resolve_s"] = time.monotonic() - t0
        return snap, False
    t1 = time.monotonic()
    snap = _download(repo_id, hub_root, token)
    t["download_s"] = time.monotonic() - t1
    return snap, True


def acquire_nocache(repo_id: str, token: str | None = None, *, local_root: str = LOCAL_ROOT) -> Acquired:
    """Download into a brand-new directory: no cache of any kind survives between cold starts."""
    stale = [p for p in Path(local_root).glob("nocache-*") if p.is_dir()]
    leftover = sum(tree_bytes(p) for p in stale)
    for p in stale:                    # FlashBoot restarts can keep container disk
        shutil.rmtree(p, ignore_errors=True)
    cache_dir = os.path.join(local_root, f"nocache-{uuid.uuid4().hex[:8]}")
    t: dict[str, float] = {}
    t0 = time.monotonic()
    snap = _download(repo_id, cache_dir, token)
    t["download_s"] = time.monotonic() - t0
    return Acquired(path=str(snap), timings=t, bytes_total=tree_bytes(cache_dir), hub_online=True,
                    notes={"cache_dir": cache_dir, "leftover_bytes": leftover})


def acquire_volume(repo_id: str, token: str | None = None, *, hub_root: str = VOLUME_HF_CACHE) -> Acquired:
    """HF cache on the network volume; the model is read off the mount at load time."""
    t: dict[str, float] = {}
    snap, downloaded = _resolve_or_download(repo_id, hub_root, token, t)
    return Acquired(path=str(snap), timings=t, bytes_total=sum(p.stat().st_size for p in snapshot_files(snap)),
                    hub_online=downloaded, notes={"populated_now": downloaded},
                    shared_storage_files=[str(p) for p in snapshot_files(snap)])


def load_volumecache_class():
    """Prefer the runpod SDK's VolumeCache; fall back to the vendored 1.12.0 copy."""
    try:
        from runpod.serverless import VolumeCache  # runpod>=1.12.0
        return VolumeCache, "runpod"
    except ImportError:
        from vendor.rp_volume_cache import VolumeCache
        return VolumeCache, "vendored-1.12.0"


def acquire_volumecache(repo_id: str, token: str | None = None, *, namespace: str,
                        local_root: str = LOCAL_ROOT, volume_path: str = VOLUME_ROOT,
                        cache_cls=None) -> Acquired:
    """HF cache on container disk; VolumeCache hydrates it from the volume mirror before
    load and (on first population) syncs it back synchronously."""
    impl = "injected"
    if cache_cls is None:
        cache_cls, impl = load_volumecache_class()
    hub_root = os.path.join(local_root, "vc", "hub")
    leftover = tree_bytes(hub_root)
    # A restarted (FlashBoot) container can keep its disk. Wipe it so every cold
    # start measures the strategy, not a leftover local copy; `leftover_bytes`
    # records how much a real worker would have been able to reuse.
    shutil.rmtree(os.path.join(local_root, "vc"), ignore_errors=True)
    os.makedirs(hub_root, exist_ok=True)
    vc = cache_cls(dirs=[hub_root], namespace=namespace, volume_path=volume_path, best_effort=False)
    t: dict[str, float] = {}
    t0 = time.monotonic()
    restored = vc.hydrate()
    t["hydrate_s"] = time.monotonic() - t0
    hydrated_bytes = tree_bytes(hub_root)
    snap, _ = _resolve_or_download_linked(repo_id, hub_root, token, t)
    fetched = tree_bytes(hub_root) - hydrated_bytes
    synced = False
    mirror_bytes = None
    # Relinking after hydrate writes only refs/ (~100 bytes, excluded from the
    # mirror anyway); real model bytes mean this was the first population, so
    # write the mirror before returning so the next cold start can use it.
    if fetched > POPULATION_MIN_BYTES:
        t1 = time.monotonic()
        vc.sync(background=False)
        t["sync_s"] = time.monotonic() - t1
        synced = True
        mirror_bytes = tree_bytes(Path(volume_path) / ".cache" / namespace)
    return Acquired(path=str(snap), timings=t, bytes_total=tree_bytes(hub_root),
                    hub_online="hub_s" in t or "download_s" in t,
                    notes={"impl": impl, "files_hydrated": restored, "hydrated_bytes": hydrated_bytes,
                           "hub_fetched_bytes": fetched, "synced": synced, "leftover_bytes": leftover,
                           "mirror_bytes_after_sync": mirror_bytes})


def _resolve_or_download_linked(repo_id: str, hub_root: str, token: str | None, t: dict) -> tuple[Path, bool]:
    """Like _resolve_or_download, but VolumeCache restores blobs without snapshot
    symlinks or refs/, so an offline miss here usually means "relink via Hub
    metadata", not "download bytes". Time it as hub_s either way; bytes fetched
    are measured by the caller."""
    t0 = time.monotonic()
    snap = resolve_snapshot(hub_root, repo_id)
    if snap is not None:
        t["resolve_s"] = time.monotonic() - t0
        return snap, False
    t1 = time.monotonic()
    snap = _download(repo_id, hub_root, token)
    t["hub_s"] = time.monotonic() - t1
    return snap, True


def acquire_modelstore(repo_id: str, token: str | None = None, *, hub_root: str = MODELSTORE_HUB) -> Acquired:
    """Runpod cached model: the platform staged it before the worker started."""
    t0 = time.monotonic()
    snap = resolve_snapshot(hub_root, repo_id)
    if snap is None:
        listing = sorted(p.name for p in Path(hub_root).iterdir()) if Path(hub_root).is_dir() else "missing"
        raise FileNotFoundError(
            f"cached model {repo_id} not staged under {hub_root} (contents: {listing}); "
            f"is modelReferences set on this endpoint?")
    files = snapshot_files(snap)
    return Acquired(path=str(snap), timings={"resolve_s": time.monotonic() - t0},
                    bytes_total=sum(p.stat().st_size for p in files),
                    notes={"snapshot": str(snap)}, shared_storage_files=[str(p) for p in files])


def acquire_globalstore(repo_id: str, token: str | None = None, *, models_root: str = GLOBALSTORE_MODELS,
                        stage_root: str = LOCAL_ROOT) -> Acquired:
    """Load straight off Global Store.

    Global Store cannot hold an HF hub cache: snapshot symlinks written by one
    worker come back as 0-byte regular files on the next worker's mount, so the
    snapshot "resolves" and then fails to load. The store therefore holds a
    plain directory of real files, populated once by staging a download on
    container disk and copying with copyfile (chmod/utime are EPERM there). A
    marker written last makes a half-copied directory count as absent.
    """
    target = Path(models_root) / repo_id.replace("/", "--")
    t: dict[str, float] = {}
    t0 = time.monotonic()
    populated = False
    if not (target / COMPLETE_MARKER).is_file():
        t1 = time.monotonic()
        stage = Path(stage_root) / "gs-stage"
        shutil.rmtree(stage, ignore_errors=True)
        snap = _download(repo_id, str(stage), token)
        t["download_s"] = time.monotonic() - t1
        t2 = time.monotonic()
        copy_real_files(snap, target)
        (target / COMPLETE_MARKER).write_text(repo_id)
        t["upload_s"] = time.monotonic() - t2
        shutil.rmtree(stage, ignore_errors=True)
        populated = True
    else:
        t["resolve_s"] = time.monotonic() - t0
    files = [p for p in target.iterdir() if p.is_file() and p.name != COMPLETE_MARKER]
    return Acquired(path=str(target), timings=t, bytes_total=sum(p.stat().st_size for p in files),
                    hub_online=populated, notes={"populated_now": populated},
                    shared_storage_files=[str(p) for p in files])


def copy_real_files(snapshot: Path, target: Path) -> int:
    """Copy a snapshot's files (symlinks resolved) into a flat directory; data only."""
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        dst = target / src.relative_to(snapshot)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src.resolve(), dst)
        copied += dst.stat().st_size
    return copied


def global_store_volumecache_class():
    """VolumeCache that works on Global Store.

    Stock VolumeCache mirrors big files with shutil.copy2, whose copystat
    (chmod/utime) raises EPERM on Global Store; _copy_file swallows the error,
    so every file >= 256 KiB is silently left out of the mirror and each cold
    start re-downloads the weights. This subclass copies data with copyfile and
    applies metadata only where the filesystem allows it. Mirror mtimes then
    reflect upload time (newer than the source), which the size/mtime diff
    tolerates: hydrate copies them because the local file is missing, and
    copystat onto local disk makes the next comparison converge.
    """
    import threading

    base, impl = load_volumecache_class()

    class GlobalStoreVolumeCache(base):
        def _copy_file(self, src, dst):
            tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.rpvc.tmp"
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, tmp)
                try:
                    shutil.copystat(src, tmp)
                except OSError:
                    pass      # Global Store: no chmod/utime; data is what matters
                os.replace(tmp, dst)
                return True
            except OSError:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return False

    return GlobalStoreVolumeCache, f"{impl}+globalstore-copyfile"


def evict_page_cache(paths: list[str]) -> int:
    """posix_fadvise(DONTNEED) each file so the next read hits storage, not RAM.
    Needs no root. Returns bytes advised. Userspace caches (e.g. a FUSE client's)
    are out of reach -- the report says so."""
    if not hasattr(os, "posix_fadvise"):   # macOS (unit tests); workers are Linux
        return 0
    advised = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            advised += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return advised


def volumecache_mirror_files(namespace: str, volume_path: str = VOLUME_ROOT) -> list[str]:
    root = Path(volume_path) / ".cache" / namespace
    return [str(p) for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
