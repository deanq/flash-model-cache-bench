"""Model-acquisition strategies under test.

Each strategy turns a Hugging Face repo id into a local snapshot directory that
`from_pretrained` can load, and reports how long each step took. They differ
only in *where the bytes come from*:

- nocache      fresh download from the Hub into container disk, every cold start
- volume       HF cache lives on the network volume; load straight off the mount
- volumecache  HF cache on container disk, mirrored to the volume by VolumeCache
               (SDK default hydrate settings)
- volumecache864  same mirror as `volumecache`, hydrated with explicit 8 x 64 MiB
               range reads (the network-volume default on runpod-python main, #691)
- modelstore   Runpod cached model, pre-staged by the platform
- globalstore  Runpod Global Store (object-backed, geesefs) mounted at /runpod-volume;
               load straight off the mount
- gsvolumecache  HF cache on container disk, mirrored to Global Store by VolumeCache
- gshydrate    plain model folder on Global Store, copied to container disk with many
               concurrent range reads, then loaded locally
- volumeprefetch  same network-volume HF cache as `volume`, but read into the page cache
               with concurrent range reads first (vLLM's `prefetch` load strategy), then
               loaded in place; isolates the access pattern from the storage

Every strategy resolves offline first and only calls the Hub when the files are
not already usable -- that is the best-practice version of each strategy, and
`hub_online` in the result records when a strategy could not avoid the network.

Local caches deliberately live outside /root/.cache: the Flash 1.x worker tarred
new files under /root/.cache onto an attached network volume, which would
silently turn every strategy into a volume-backed one. Keeping them elsewhere
stays correct whatever the runtime does with that directory.

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

STRATEGIES = ("nocache", "volume", "volumecache", "modelstore", "globalstore", "gsvolumecache", "gshydrate",
              "volumeprefetch", "volumecache864")

VOLUME_ROOT = "/runpod-volume"
LOCAL_ROOT = "/opt/hfbench"                       # container disk, outside /root/.cache
VOLUME_HF_CACHE = f"{VOLUME_ROOT}/hfbench/hub"    # `volume` strategy's HF cache
MODELSTORE_HUB = f"{VOLUME_ROOT}/huggingface-cache/hub"
GLOBALSTORE_MODELS = f"{VOLUME_ROOT}/hfbench/models"   # plain files, not an HF cache (see acquire_globalstore)
COMPLETE_MARKER = ".hfbench-complete"
DEFAULT_REVISION = "main"
POPULATION_MIN_BYTES = 1 << 20

# gshydrate tuning, measured on pods (gv-hf-bench): geesefs serves each read as an
# object GET with ~0.5-2s latency, so throughput comes from concurrency, not from
# sequential readahead (~140 MiB/s per mount regardless of how many files are read
# at once). What matters is bytes in flight (workers x chunk): 64 x 8 MiB reached
# ~1.1 GiB/s (the host NIC) on quiet hosts, while 96 x 16 MiB overflowed geesefs'
# buffer pool on a busy 8-GPU host (ENOMEM storm, throughput fell to ~100 MiB/s).
HYDRATE_WORKERS = 64
HYDRATE_CHUNK_BYTES = 8 << 20
# volumeprefetch tuning: the network-volume optimum measured for VolumeCache's hydrate
# (testbeds-runpod-2.1: 8 x 64 MiB ~1000-1400 MiB/s vs 64 x 8 MiB ~360 MiB/s).
PREFETCH_WORKERS = 8
PREFETCH_CHUNK_BYTES = 64 << 20
# volumecache864: the same tuning passed to VolumeCache explicitly. runpod 2.2.1
# hydrates with 64 x 8 MiB (tuned for Global Store); main's #691 makes 8 x 64 MiB
# the network-volume default. Keyword names are VolumeCache's (runpod >= 2.1.0).
VOLUMECACHE_864 = {"hydrate_workers": PREFETCH_WORKERS, "hydrate_chunk_size": PREFETCH_CHUNK_BYTES}
HYDRATE_RETRY_ERRNOS = frozenset({5, 11, 12})   # EIO, EAGAIN, ENOMEM: geesefs buffer pool exhausted
HYDRATE_MAX_ATTEMPTS = 8
HYDRATE_RETRY_BASE_S = 0.05


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


def acquire_volumeprefetch(repo_id: str, token: str | None = None, *, hub_root: str | None = None,
                           workers: int = PREFETCH_WORKERS, chunk_bytes: int = PREFETCH_CHUNK_BYTES) -> Acquired:
    """`volume`, plus a concurrent read of every snapshot file into the page cache before
    load, so the lazy (mmap) safetensors load is served from RAM, not the volume.

    shared_storage_files is empty on purpose: evicting after acquire would throw away the
    prefetch. The caller evicts before (see runtime._acquire)."""
    acq = acquire_volume(repo_id, token, hub_root=hub_root or VOLUME_HF_CACHE)
    files = acq.shared_storage_files
    t0 = time.monotonic()
    stats = prefetch_files(files, workers=workers, chunk_bytes=chunk_bytes)
    acq.timings["prefetch_s"] = time.monotonic() - t0
    acq.notes.update(prefetch=stats, prefetched_files=files, cgroup_memory_max=cgroup_memory_limit())
    acq.shared_storage_files = []
    return acq


def cgroup_memory_limit(path: str = "/sys/fs/cgroup/memory.max") -> int | str | None:
    """Container memory limit (cgroup v2). Page cache counts against it, so a limit
    below the model size means prefetched pages are evicted before the load reads them."""
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else raw


def prefetch_files(paths: list[str], *, workers: int, chunk_bytes: int) -> dict:
    """Read every byte of `paths` with `workers` concurrent `chunk_bytes` preads and
    discard it: the point is the kernel page cache, not the data. A short read raises
    with the file and offset. Returns {"files", "bytes", "ranges", "workers", "chunk_bytes"}."""
    from concurrent.futures import ThreadPoolExecutor

    work = [(p, off, min(chunk_bytes, size - off))
            for p in paths for size in (os.stat(p).st_size,) for off in range(0, size, chunk_bytes)]
    work.sort(key=lambda w: (w[1], w[0]))      # spread concurrent readers across all files

    def run(item: tuple[str, int, int]) -> int:
        path, off, length = item
        fd = os.open(path, os.O_RDONLY)
        try:
            done = 0
            while done < length:
                n = len(os.pread(fd, min(length - done, chunk_bytes), off + done))
                if not n:
                    raise OSError(5, f"prefetch {path} @ {off}: short read, got {done} of {length} bytes")
                done += n
            return done
        finally:
            os.close(fd)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        read = sum(ex.map(run, work))
    return {"files": len(paths), "bytes": read, "ranges": len(work), "workers": workers, "chunk_bytes": chunk_bytes}


def load_volumecache_class():
    """The runpod SDK's VolumeCache. `rp flash deploy` vendors the deploying client's
    runpod package (>= 2.1.0) into the artifact, so the worker always has it."""
    from runpod.serverless import VolumeCache

    return VolumeCache, f"runpod-{runpod_code_version()}"


def runpod_code_version() -> str:
    """Version of the runpod code actually imported. `runpod.__version__` reads the
    installed dist metadata, which on a Flash worker is the image's runpod (2.0.0 on
    the 2026-11 images) even though the artifact's vendored 2.1.0 tree is what runs."""
    try:
        from runpod._version import __version__
        return __version__
    except ImportError:
        import runpod
        return f"{runpod.__version__}(metadata)"


def acquire_volumecache(repo_id: str, token: str | None = None, *, namespace: str,
                        local_root: str = LOCAL_ROOT, volume_path: str = VOLUME_ROOT,
                        cache_cls=None, hydrate_settings: dict | None = None) -> Acquired:
    """HF cache on container disk; VolumeCache hydrates it from the volume mirror before
    load and (on first population) syncs it back synchronously.

    hydrate_settings (VolumeCache keywords) are passed only when given, so the stock
    strategy measures whatever default the deployed runpod release ships."""
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
    vc = cache_cls(dirs=[hub_root], namespace=namespace, volume_path=volume_path, best_effort=False,
                   **(hydrate_settings or {}))
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
                           "mirror_bytes_after_sync": mirror_bytes,
                           "hydrate_settings": dict(hydrate_settings) if hydrate_settings else "sdk-default"})


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
            f"is model= (modelReferences) set on this endpoint?")
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
    t: dict[str, float] = {}
    target, populated = _ensure_globalstore_copy(repo_id, token, models_root, stage_root, t)
    files = _globalstore_files(target)
    return Acquired(path=str(target), timings=t, bytes_total=sum(p.stat().st_size for p in files),
                    hub_online=populated, notes={"populated_now": populated},
                    shared_storage_files=[str(p) for p in files])


def _ensure_globalstore_copy(repo_id: str, token: str | None, models_root: str, stage_root: str,
                             t: dict) -> tuple[Path, bool]:
    """The model as a plain folder on Global Store; populated from the Hub on first use."""
    target = Path(models_root) / repo_id.replace("/", "--")
    t0 = time.monotonic()
    if (target / COMPLETE_MARKER).is_file():
        t["resolve_s"] = time.monotonic() - t0
        return target, False
    stage = Path(stage_root) / "gs-stage"
    shutil.rmtree(stage, ignore_errors=True)
    snap = _download(repo_id, str(stage), token)
    t["download_s"] = time.monotonic() - t0
    t1 = time.monotonic()
    copy_real_files(snap, target)
    (target / COMPLETE_MARKER).write_text(repo_id)
    t["upload_s"] = time.monotonic() - t1
    shutil.rmtree(stage, ignore_errors=True)
    return target, True


def _globalstore_files(target: Path) -> list[Path]:
    return sorted(p for p in target.rglob("*") if p.is_file() and p.name != COMPLETE_MARKER)


def acquire_gshydrate(repo_id: str, token: str | None = None, *, models_root: str = GLOBALSTORE_MODELS,
                      local_root: str = LOCAL_ROOT, workers: int = HYDRATE_WORKERS,
                      chunk_bytes: int = HYDRATE_CHUNK_BYTES) -> Acquired:
    """Copy the Global Store folder to container disk with concurrent range reads, load locally."""
    t: dict[str, float] = {}
    target, populated = _ensure_globalstore_copy(repo_id, token, models_root, local_root, t)
    local = Path(local_root) / "gshydrate" / target.name
    leftover = tree_bytes(local)
    shutil.rmtree(local, ignore_errors=True)       # FlashBoot restarts can keep container disk
    t0 = time.monotonic()
    stats = parallel_hydrate(target, local, workers=workers, chunk_bytes=chunk_bytes)
    t["hydrate_s"] = time.monotonic() - t0
    return Acquired(path=str(local), timings=t, bytes_total=stats["bytes"], hub_online=populated,
                    notes={"populated_now": populated, "leftover_bytes": leftover, "hydrate": stats},
                    shared_storage_files=[str(p) for p in _globalstore_files(target)])


def parallel_hydrate(src: Path, dst: Path, *, workers: int, chunk_bytes: int) -> dict:
    """Copy a directory tree with `workers` concurrent pread/pwrite range copies.

    Each range opens its own descriptor, so geesefs serves ranges as independent
    concurrent GETs instead of one sequential readahead stream. Transient FUSE
    errors (ENOMEM etc. under load) are retried with backoff; anything else raises
    with the file and offset. Returns {"files", "bytes", "ranges", "retries"}.
    """
    from concurrent.futures import ThreadPoolExecutor

    src, dst = Path(src), Path(dst)
    files = _globalstore_files(src)
    work: list[tuple[Path, Path, int, int]] = []
    for f in files:
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        size = f.stat().st_size
        with open(out, "wb") as fh:
            fh.truncate(size)
        work.extend((f, out, off, min(chunk_bytes, size - off)) for off in range(0, size, chunk_bytes))
    work.sort(key=lambda w: (w[2], str(w[0])))      # spread concurrent readers across all files
    retries = [0]

    def run(item: tuple[Path, Path, int, int]) -> int:
        for attempt in range(HYDRATE_MAX_ATTEMPTS):
            try:
                return _copy_range(*item)
            except OSError as exc:
                if exc.errno not in HYDRATE_RETRY_ERRNOS or attempt == HYDRATE_MAX_ATTEMPTS - 1:
                    raise OSError(exc.errno, f"hydrate {item[0]} @ {item[2]}: {exc.strerror}") from exc
                retries[0] += 1
                time.sleep(HYDRATE_RETRY_BASE_S * (2 ** attempt))
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        copied = sum(ex.map(run, work))
    return {"files": len(files), "bytes": copied, "ranges": len(work), "retries": retries[0],
            "workers": workers, "chunk_bytes": chunk_bytes}


def _copy_range(src: Path, dst: Path, off: int, length: int) -> int:
    sfd = os.open(src, os.O_RDONLY)
    try:
        dfd = os.open(dst, os.O_WRONLY)
        try:
            done = 0
            while done < length:
                buf = os.pread(sfd, length - done, off + done)
                if not buf:
                    raise OSError(5, f"short read: got {done} of {length} bytes")
                os.pwrite(dfd, buf, off + done)
                done += len(buf)
            return done
        finally:
            os.close(dfd)
    finally:
        os.close(sfd)


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


def volume_snapshot_files(repo_id: str, hub_root: str | None = None) -> list[str]:
    """Files of the `volume` HF cache snapshot, or [] before it is populated."""
    snap = resolve_snapshot(hub_root or VOLUME_HF_CACHE, repo_id)
    return [str(p) for p in snapshot_files(snap)] if snap else []


def volumecache_mirror_files(namespace: str, volume_path: str = VOLUME_ROOT) -> list[str]:
    root = Path(volume_path) / ".cache" / namespace
    return [str(p) for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
