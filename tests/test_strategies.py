import os
from pathlib import Path

import pytest

import strategies as st

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"


def make_hf_cache(hub: Path, repo: str = REPO, *, refs: bool = True, symlinks: bool = True,
                  dirname: str | None = None, weight_bytes: int = 4096) -> Path:
    """Minimal HF hub-cache layout: blobs/, snapshots/<commit>/ (symlinks), refs/main."""
    base = hub / (dirname or st.repo_dir_name(repo))
    blobs, snap = base / "blobs", base / "snapshots" / COMMIT
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, data in {"config.json": b"{}", "model.safetensors": b"x" * weight_bytes}.items():
        blob = blobs / f"blob-{name}"
        blob.write_bytes(data)
        if symlinks:
            (snap / name).symlink_to(os.path.relpath(blob, snap))
    if refs:
        (base / "refs").mkdir()
        (base / "refs" / "main").write_text(COMMIT)
    return snap


def test_resolve_snapshot_via_refs(tmp_path):
    snap = make_hf_cache(tmp_path)
    assert st.resolve_snapshot(tmp_path, REPO) == snap


def test_resolve_snapshot_without_refs_uses_single_snapshot(tmp_path):
    snap = make_hf_cache(tmp_path, refs=False)
    assert st.resolve_snapshot(tmp_path, REPO) == snap


def test_resolve_snapshot_is_case_insensitive_for_model_store(tmp_path):
    make_hf_cache(tmp_path, dirname="models--qwen--qwen2.5-0.5b-instruct")
    assert st.resolve_snapshot(tmp_path, REPO) is not None


def test_resolve_snapshot_missing_or_unlinked_returns_none(tmp_path):
    assert st.resolve_snapshot(tmp_path / "nope", REPO) is None
    make_hf_cache(tmp_path, symlinks=False, refs=False)   # what a VolumeCache hydrate leaves behind
    assert st.resolve_snapshot(tmp_path, REPO) is None


def test_snapshot_files_resolve_symlinks_to_blobs(tmp_path):
    snap = make_hf_cache(tmp_path)
    files = st.snapshot_files(snap)
    assert all("blobs" in str(p) for p in files) and len(files) == 2


def test_modelstore_missing_model_error_is_actionable(tmp_path):
    (tmp_path / "models--other--thing").mkdir()
    with pytest.raises(FileNotFoundError, match="modelReferences"):
        st.acquire_modelstore(REPO, hub_root=str(tmp_path))


def test_modelstore_reports_shared_storage_files(tmp_path):
    make_hf_cache(tmp_path)
    acq = st.acquire_modelstore(REPO, hub_root=str(tmp_path))
    assert not acq.hub_online and len(acq.shared_storage_files) == 2 and acq.bytes_total == 4098


def test_volume_downloads_once_then_resolves_offline(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo, cache_dir, token):
        calls.append(cache_dir)
        return make_hf_cache(Path(cache_dir))

    monkeypatch.setattr(st, "_download", fake_download)
    first = st.acquire_volume(REPO, hub_root=str(tmp_path))
    second = st.acquire_volume(REPO, hub_root=str(tmp_path))
    assert first.hub_online and "download_s" in first.timings
    assert not second.hub_online and "resolve_s" in second.timings
    assert len(calls) == 1


def test_nocache_always_downloads_and_wipes_leftovers(tmp_path, monkeypatch):
    stale = tmp_path / "nocache-old"
    stale.mkdir()
    (stale / "blob").write_bytes(b"y" * 100)
    monkeypatch.setattr(st, "_download", lambda repo, cache_dir, token: make_hf_cache(Path(cache_dir)))
    acq = st.acquire_nocache(REPO, local_root=str(tmp_path))
    assert acq.hub_online and acq.notes["leftover_bytes"] == 100
    assert not stale.exists()


class FakeVolumeCache:
    """Mimics VolumeCache's contract: hydrate restores blobs but skips symlinks/refs."""
    mirror: dict[str, bytes] = {}

    def __init__(self, dirs, namespace, volume_path, best_effort):
        self.root = Path(dirs[0])

    def hydrate(self):
        for rel, data in self.mirror.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return len(self.mirror)

    def sync(self, background=True):
        FakeVolumeCache.mirror = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*")
                                  if p.is_file() and not p.is_symlink() and "/refs/" not in str(p)}


def test_volumecache_populates_then_hydrates_and_relinks(tmp_path, monkeypatch):
    FakeVolumeCache.mirror = {}
    WEIGHTS = 2 * st.POPULATION_MIN_BYTES

    def fake_download(repo, cache_dir, token):
        hub = Path(cache_dir)
        base = hub / st.repo_dir_name(repo)
        if (base / "blobs").exists():            # relink only: write refs + symlinks, no blob bytes
            snap = base / "snapshots" / COMMIT
            snap.mkdir(parents=True, exist_ok=True)
            for blob in (base / "blobs").iterdir():
                (snap / blob.name.removeprefix("blob-")).symlink_to(os.path.relpath(blob, snap))
            (base / "refs").mkdir(exist_ok=True)
            (base / "refs" / "main").write_text(COMMIT)
            return snap
        return make_hf_cache(hub, repo, weight_bytes=WEIGHTS)

    monkeypatch.setattr(st, "_download", fake_download)
    populate = st.acquire_volumecache(REPO, namespace="ns", local_root=str(tmp_path), cache_cls=FakeVolumeCache)
    assert populate.notes["synced"] and populate.notes["hub_fetched_bytes"] > 0
    cold = st.acquire_volumecache(REPO, namespace="ns", local_root=str(tmp_path), cache_cls=FakeVolumeCache)
    assert cold.notes["hydrated_bytes"] == WEIGHTS + 2
    assert cold.notes["hub_fetched_bytes"] < st.POPULATION_MIN_BYTES and not cold.notes["synced"]
    assert cold.hub_online and "hub_s" in cold.timings        # relink needed the Hub
    assert cold.notes["leftover_bytes"] > 0                   # previous local copy was wiped first


def test_volumecache_is_the_sdk_class():
    from runpod.serverless import VolumeCache

    cls, impl = st.load_volumecache_class()
    assert cls is VolumeCache and impl.startswith("runpod-2.")


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="posix_fadvise is Linux-only")
def test_evict_page_cache_counts_bytes_and_skips_missing(tmp_path):
    f = tmp_path / "a"
    f.write_bytes(b"z" * 10)
    assert st.evict_page_cache([str(f), str(tmp_path / "missing")]) == 10


def test_globalstore_populates_plain_files_once_then_resolves(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo, cache_dir, token):
        calls.append(cache_dir)
        return make_hf_cache(Path(cache_dir), repo)

    monkeypatch.setattr(st, "_download", fake_download)
    models = tmp_path / "gs"
    first = st.acquire_globalstore(REPO, models_root=str(models), stage_root=str(tmp_path / "local"))
    target = Path(first.path)
    assert first.notes["populated_now"] and "upload_s" in first.timings
    assert (target / st.COMPLETE_MARKER).is_file()
    assert not any(p.is_symlink() for p in target.iterdir())          # real files, no HF symlinks
    assert (target / "model.safetensors").stat().st_size == 4096
    second = st.acquire_globalstore(REPO, models_root=str(models), stage_root=str(tmp_path / "local"))
    assert not second.hub_online and "resolve_s" in second.timings and len(calls) == 1
    assert st.COMPLETE_MARKER not in [Path(p).name for p in second.shared_storage_files]


def test_globalstore_without_marker_is_treated_as_absent(tmp_path, monkeypatch):
    models = tmp_path / "gs"
    partial = models / REPO.replace("/", "--")
    partial.mkdir(parents=True)
    (partial / "model.safetensors").write_bytes(b"half")               # copy interrupted before the marker
    monkeypatch.setattr(st, "_download", lambda repo, cache_dir, token: make_hf_cache(Path(cache_dir), repo))
    acq = st.acquire_globalstore(REPO, models_root=str(models), stage_root=str(tmp_path / "local"))
    assert acq.notes["populated_now"] and (partial / "model.safetensors").stat().st_size == 4096


def test_sdk_volumecache_copies_when_metadata_is_rejected(tmp_path, monkeypatch):
    """Why gsvolumecache needs no patch any more: Global Store rejects chmod/utime with
    EPERM, and runpod>=2.1.0 treats metadata as best-effort (1.12 dropped the file)."""
    from runpod.serverless.utils import rp_volume_cache

    def eperm(*a, **k):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(rp_volume_cache.os, "utime", eperm)
    monkeypatch.setattr(rp_volume_cache.shutil, "copymode", eperm)
    cls, _ = st.load_volumecache_class()
    src = tmp_path / "src"
    src.write_bytes(b"w" * 1000)
    vc = cls.__new__(cls)
    assert vc._copy_file(str(src), str(tmp_path / "mirror" / "big")) is True
    assert (tmp_path / "mirror" / "big").read_bytes() == b"w" * 1000


# ------------------------------------------------------------- gshydrate
def _tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_parallel_hydrate_copies_tree_byte_identically_across_chunk_boundaries(tmp_path):
    files = {"model-1.safetensors": os.urandom(10_000), "sub/tok.json": b"{}", "empty.txt": b"",
             "exact.bin": os.urandom(4096)}
    _tree(tmp_path / "src", files)
    stats = st.parallel_hydrate(tmp_path / "src", tmp_path / "dst", workers=4, chunk_bytes=4096)
    for rel, data in files.items():
        assert (tmp_path / "dst" / rel).read_bytes() == data
    assert stats["bytes"] == sum(map(len, files.values()))
    assert stats["files"] == len(files) and stats["retries"] == 0
    assert stats["ranges"] == 3 + 1 + 0 + 1        # 10_000 B -> 3 chunks, empty file -> none


def test_parallel_hydrate_skips_completion_marker(tmp_path):
    _tree(tmp_path / "src", {"a.bin": b"a", st.COMPLETE_MARKER: b"repo"})
    st.parallel_hydrate(tmp_path / "src", tmp_path / "dst", workers=2, chunk_bytes=1024)
    assert not (tmp_path / "dst" / st.COMPLETE_MARKER).exists()


def test_parallel_hydrate_retries_transient_fuse_errors(tmp_path, monkeypatch):
    data = os.urandom(9000)
    _tree(tmp_path / "src", {"w.safetensors": data})
    real_pread, failures = os.pread, {"left": 3}

    def flaky_pread(fd, n, off):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError(12, "Cannot allocate memory")   # what geesefs returns when its buffer pool is full
        return real_pread(fd, n, off)

    monkeypatch.setattr(st.os, "pread", flaky_pread)
    monkeypatch.setattr(st, "HYDRATE_RETRY_BASE_S", 0.0)
    stats = st.parallel_hydrate(tmp_path / "src", tmp_path / "dst", workers=2, chunk_bytes=4096)
    assert (tmp_path / "dst" / "w.safetensors").read_bytes() == data
    assert stats["retries"] == 3


def test_parallel_hydrate_raises_permanent_errors_with_the_path(tmp_path, monkeypatch):
    _tree(tmp_path / "src", {"w.safetensors": b"x" * 100})

    def denied(fd, n, off):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(st.os, "pread", denied)
    with pytest.raises(OSError, match="w.safetensors"):
        st.parallel_hydrate(tmp_path / "src", tmp_path / "dst", workers=2, chunk_bytes=64)


def test_gshydrate_populates_once_then_hydrates_locally_without_the_hub(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo, cache_dir, token):
        calls.append(cache_dir)
        return make_hf_cache(Path(cache_dir), repo)

    monkeypatch.setattr(st, "_download", fake_download)
    kw = dict(models_root=str(tmp_path / "gs"), local_root=str(tmp_path / "local"), workers=4, chunk_bytes=1024)
    first = st.acquire_gshydrate(REPO, **kw)
    assert first.hub_online and first.notes["populated_now"]
    (Path(first.path) / "leftover.bin").write_bytes(b"old")     # FlashBoot restart keeps container disk
    second = st.acquire_gshydrate(REPO, **kw)
    local = Path(second.path)
    assert not second.hub_online and len(calls) == 1
    assert str(local).startswith(str(tmp_path / "local"))       # loads from container disk, not the mount
    assert (local / "model.safetensors").read_bytes() == b"x" * 4096
    assert not (local / "leftover.bin").exists()
    assert second.notes["leftover_bytes"] == first.bytes_total + 3   # previous hydrate + stray file
    assert "hydrate_s" in second.timings and second.notes["hydrate"]["bytes"] == 4096 + 2
    assert second.shared_storage_files                          # benchmark evicts these before reading


# ---------------------------------------------------------- volumeprefetch
def test_prefetch_files_reads_every_byte_across_chunk_boundaries(tmp_path, monkeypatch):
    sizes = {"a.safetensors": 10_000, "b.safetensors": 4096, "empty.json": 0}
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(os.urandom(size))
    read: list[int] = []
    real_pread = os.pread

    def counting_pread(fd, n, off):
        buf = real_pread(fd, n, off)
        read.append(len(buf))
        return buf

    monkeypatch.setattr(st.os, "pread", counting_pread)
    stats = st.prefetch_files([str(tmp_path / n) for n in sizes], workers=3, chunk_bytes=4096)
    assert stats["bytes"] == sum(sizes.values()) == sum(read)
    assert stats["files"] == 3 and stats["ranges"] == 3 + 1   # 10_000 -> 3 chunks, 4096 -> 1, empty -> 0
    assert (stats["workers"], stats["chunk_bytes"]) == (3, 4096)


def test_prefetch_files_reports_short_reads_with_the_path(tmp_path, monkeypatch):
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"x" * 100)
    monkeypatch.setattr(st.os, "pread", lambda fd, n, off: b"")
    with pytest.raises(OSError, match="w.safetensors @ 0"):
        st.prefetch_files([str(f)], workers=1, chunk_bytes=64)


def test_volumeprefetch_loads_in_place_from_the_volume_cache_after_prefetching(tmp_path):
    snap = make_hf_cache(tmp_path, weight_bytes=9000)
    acq = st.acquire_volumeprefetch(REPO, hub_root=str(tmp_path), workers=2, chunk_bytes=4096)
    assert acq.path == str(snap)                     # model is loaded straight off the volume
    assert acq.hub_online is False and "prefetch_s" in acq.timings
    assert acq.notes["prefetch"]["bytes"] == acq.bytes_total == 9000 + 2
    # The load must hit the pages just prefetched: nothing to evict between acquire and load.
    assert acq.shared_storage_files == []
    assert sorted(acq.notes["prefetched_files"]) == sorted(str(p) for p in st.snapshot_files(snap))


def test_runtime_evicts_volumeprefetch_files_before_prefetching_not_after(tmp_path, monkeypatch):
    import runtime

    make_hf_cache(tmp_path)
    calls: list[str] = []
    real_prefetch = st.prefetch_files
    monkeypatch.setattr(st, "VOLUME_HF_CACHE", str(tmp_path))
    monkeypatch.setattr(st, "evict_page_cache", lambda paths: calls.append("evict") or 7)
    monkeypatch.setattr(st, "prefetch_files", lambda *a, **k: calls.append("prefetch") or real_prefetch(*a, **k))
    acq = runtime._acquire("volumeprefetch", REPO, None, evict=True)
    assert calls == ["evict", "prefetch"]
    assert acq.notes["evicted_volume_bytes"] == 7 and acq.shared_storage_files == []


def test_cgroup_memory_limit_reads_v2_and_reports_unlimited_or_missing(tmp_path):
    f = tmp_path / "memory.max"
    f.write_text("68719476736\n")
    assert st.cgroup_memory_limit(str(f)) == 68719476736
    f.write_text("max\n")
    assert st.cgroup_memory_limit(str(f)) == "max"
    assert st.cgroup_memory_limit(str(tmp_path / "absent")) is None
