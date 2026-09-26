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


def test_vendored_volumecache_imports():
    cls, impl = st.load_volumecache_class()
    assert hasattr(cls, "hydrate") and impl in ("runpod", "vendored-1.12.0")


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


def test_global_store_volumecache_copies_when_metadata_is_rejected(tmp_path, monkeypatch):
    def eperm(*a, **k):
        raise PermissionError(1, "Operation not permitted")

    cls, impl = st.global_store_volumecache_class()
    assert impl.endswith("+globalstore-copyfile")
    src = tmp_path / "src"
    src.write_bytes(b"w" * 1000)
    monkeypatch.setattr(st.shutil, "copystat", eperm)                 # what Global Store does
    vc = cls.__new__(cls)
    assert vc._copy_file(str(src), str(tmp_path / "mirror" / "big")) is True
    assert (tmp_path / "mirror" / "big").read_bytes() == b"w" * 1000
    base, _ = st.load_volumecache_class()
    assert base._copy_file(vc, str(src), str(tmp_path / "mirror" / "stock")) is False   # the stock bug


def test_global_store_volumecache_records_copy_failures(tmp_path):
    cls, _ = st.global_store_volumecache_class()
    vc = cls.__new__(cls)
    assert vc._copy_file(str(tmp_path / "missing"), str(tmp_path / "out")) is False
    assert len(vc.copy_errors) == 1 and "FileNotFoundError" in vc.copy_errors[0]
