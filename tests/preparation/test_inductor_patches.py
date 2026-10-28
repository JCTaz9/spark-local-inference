"""Installation-file patching, dependency checks, and automatic preparation startup."""

import os
import re
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from types import SimpleNamespace

import pytest

from ignition import patches
from ignition.cli import environment
from ignition.cli.config import Configuration
from ignition.errors import PreparationError
from ignition.patches.__main__ import main


@pytest.fixture
def installation(monkeypatch, tmp_path):
    """Install distribution metadata and the exact source contexts from both diffs."""
    for name in ("triton_heuristics", "static_triton_launcher"):
        monkeypatch.delitem(sys.modules, f"torch._inductor.runtime.{name}", raising=False)
    sources, expected, diffs = {}, {}, []
    for name in ("torch-inductor-triton-cache-filenames.patch", "torch-inductor-cooperative-static-launch.patch"):
        diff = files(patches).joinpath(name).read_text(encoding="utf-8")
        diffs.append(diff)
        relative = None
        for line in diff.splitlines(keepends=True):
            if line.startswith("--- a/"):
                relative = line.removeprefix("--- a/").strip()
            elif line.startswith("@@ "):
                position = int(re.match(r"@@ -(\d+)", line)[1]) - 1
            elif relative is not None and line.startswith((" ", "-")):
                sources.setdefault(relative, {})[position] = line[1:]
                position += 1
    root = tmp_path / "site-packages"
    for relative, lines in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines.get(i, f"# source line {i}\n") for i in range(max(lines) + 1)))
        sources[relative] = path.read_bytes()
    for name, version in (("torch", "2.14.1+cu132"), ("triton", "3.8.0")):
        metadata = root / f"{name}-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    monkeypatch.syspath_prepend(str(root))
    for relative, before, after in patches._hunks("".join(diffs)):
        expected[relative] = expected.get(relative, sources[relative].decode()).replace(before, after, 1)
    return SimpleNamespace(root=root, sources=sources, expected=expected, diffs=diffs)


def assert_patched(installation):
    for relative, expected in installation.expected.items():
        assert (installation.root / relative).read_text() == expected


def test_preparation_startup_applies_patches_without_importing_torch(installation, tmp_path, monkeypatch):
    for module in ("torch", "triton"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nkernel_cache = "{tmp_path}/kernels"\n')
    environment.apply(Configuration.load(config), ())
    assert_patched(installation)
    assert "torch" not in sys.modules and "triton" not in sys.modules


def test_repeated_startup_keeps_files_and_reinstall_reapplies_patches(installation):
    assert len(patches.apply()) == 2
    installed = {relative: (installation.root / relative).stat() for relative in installation.sources}
    assert patches.apply() == ()
    for relative, before in installed.items():
        after = (installation.root / relative).stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    for relative, source in installation.sources.items():
        (installation.root / relative).write_bytes(source)
    assert len(patches.apply()) == 2
    assert_patched(installation)


@pytest.mark.parametrize("patch_index", [0, 1])
@pytest.mark.parametrize("partial", [False, True])
def test_existing_manual_patch_is_accepted(installation, patch_index, partial):
    for relative, before, after in patches._hunks(installation.diffs[patch_index]):
        path = installation.root / relative
        path.write_text(path.read_text().replace(before, after, 1))
        if partial:
            break
    patches.apply()
    assert_patched(installation)


def test_file_replacement_preserves_permissions_and_wheel_cache_hardlinks(installation, tmp_path):
    links = tmp_path / "wheel-cache"
    links.mkdir()
    for relative in installation.sources:
        target = installation.root / relative
        target.chmod(0o644)
        os.link(target, links / target.name)
    patches.apply()
    assert_patched(installation)
    for relative, source in installation.sources.items():
        target = installation.root / relative
        assert (links / target.name).read_bytes() == source
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        assert target.stat().st_ino != (links / target.name).stat().st_ino


@pytest.mark.parametrize("name, version", [("torch", "2.14.0+cu132"), ("triton", "3.7.0")])
def test_incompatible_dependency_versions_fail_before_writes(installation, name, version):
    metadata = next(installation.root.glob(f"{name}-*.dist-info/METADATA"))
    metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    with pytest.raises(PreparationError, match=f"require {name}=="):
        patches.apply()
    for relative, source in installation.sources.items():
        assert (installation.root / relative).read_bytes() == source


@pytest.mark.parametrize("ambiguous", [False, True])
def test_incompatible_context_fails_before_any_file_is_written(installation, ambiguous):
    relative = "torch/_inductor/runtime/static_triton_launcher.py"
    path = installation.root / relative
    source = path.read_text()
    path.write_text(source + source if ambiguous else source.replace("self.full_constexprs = constants", "pass"))
    before = {relative: (installation.root / relative).read_bytes() for relative in installation.sources}
    with pytest.raises(PreparationError, match="context is missing or ambiguous"):
        patches.apply()
    for relative, source in before.items():
        assert (installation.root / relative).read_bytes() == source


def test_imported_inductor_module_requires_process_restart(installation, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.static_triton_launcher", SimpleNamespace())
    with pytest.raises(PreparationError, match="restart Python"):
        patches.apply()
    for relative, source in installation.sources.items():
        assert (installation.root / relative).read_bytes() == source


def test_simultaneous_startup_produces_the_same_patched_files(installation):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: patches.apply(), range(8)))
    assert_patched(installation)
    assert patches.apply() == ()


def test_explicit_entry_point_reports_errors_and_repeated_application(installation, capsys):
    assert main() == 0
    assert capsys.readouterr().out.count("Patched ") == 2
    assert main() == 0
    assert capsys.readouterr().out == ""
    next(installation.root.glob("triton-*.dist-info/METADATA")).write_text("Name: triton\nVersion: 3.7.0\n")
    assert main() == PreparationError.exit_code
    assert "require triton==3.8.0" in capsys.readouterr().err


def test_bundled_patch_results_match_system_patch(installation, tmp_path):
    executable = shutil.which("patch")
    if executable is None:
        pytest.skip("system patch is unavailable for the independent comparison")
    reference = tmp_path / "reference"
    for relative, source in installation.sources.items():
        path = reference / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
    for diff in installation.diffs:
        subprocess.run([executable, "-d", str(reference), "-p1"], input=diff, text=True, check=True, capture_output=True)
    patches.apply()
    for relative in installation.sources:
        assert (installation.root / relative).read_bytes() == (reference / relative).read_bytes()
