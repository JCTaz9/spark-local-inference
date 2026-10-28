"""Apply the pinned PyTorch Inductor fixes before importing preparation libraries."""

import os
import stat
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import files
from pathlib import Path

from ignition.errors import PreparationError

_VERSIONS = {"torch": "2.14.1+cu132", "triton": "3.8.0"}
_PATCHES = (
    "torch-inductor-triton-cache-filenames.patch",
    "torch-inductor-cooperative-static-launch.patch",
)


def _hunks(patch):
    """Read file names and exact replacement blocks from the bundled unified diffs."""
    path = None
    before = after = None
    for line in patch.splitlines(keepends=True):
        if line.startswith(("diff --git ", "@@ ")):
            if before is not None:
                yield path, "".join(before), "".join(after)
            before = after = None
            if line.startswith("diff --git "):
                _, _, source, target = line.split()
                if source[2:] != target[2:] or not target.startswith("b/torch/_inductor/"):
                    raise PreparationError("Inductor patches must modify existing torch/_inductor files")
                path = target[2:]
            else:
                before, after = [], []
        elif before is not None:
            if line.startswith((" ", "-")):
                before.append(line[1:])
            if line.startswith((" ", "+")):
                after.append(line[1:])
    if before is not None:
        yield path, "".join(before), "".join(after)


def apply() -> tuple[Path, ...]:
    """Validate both patches, then atomically replace each changed installation file.

    Matching uses complete hunk context without fuzz. Already applied
    hunks are accepted, including installations with only one patch applied.
    Replacing files breaks wheel-cache hardlinks instead of modifying shared bytes.
    """
    try:
        packages = {name: distribution(name) for name in _VERSIONS}
        for name, expected in _VERSIONS.items():
            if packages[name].version != expected:
                raise PreparationError(
                    f"Inductor patches require {name}=={expected}; found {packages[name].version}. "
                    "Install the locked preparation dependencies with uv sync --locked --extra prepare.")

        sources, patched = {}, {}
        for name in _PATCHES:
            for relative, before, after in _hunks(files(__package__).joinpath(name).read_text(encoding="utf-8")):
                path = Path(packages["torch"].locate_file(relative))
                if path not in sources:
                    sources[path] = path.read_text(encoding="utf-8")
                    patched[path] = sources[path]
                source = patched[path]
                counts = source.count(before), source.count(after)
                if counts == (1, 0):
                    patched[path] = source.replace(before, after, 1)
                elif counts != (0, 1):
                    raise PreparationError(
                        f"Cannot apply {name} to {path}: patch context is missing or ambiguous. "
                        "Reinstall Torch with uv sync --locked --extra prepare --reinstall-package torch.")

        changed = tuple(path for path in sources if patched[path] != sources[path])
        for path in changed:
            module = "torch._inductor.runtime." + path.stem
            if module in sys.modules:
                raise PreparationError(f"Cannot patch imported module {module}; restart Python before preparation.")
        for path in changed:
            # Keep replacement files beside their targets for atomic publication on one filesystem.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    output.write(patched[path].encode("utf-8"))
                    output.close()
                    temporary.chmod(stat.S_IMODE(path.stat().st_mode))
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return changed
    except PackageNotFoundError as error:
        raise PreparationError(
            "Inductor patches require the preparation dependencies; "
            "install them with uv sync --locked --extra prepare.") from error
    except OSError as error:
        raise PreparationError(f"Cannot apply Inductor patches in the preparation environment: {error}") from error
