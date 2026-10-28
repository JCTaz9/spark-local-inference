"""Prepared mHC packages replay all outputs without the Python exporter."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def native_reader():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        "cargo", "build", "--locked", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools",
        "--bin", "test_fixed_runtime_package",
    ], cwd=root, check=True, timeout=600)
    return root / "helios/target/debug/test_fixed_runtime_package"


@pytest.mark.parametrize("finalizer", ("fused", "separate"))
@pytest.mark.parametrize("precision", ("bf16", "fp32"))
def test_mhc_replays_without_python(tmp_path, native_reader, finalizer, precision):
    import torch

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a GPU before running this test"
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("composed mHC requires SM103")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path / f"{finalizer}-{precision}"
    exported = subprocess.run([
        sys.executable, str(root / "helios/tests/mhc_package_fixture.py"),
        str(directory), finalizer, precision,
    ], cwd=root, capture_output=True, text=True, timeout=600, check=False)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert f"exported mHC {finalizer} {precision}" in exported.stdout
    # This process cannot borrow exporter-side Python, Torch, modules or tensors.
    replayed = subprocess.run([str(native_reader), str(directory)], cwd=root,
                              capture_output=True, text=True, timeout=120, check=False)
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert "native fixed-address replay passed" in replayed.stdout
