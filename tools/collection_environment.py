"""Best-effort provenance for a baseline run; probes never require extra packages."""

from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def _command(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                                timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _version(name):
    try:
        return metadata.version(name)
    except Exception:
        return None


def collect_environment(torch, run):
    """Record the selected CUDA device, separately from the configured reference chip.

    Missing tools/optional dependencies are represented by null. Only explicitly
    selected environment variables are recorded, never the full process environment.
    """
    root = Path(__file__).resolve().parent.parent
    gpu = {"name": None, "uuid": None, "total_memory_bytes": None,
           "compute_capability": None, "multiprocessor_count": None,
           "cuda_device_index": None, "driver_version": None}
    try:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu.update(name=props.name, total_memory_bytes=props.total_memory,
                   compute_capability=f"{props.major}.{props.minor}",
                   multiprocessor_count=props.multi_processor_count,
                   cuda_device_index=index)
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            gpu["uuid"] = str(uuid)
    except Exception:
        pass
    # UUID selection respects CUDA_VISIBLE_DEVICES remapping. Never use a CUDA
    # logical ordinal as an nvidia-smi physical index (including on MIG systems).
    if gpu["uuid"]:
        driver = _command(["nvidia-smi", "--id", gpu["uuid"],
                           "--query-gpu=driver_version", "--format=csv,noheader"])
        gpu["driver_version"] = driver or None
    status = _command(["git", "status", "--porcelain", "--untracked-files=normal"], root)
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": gpu, "cpu_model": platform.processor() or None,
                     "logical_cpu_count": os.cpu_count()},
        "software": {
            "os": platform.system(), "os_release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "packages": {name: _version(name) for name in
                         ("torch", "triton", "vllm", "sglang")},
            "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "nvcc_version": _command(["nvcc", "--version"]),
            "ncu_version": _command(["ncu", "--version"]),
        },
        "source": {
            "git_commit": _command(["git", "rev-parse", "HEAD"], root),
            "git_branch": _command(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
            "git_dirty": None if status is None else bool(status),
        },
        "run": json.loads(json.dumps(run)),
        "environment_variables": {name: os.environ.get(name) for name in
                                  ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
    }
