from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_env_path(*names: str) -> Path | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return Path(value).expanduser()
    return None


def _first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def benchmark_output_root() -> Path:
    return (
        _first_env_path("SVLLM_BENCHMARK_OUTPUT_DIR", "DELTAKV_OUTPUT_DIR", "DELTAKV_OUTPUT_BASE")
        or REPO_ROOT / "benchmark" / "results"
    )


def benchmark_data_root() -> Path | None:
    return _first_env_path("SVLLM_BENCHMARK_DATA_DIR", "SVLLM_DATA_DIR", "DELTAKV_DATA_DIR")


def longbench_data_root() -> Path | None:
    env_path = _first_env_path(
        "SVLLM_LONGBENCH_DATA_DIR",
        "DELTAKV_LONGBENCH_DATA_DIR",
        "SVLLM_BENCHMARK_DATA_DIR",
        "SVLLM_DATA_DIR",
        "DELTAKV_DATA_DIR",
    )
    if env_path is not None:
        return env_path
    return _first_existing_path(
        REPO_ROOT / "benchmark" / "data" / "LongBench",
        REPO_ROOT / "benchmark" / "data" / "longbench",
    )


def scbench_preprocessed_root() -> Path | None:
    env_path = _first_env_path(
        "SVLLM_SCBENCH_PREPROCESSED_ROOT",
        "SCBENCH_PREPROCESSED_ROOT",
        "SVLLM_BENCHMARK_DATA_DIR",
        "SVLLM_DATA_DIR",
        "DELTAKV_DATA_DIR",
    )
    if env_path is not None:
        return env_path
    return _first_existing_path(
        REPO_ROOT / "benchmark" / "data" / "SCBench-preprocessed",
        REPO_ROOT / "benchmark" / "data" / "scbench-preprocessed",
    )


def default_output_path(*parts: str) -> str:
    return str(benchmark_output_root().joinpath(*parts))


def require_existing_dir(path: str | Path | None, label: str, env_hint: str | None = None) -> Path:
    if path is None or str(path).strip() == "":
        hint = f" Set {env_hint} or pass the corresponding CLI argument." if env_hint else ""
        raise FileNotFoundError(f"Missing required {label}.{hint}")
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def require_existing_file(path: str | Path | None, label: str, env_hint: str | None = None) -> Path:
    if path is None or str(path).strip() == "":
        hint = f" Set {env_hint} or pass the corresponding CLI argument." if env_hint else ""
        raise FileNotFoundError(f"Missing required {label}.{hint}")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return resolved
