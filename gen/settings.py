from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class Config:
    # Server
    port: int
    gpu_id: int  # -1 for CPU (not recommended), 0..N for CUDA device

    # Trellis
    model_id: str
    seed: int  # <0 means "random each attempt"
    attn_backend: str | None
    spconv_algo: str  # "native" or "auto"

    # Validate+retry
    validator_url: str
    vld_threshold: float
    max_attempts: int
    inner_deadline_seconds: float  # soft budget inside the 30s hard limit

    # Validator HTTP timeouts (seconds)
    vld_connect_timeout_s: float
    vld_read_timeout_s: float
    vld_write_timeout_s: float
    vld_pool_timeout_s: float

    # Sampler params (kept small to meet time budget)
    sparse_steps: int
    sparse_cfg: float
    slat_steps: int
    slat_cfg: float

    # Video
    video_fps: int

    # Helpers
    def sparse_sampler_params_dict(self) -> Dict[str, Any]:
        return {"steps": self.sparse_steps, "cfg_strength": self.sparse_cfg}

    def slat_sampler_params_dict(self) -> Dict[str, Any]:
        return {"steps": self.slat_steps, "cfg_strength": self.slat_cfg}


def _env_str(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _env_int(name: str, default: int | None = None) -> int:
    val = os.getenv(name, None if default is None else str(default))
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return int(val)


def _env_float(name: str, default: float | None = None) -> float:
    val = os.getenv(name, None if default is None else str(default))
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return float(val)


def _env_opt_str(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def get_config() -> Config:
    """
    Read from environment (.env supported) and return an immutable Config.
    """
    return Config(
        # Server
        port=_env_int("PORT", 8000),
        gpu_id=_env_int("GPU_ID", 0),

        # Trellis
        model_id=_env_str("MODEL_ID", "microsoft/TRELLIS-text-xlarge"),
        seed=_env_int("SEED", 1),
        attn_backend=_env_opt_str("ATTN_BACKEND", None),  # e.g., "xformers" or "flash-attn"
        spconv_algo=_env_str("SPCONV_ALGO", "native"),    # "native" avoids warmup benchmarking

        # Validate+retry
        validator_url=_env_str("VALIDATOR_URL", "http://localhost:8094/validate_txt_to_3d_ply/"),
        vld_threshold=_env_float("VALIDATION_THRESHOLD", 0.65),
        max_attempts=_env_int("MAX_ATTEMPTS", 3),
        inner_deadline_seconds=_env_float("INNER_DEADLINE_SECONDS", 24.0),

        # Validator HTTP timeouts
        vld_connect_timeout_s=_env_float("VLD_CONNECT_TIMEOUT_S", 3.0),
        vld_read_timeout_s=_env_float("VLD_READ_TIMEOUT_S", 6.0),
        vld_write_timeout_s=_env_float("VLD_WRITE_TIMEOUT_S", 3.0),
        vld_pool_timeout_s=_env_float("VLD_POOL_TIMEOUT_S", 3.0),

        # Samplers
        sparse_steps=_env_int("SPARSE_STEPS", 12),
        sparse_cfg=_env_float("SPARSE_CFG", 7.5),
        slat_steps=_env_int("SLAT_STEPS", 12),
        slat_cfg=_env_float("SLAT_CFG", 7.5),

        # Video
        video_fps=_env_int("VIDEO_FPS", 30),
    )
