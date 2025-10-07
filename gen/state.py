from __future__ import annotations

import asyncio
import base64
import os
import random
import tempfile
from io import BytesIO
from typing import Tuple, Optional

import httpx
from loguru import logger
import imageio
import numpy as np

from .settings import Config

# Trellis imports
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


class TrellisState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Suggested perf/env knobs (honor env first if user set them)
        os.environ.setdefault("SPCONV_ALGO", cfg.spconv_algo)
        if cfg.attn_backend:
            os.environ.setdefault("ATTN_BACKEND", cfg.attn_backend)

        logger.info(f"Loading Trellis pipeline: {cfg.model_id}")
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(cfg.model_id)
        if cfg.gpu_id >= 0:
            self.pipeline.cuda()

        self.validator_url: str = cfg.validator_url

    # ---------------------------
    # Public API for routes
    # ---------------------------

    async def generate_ply_buffer_validated(self, prompt: str) -> Tuple[bytes, float, int]:
        """
        Validate+retry loop under a soft budget (route enforces 30s hard).
        Returns (ply_bytes, best_score, attempts).
        """
        attempts = 0
        best_score = -1.0
        best_ply: Optional[bytes] = None

        deadline = asyncio.get_event_loop().time() + self.cfg.inner_deadline_seconds

        for attempt in range(1, self.cfg.max_attempts + 1):
            attempts = attempt
            # Stop if our soft budget is spent; route has a hard 30s anyway.
            if asyncio.get_event_loop().time() >= deadline:
                logger.info("[state] inner deadline reached; stopping retries")
                break

            seed = self._seed_for_attempt(attempt)
            logger.info(f"[gen] attempt={attempt}/{self.cfg.max_attempts}, seed={seed}")

            # Run Trellis on threadpool to avoid blocking the loop.
            outputs = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.pipeline.run(
                    prompt,
                    seed=seed,
                    sparse_structure_sampler_params=self.cfg.sparse_sampler_params_dict(),
                    slat_sampler_params=self.cfg.slat_sampler_params_dict(),
                ),
            )

            # Extract Gaussian to PLY
            ply_bytes = await self._export_gaussian_ply_to_bytes(outputs)
            score, passed, raw = await self._call_external_validator(prompt, ply_bytes)
            logger.info(f"[validator] score={score:.3f}, passed={passed}")

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if passed:
                break

        if best_ply is None:
            # Fallback empty bytes if truly nothing
            best_ply = b""

        return best_ply, float(best_score if best_score >= 0 else 0.0), attempts

    async def generate_orbit_mp4_validated(self, prompt: str, res: int = 1088) -> Tuple[BytesIO, float, int]:
        """
        Generate once or with retries until passing validation (same loop as PLY),
        then render an orbit video (gaussian normal/color). Returns (mp4_buf, score, attempts).
        """
        # Reuse the PLY generator (which already validates) but keep outputs for video.
        attempts = 0
        best_score = -1.0
        best_outputs = None

        deadline = asyncio.get_event_loop().time() + self.cfg.inner_deadline_seconds

        for attempt in range(1, self.cfg.max_attempts + 1):
            attempts = attempt
            if asyncio.get_event_loop().time() >= deadline:
                logger.info("[state] inner deadline reached (video); stopping retries")
                break

            seed = self._seed_for_attempt(attempt)
            logger.info(f"[gen-video] attempt={attempt}/{self.cfg.max_attempts}, seed={seed}")

            outputs = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.pipeline.run(
                    prompt,
                    seed=seed,
                    sparse_structure_sampler_params=self.cfg.sparse_sampler_params_dict(),
                    slat_sampler_params=self.cfg.slat_sampler_params_dict(),
                ),
            )

            # Validate using gaussian->PLY
            ply_bytes = await self._export_gaussian_ply_to_bytes(outputs)
            score, passed, _ = await self._call_external_validator(prompt, ply_bytes)
            logger.info(f"[validator-video] score={score:.3f}, passed={passed}")

            if score > best_score:
                best_score, best_outputs = score, outputs
            if passed:
                best_outputs = outputs
                break

        if best_outputs is None:
            # Nothing generated; return empty mp4 buffer
            return BytesIO(b""), float(best_score if best_score >= 0 else 0.0), attempts

        # Render a short orbit from gaussian (color). Keep it fast.
        mp4_buf = await self._render_gaussian_orbit_mp4(best_outputs, res=res)
        return mp4_buf, float(best_score if best_score >= 0 else 0.0), attempts

    # ---------------------------
    # Helpers
    # ---------------------------

    def _seed_for_attempt(self, attempt: int) -> int:
        if self.cfg.seed >= 0:
            # Deterministic but varied across attempts
            return (self.cfg.seed + 31 * (attempt - 1)) & 0x7FFFFFFF
        # Randomized
        return random.randint(0, 2**31 - 1)

    async def _export_gaussian_ply_to_bytes(self, outputs: dict) -> bytes:
        """
        Save gaussian to PLY in temp file and return bytes.
        """
        if not outputs.get("gaussian"):
            return b""
        gaussian = outputs["gaussian"][0]
        with tempfile.TemporaryDirectory() as td:
            ply_path = os.path.join(td, "model.ply")
            # Trellis exposes .save_ply
            def _save():
                gaussian.save_ply(ply_path)
            await asyncio.get_running_loop().run_in_executor(None, _save)
            with open(ply_path, "rb") as f:
                return f.read()

    async def _render_gaussian_orbit_mp4(self, outputs: dict, res: int = 1088) -> BytesIO:
        """
        Renders gaussian color frames and encodes to mp4 in a temp file for robustness.
        """
        gaussian = outputs["gaussian"][0]
        # render_utils.render_video returns a dict with 'color'
        def _render():
            vid = render_utils.render_video(gaussian, resolution=res)
            # Expect vid['color'] to be (T, H, W, 3) uint8
            frames = vid.get("color")
            if frames is None:
                # Fallback to mesh normal if available
                mesh = outputs.get("mesh", [None])[0]
                if mesh is not None:
                    vid2 = render_utils.render_video(mesh)
                    frames2 = vid2.get("normal")
                    return frames2
            return frames

        frames = await asyncio.get_running_loop().run_in_executor(None, _render)
        if frames is None or len(frames) == 0:
            return BytesIO(b"")

        with tempfile.TemporaryDirectory() as td:
            mp4_path = os.path.join(td, "orbit.mp4")
            # Use imageio to write MP4 quickly
            def _encode():
                # Ensure uint8 numpy array
                arr = np.asarray(frames)
                imageio.mimsave(mp4_path, arr, fps=self.cfg.video_fps)
            await asyncio.get_running_loop().run_in_executor(None, _encode)
            with open(mp4_path, "rb") as f:
                return BytesIO(f.read())

    async def _call_external_validator(self, prompt: str, ply_bytes: bytes) -> Tuple[float, bool, dict]:
        """
        Sends the base64 PLY to the external validator and returns (score, passed, raw_json).
        Uses the `score` field of ValidationResponse directly.
        """
        payload = {
            "prompt": prompt,
            "prompt_image": None,
            "data": base64.b64encode(ply_bytes).decode("utf-8"),
            "compression": 0,
            "generate_single_preview": False,
            "generate_grid_preview": False,
            "preview_score_threshold": float(self.cfg.vld_threshold),
        }

        try:
            timeout = httpx.Timeout(
                connect=self.cfg.vld_connect_timeout_s,
                read=self.cfg.vld_read_timeout_s,
                write=self.cfg.vld_write_timeout_s,
                pool=self.cfg.vld_pool_timeout_s,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.validator_url, json=payload)
                resp.raise_for_status()
                js = resp.json()
        except Exception as e:
            logger.warning(f"[validator] error: {e}")
            return 0.0, False, {"error": str(e)}

        # Strictly use the `score` field as provided by ValidationResponse
        try:
            score = float(js.get("score", 0.0))
        except Exception:
            score = 0.0
        passed = score >= float(self.cfg.vld_threshold)
        return score, passed, js
