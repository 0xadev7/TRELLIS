from __future__ import annotations

import asyncio
from time import time

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import torch
from loguru import logger

from settings import Config, get_config
from state import TrellisState

app = FastAPI()
STATE: TrellisState | None = None
CFG: Config | None = None


def get_config_dep() -> Config:
    # Expose config via dependency for future per-route overrides if needed.
    assert CFG is not None
    return CFG


@app.on_event("startup")
def startup_event() -> None:
    global STATE, CFG
    CFG = get_config()
    if CFG.gpu_id >= 0:
        torch.cuda.set_device(CFG.gpu_id)
    STATE = TrellisState(CFG)
    logger.info(
        f"Server up. Port={CFG.port}, GPU={CFG.gpu_id}, "
        f"Validator={CFG.validator_url}, vld_threshold={CFG.vld_threshold}"
    )


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    _cfg: Config = Depends(get_config_dep),
) -> Response:
    """
    Generate a PLY and validate it via the external validator.
    MUST return within 30 seconds. If over time, returns empty bytes.
    """
    assert STATE is not None
    t0 = time()
    try:
        ply_buf, gen_score, attempts = await asyncio.wait_for(
            STATE.generate_ply_buffer_validated(prompt.strip()),
            timeout=30.0,
        )
        elapsed = time() - t0
        logger.info(
            f"[/generate] score={gen_score:.3f}, attempts={attempts}, total={elapsed:.2f}s"
        )
        return Response(ply_buf, media_type="application/octet-stream")
    except asyncio.TimeoutError:
        logger.warning("[/generate] timed out at 30s; returning empty bytes")
        return Response(b"", media_type="application/octet-stream")


@app.post("/generate_video")
async def generate_video(
    prompt: str = Form(...),
    video_res: int = Form(1088),
    _cfg: Config = Depends(get_config_dep),
) -> StreamingResponse:
    """
    Generate and validate a sample, then return an orbit render (mp4).
    This route ALSO respects the overall 30s budget.
    """
    assert STATE is not None
    t0 = time()
    try:
        mp4_buf, gen_score, attempts = await asyncio.wait_for(
            STATE.generate_orbit_mp4_validated(prompt.strip(), res=video_res),
            timeout=30.0,
        )
        elapsed = time() - t0
        logger.info(
            f"[/generate_video] score={gen_score:.3f}, attempts={attempts}, total={elapsed:.2f}s"
        )
        return StreamingResponse(content=mp4_buf, media_type="video/mp4")
    except asyncio.TimeoutError:
        logger.warning("[/generate_video] timed out at 30s; returning empty bytes")
        return StreamingResponse(content=b"", media_type="video/mp4")


if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(app, host="0.0.0.0", port=cfg.port)
