"""GUI 탭들이 공유하는 얇은 헬퍼. 파이프라인 로직은 전부 pipeline/ 에 있고 여기서는 호출만 한다."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import yaml

from vr180 import paths
from vr180.pipeline.job import Job, _deep_update, stages
from vr180.util import imageio
from vr180.util.log import Cancelled, RunContext

STATUS_ICON = {"done": "✅", "stale": "⚠️", "missing": "⬜"}
_running: dict[str, RunContext] = {}


# ----------------------------------------------------------------------------- Job
def job_choices() -> list[str]:
    return Job.list_jobs()


def get_job(job_id: str | None) -> Job | None:
    if not job_id:
        return None
    try:
        return Job.open(job_id)
    except FileNotFoundError:
        return None


def status_md(job_id: str | None) -> str:
    job = get_job(job_id)
    if job is None:
        return "Job 없음 — 위에서 입력 이미지를 올리고 **새 Job** 을 누르세요."
    cells = []
    for i, st in enumerate(stages()):
        s = job.status(i)
        cells.append(f"{STATUS_ICON[s]} {i} {st.NAME}")
    return (
        f"**Job `{job.id}`** &nbsp; "
        + " · ".join(cells)
        + "  \n(✅ 최신 · ⚠️ stale: 입력/파라미터가 바뀜 · ⬜ 없음)"
    )


def create_job(file_path: str | None, job_id: str | None) -> tuple[str, Any, str]:
    if not file_path:
        raise gr.Error("입력 SBS 이미지를 먼저 올리세요")
    job = Job.create(file_path, job_id=(job_id or None))
    return job.id, gr.update(choices=job_choices(), value=job.id), status_md(job.id)


# ----------------------------------------------------------------------------- 파라미터
def get_param(job_id: str | None, stage_idx: int, key: str, default: Any = None) -> Any:
    job = get_job(job_id)
    if job is None:
        return default
    v = job.stage_params(stage_idx).get(key, default)
    return default if v is None else v


def set_params(job: Job, stage_idx: int, values: dict[str, Any]) -> None:
    for k, v in values.items():
        job.set_param(stage_idx, k, v)
    job.save_params()


def set_model_param(job: Job, key: str, value: Any) -> None:
    job.model_params()[key] = value
    job.save_params()


# ----------------------------------------------------------------------------- 실행 (제너레이터: 로그를 흘려보낸다)
def run_pipeline(
    job_id: str | None,
    from_stage: int,
    to_stage: int | None,
    param_updates: dict[int, dict[str, Any]] | None = None,
    model_updates: dict | None = None,
    force_from: bool = True,
):
    """yield (log_text, status_md). GUI 버튼 핸들러에서 `yield from` 으로 쓴다."""
    job = get_job(job_id)
    if job is None:
        raise gr.Error("Job 을 먼저 만들거나 여세요")
    for si, vals in (param_updates or {}).items():
        for k, v in vals.items():
            job.set_param(si, k, v)
    if model_updates:
        job.model_params().update(model_updates)
    job.save_params()
    if job.id in _running:
        raise gr.Error("이 Job 은 이미 실행 중입니다 (취소 후 다시)")

    ctx = RunContext(log_fn=lambda s: None)
    _running[job.id] = ctx
    err: list[BaseException] = []

    def worker():
        try:
            job.run(from_stage=from_stage, to_stage=to_stage, ctx=ctx, force_from=force_from)
        except Cancelled:
            ctx.log("취소됨")
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            ctx.log(f"오류: {type(e).__name__}: {e}")

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    last = 0
    try:
        while th.is_alive():
            if len(ctx.lines) != last:
                last = len(ctx.lines)
                yield "\n".join(ctx.lines[-200:]), status_md(job.id)
            time.sleep(0.3)
        th.join()
    finally:
        _running.pop(job.id, None)
    yield "\n".join(ctx.lines[-200:]), status_md(job.id)
    if err:
        raise gr.Error(f"{type(err[0]).__name__}: {err[0]}")


def cancel_run(job_id: str | None) -> str:
    ctx = _running.get(job_id or "")
    if ctx is None:
        return "실행 중인 작업 없음"
    ctx.cancel()
    return "취소 요청함 — 현재 타일/단계가 끝나면 멈춥니다"


# ----------------------------------------------------------------------------- 미리보기
def preview(path: Path | None, max_side: int = 2048, checker: bool = False):
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.suffix == ".npy":
        arr = imageio.read_npy(p)
        rgb, _ = imageio.colormap_preview(arr)
        return imageio.downscale_for_preview(rgb, max_side)
    img = imageio.read_rgba(p)
    if checker:
        img = imageio.checkerboard_background(img)
    else:
        img = img[..., :3]
    return imageio.downscale_for_preview(img, max_side)


def preview_mask(path: Path | None, max_side: int = 2048):
    if path is None or not Path(path).exists():
        return None
    m = imageio.read_mask(path)
    return imageio.downscale_for_preview(cv2.cvtColor(m, cv2.COLOR_GRAY2RGB), max_side)


def histogram_image(values: list[float], title: str = "", width: int = 640, height: int = 240) -> np.ndarray:
    img = np.full((height, width, 3), 255, np.uint8)
    if values:
        v = np.asarray(values, dtype=np.float64)
        lo, hi = np.percentile(v, [1, 99])
        if hi - lo < 1e-6:
            lo, hi = lo - 1, hi + 1
        hist, edges = np.histogram(v, bins=60, range=(lo, hi))
        hmax = max(1, hist.max())
        bw = (width - 60) / 60
        for i, h in enumerate(hist):
            x0 = int(40 + i * bw)
            y0 = int(height - 30 - h / hmax * (height - 60))
            cv2.rectangle(img, (x0, y0), (int(x0 + bw - 1), height - 30), (60, 90, 200), -1)
        zero_x = int(40 + (0 - lo) / (hi - lo) * (width - 60))
        if 40 <= zero_x <= width - 20:
            cv2.line(img, (zero_x, 20), (zero_x, height - 30), (200, 30, 30), 1)
        cv2.putText(
            img, f"{lo:.1f}", (40, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            img,
            f"{hi:.1f}",
            (width - 70, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(img, title, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def json_md(path: Path | None, keys: list[str] | None = None) -> str:
    if path is None or not Path(path).exists():
        return "_없음_"
    d = imageio.read_json(path)
    rows = [(k, d[k]) for k in (keys or d.keys()) if k in d and not isinstance(d[k], (list, dict))]
    return "\n".join(f"- **{k}**: {v}" for k, v in rows) or "_비어 있음_"


def open_folder(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return "폴더 없음"
    p = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
        return f"열기: {p}"
    except Exception as e:  # noqa: BLE001
        return f"폴더를 열 수 없음 ({e}): {p}"


# ----------------------------------------------------------------------------- 프리셋
def presets_dir() -> Path:
    d = paths.REPO_ROOT / "configs" / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preset_choices() -> list[str]:
    return sorted(p.stem for p in presets_dir().glob("*.yaml"))


def save_preset(job_id: str | None, name: str) -> Any:
    job = get_job(job_id)
    if job is None or not name:
        raise gr.Error("Job 과 프리셋 이름이 필요합니다")
    data = {k: v for k, v in job.params.items() if not k.startswith("_")}
    (presets_dir() / f"{name}.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return gr.update(choices=preset_choices(), value=name)


def load_preset(job_id: str | None, name: str) -> str:
    job = get_job(job_id)
    if job is None or not name:
        raise gr.Error("Job 과 프리셋이 필요합니다")
    data = yaml.safe_load((presets_dir() / f"{name}.yaml").read_text(encoding="utf-8"))
    _deep_update(job.params, data)
    job.save_params()
    return status_md(job.id)
