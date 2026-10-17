"""GUI 탭들이 공유하는 얇은 헬퍼. 파이프라인 로직은 전부 pipeline/ 에 있고 여기서는 호출만 한다."""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import threading
import time
import traceback
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


def log_dir() -> Path:
    d = paths.REPO_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _StreamTee(io.TextIOBase):
    """stderr/stdout 복제기. 콘솔은 원본 그대로, 파일/리스트에는 완성 라인(개행 기준)만 기록.
    tqdm \r 진행바는 마지막 세그먼트만 보존 (스팸 완화)."""

    def __init__(self, console, *files, buffer: list[str] | None = None):
        self._console = console
        self._files = files
        self._buffer = buffer
        self._buf = ""

    def write(self, s: str) -> int:
        try:
            self._console.write(s)
            self._console.flush()
        except Exception:
            pass
        self._buf += s
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line, self._buf = self._buf[:nl], self._buf[nl + 1 :]
            if "\r" in line:
                line = line.rsplit("\r", 1)[-1]
            if not line.strip():
                continue
            for f in self._files:
                try:
                    f.write(line + "\n")
                    f.flush()
                except Exception:
                    pass
            if self._buffer is not None:
                self._buffer.append(line)
        return len(s)

    def flush(self) -> None:
        try:
            self._console.flush()
        except Exception:
            pass
        for f in self._files:
            try:
                f.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return self._console.isatty()
        except Exception:
            return False

STATUS_ICON = {"done": "✅", "stale": "⚠️", "missing": "⬜", "skipped": "➖"}
STATUS_STYLE = {
    "done": "background:#d1fae5;color:#065f46;padding:3px 8px;border-radius:6px;font-weight:600;border:1px solid #10b981",
    "stale": "background:#fef3c7;color:#92400e;padding:3px 8px;border-radius:6px;border:1px solid #f59e0b",
    "missing": "background:#f3f4f6;color:#6b7280;padding:3px 8px;border-radius:6px;border:1px solid #d1d5db",
    "skipped": "background:#e5e7eb;color:#4b5563;padding:3px 8px;border-radius:6px;border:1px dashed #9ca3af;font-style:italic",
}
_running: dict[str, RunContext] = {}

RUN_LOCK_LABEL = "⏳ 실행 중..."

# meta.json 이 없을 때 쓰는 fallback 예상 시간(초). 한 번 돌리고 나면 각 Job 의 실제 elapsed_s 가 사용됨.
DEFAULT_STAGE_ELAPSED = {
    0: 3.0,
    1: 5.0,
    2: 6.0,
    3: 180.0,
    4: 30.0,
    5: 60.0,
    6: 90.0,
    7: 3.0,
}

_RE_STAGE_RUN = re.compile(r"Stage (\d+) \S+: 실행\s*$")
_RE_STAGE_DONE = re.compile(r"Stage (\d+) \S+: 완료 \(([\d.]+)s\)\s*$")
_RE_STAGE_CACHE = re.compile(r"Stage (\d+) \S+: 캐시 사용\s*$")


def _fmt_s(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _stage_eta(job: Job, i: int) -> float:
    m = job.meta(i)
    if m and isinstance(m.get("elapsed_s"), (int, float)):
        return float(m["elapsed_s"])
    return DEFAULT_STAGE_ELAPSED.get(i, 30.0)


def _planned_stages(job: Job, from_stage: int, to_stage: int) -> list[int]:
    """from..to 범위의 stage + 산출물이 없어서 자동 실행될 이전 stage."""
    pre = [i for i in range(0, from_stage) if not job.outputs_exist(i)]
    return pre + list(range(from_stage, to_stage + 1))

# info= 로 넣은 설명을 평소에는 숨기고 라벨/위젯 호버 시에만 보이게. 액션바 통일 간격.
CSS = """
.vr180-tip .info,
.vr180-tip .label-wrap + .info,
.vr180-tip .svelte-1gfkn6j,
.vr180-tip span.info,
.vr180-tip > div > .info {
    display: none !important;
}
.vr180-tip:hover .info,
.vr180-tip:hover .label-wrap + .info,
.vr180-tip:hover .svelte-1gfkn6j,
.vr180-tip:hover span.info,
.vr180-tip:hover > div > .info {
    display: block !important;
    color: #4b5563;
    font-size: 11px;
    line-height: 1.3;
    margin-top: 2px;
    animation: vr180-fade 0.12s ease-out;
}
@keyframes vr180-fade { from { opacity: 0; } to { opacity: 1; } }

.vr180-actions {
    gap: 8px !important;
    margin: 8px 0 4px !important;
    padding: 8px;
    background: #f3f4f6;
    border-radius: 8px;
}
.dark .vr180-actions { background: #1f2937; }
.vr180-actions .wrap { align-items: stretch; }

.vr180-run button[disabled],
.vr180-run button[aria-disabled="true"] {
    cursor: progress !important;
    opacity: 0.85;
}

/* "기존 Job 열기" 팝업 — 반고정 패널. Gradio 레이아웃 안이지만 shadow/border 로 떠보이게. */
.vr180-popup {
    position: fixed !important;
    top: 50% !important; left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 1000;
    max-width: 920px; width: 85vw;
    max-height: 80vh; overflow: auto;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
    padding: 20px !important;
}
.dark .vr180-popup { background: #111827; border-color: #374151; }
.vr180-popup::before {
    content: "";
    position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: -1;
}

/* 썸네일 영역 */
.vr180-thumb { max-height: 180px; }
"""


def with_button_lock(gen_fn, idle_label: str, n_other: int = 0):
    """(log_brief, log_full, status, progress_md, button_short, tab_update) 6-tuple 을
    yield 하는 generator 를 감싸서 실행 중 버튼 잠금 + 라벨 교체. n_other 개의 다른 run 버튼
    들도 함께 비활성화 → 활성화. finally 로 에러 시에도 반드시 복구.

    click outputs 순서: [clicked_btn, *other_btns, log_brief, log_full, status, progress, stage_tabs].
    총 길이 = 1 + n_other + 5."""

    def wrapped(*args):
        other_disable = [gr.update(interactive=False)] * n_other
        other_noop = [gr.update()] * n_other
        other_enable = [gr.update(interactive=True)] * n_other
        try:
            yield (
                gr.update(value=RUN_LOCK_LABEL, interactive=False),
                *other_disable,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
            for log_brief, log_full, status, progress_md, button_short, tab_update in gen_fn(*args):
                yield (
                    gr.update(value=button_short, interactive=False),
                    *other_noop,
                    log_brief,
                    log_full,
                    status,
                    progress_md,
                    tab_update,
                )
        finally:
            yield (
                gr.update(value=idle_label, interactive=True),
                *other_enable,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

    return wrapped


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
        return "_Job 없음 — 입력 이미지를 올리고 **새 Job** 을 누르거나 **기존 Job 열기** 로 선택._"
    cells = []
    for i, st in enumerate(stages()):
        s = job.status(i)
        cells.append(
            f'<span style="{STATUS_STYLE[s]};margin-right:4px">{STATUS_ICON[s]} {i} {st.NAME}</span>'
        )
    mode_label = {"mono": "📷 mono (2D)", "sbs": "👓 sbs (3D)"}.get(job.input_mode, job.input_mode)
    return (
        f"**Job** `{job.id}` &nbsp;·&nbsp; **{mode_label}** &nbsp;&nbsp; "
        + "".join(cells)
        + '  \n<span style="color:#6b7280;font-size:11px">✅ 완료 · ⚠️ stale (파라미터 바뀜) · ⬜ 없음 · ➖ 모드에서 건너뜀</span>'
    )


def create_job(file_path: str | None, job_id: str | None, input_mode: str | None = None) -> tuple[str, Any, str]:
    if not file_path:
        raise gr.Error("입력 이미지를 먼저 올리세요")
    job = Job.create(file_path, job_id=(job_id or None), input_mode=input_mode)
    return job.id, gr.update(choices=job_choices(), value=job.id), status_md(job.id)


def job_input_path(job_id: str | None) -> Path | None:
    """Job 의 입력 SBS 이미지 경로. 썸네일/미리보기용."""
    job = get_job(job_id)
    if job is None:
        return None
    try:
        return job.input_path
    except FileNotFoundError:
        return None


def input_preview(job_id: str | None, max_side: int = 640):
    """Job 의 입력 SBS 이미지를 다운스케일해서 numpy 로 반환."""
    p = job_input_path(job_id)
    if p is None:
        return None
    img = imageio.read_rgba(p)[..., :3]
    return imageio.downscale_for_preview(img, max_side)


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


# ----------------------------------------------------------------------------- 실행 (제너레이터: 로그/진행률을 흘려보낸다)
def run_pipeline(
    job_id: str | None,
    from_stage: int,
    to_stage: int | None,
    param_updates: dict[int, dict[str, Any]] | None = None,
    model_updates: dict | None = None,
    force_from: bool = True,
):
    """yield (log_text, status_md, progress_md, button_short).
    과거 meta.json 의 elapsed_s 를 기반으로 Stage ETA 를 추정.
    GUI 버튼 핸들러에서 `yield from` 으로 쓴다."""
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

    _to = (len(stages()) - 1) if to_stage is None else to_stage
    plan = _planned_stages(job, from_stage, _to)
    preds: dict[int, float] = {i: _stage_eta(job, i) for i in plan}
    actuals: dict[int, float] = {}
    cached: set[int] = set()
    state = {"current": None, "started_at": 0.0}  # mutable so closures can see
    scan_pos = [0]
    last_completed = [-1]  # Stage 완료 알림 중복 방지. 7(마지막) 완료 시엔 전환 안 함.
    N_STAGES = len(stages())

    def _tab_update_if_newly_done():
        done = sorted(set(actuals.keys()) | cached)
        if not done:
            return gr.update()
        latest = done[-1]
        if latest > last_completed[0] and latest + 1 < N_STAGES:
            last_completed[0] = latest
            return gr.update(selected=latest + 1)
        last_completed[0] = max(last_completed[0], latest)
        return gr.update()

    # 로그 파일 두 곳에 기록. latest.log 는 매 실행마다 덮어쓰기, run.log 는 Job 영구 보관.
    latest_fh = (log_dir() / "latest.log").open("w", encoding="utf-8", buffering=1)
    job_fh = (job.dir / "run.log").open("a", encoding="utf-8", buffering=1)
    header = (
        f"\n{'=' * 72}\n"
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] job={job.id}  "
        f"from={from_stage} to={_to}  plan={plan}\n{'=' * 72}\n"
    )
    for fh in (latest_fh, job_fh):
        fh.write(header)

    # 상세 로그: ctx.log + stderr/stdout 전부 (= latest.log 내용과 동일한 라인). GUI 상세 탭에서 보여줌.
    full_lines: list[str] = header.strip("\n").split("\n")

    def _file_log(line: str) -> None:
        for fh in (latest_fh, job_fh):
            try:
                fh.write(line + "\n")
            except Exception:
                pass
        full_lines.append(line)

    ctx = RunContext(log_fn=_file_log)
    _running[job.id] = ctx
    err: list[BaseException] = []

    def _scan(limit: int) -> None:
        while scan_pos[0] < limit:
            line = ctx.lines[scan_pos[0]]
            scan_pos[0] += 1
            if (m := _RE_STAGE_RUN.search(line)) is not None:
                state["current"] = int(m.group(1))
                state["started_at"] = time.time()
                continue
            if (m := _RE_STAGE_DONE.search(line)) is not None:
                idx = int(m.group(1))
                actuals[idx] = float(m.group(2))
                if state["current"] == idx:
                    state["current"] = None
                continue
            if (m := _RE_STAGE_CACHE.search(line)) is not None:
                cached.add(int(m.group(1)))

    def _progress():
        now = time.time()
        done_elapsed = sum(actuals.values())
        cur = state["current"]
        if cur is not None:
            cur_elapsed = now - state["started_at"]
            cur_pred = max(preds.get(cur, 0.0), cur_elapsed * 1.1)
        else:
            cur_elapsed = 0.0
            cur_pred = 0.0
        rem = [i for i in plan if i != cur and i not in actuals and i not in cached]
        rem_pred = sum(preds.get(i, 0.0) for i in rem)
        total_est = done_elapsed + cur_pred + rem_pred
        elapsed = done_elapsed + cur_elapsed
        remaining = max(0.0, total_est - elapsed)
        done_cnt = len(actuals) + len(cached)
        if cur is not None:
            try:
                name = stages()[cur].NAME
            except (IndexError, AttributeError):
                name = ""
            pos = done_cnt + 1
            label = f"Stage {cur} `{name}` · {pos}/{len(plan)} 진행 중"
        elif done_cnt >= len(plan):
            label = f"완료 ({done_cnt}/{len(plan)})"
        elif done_cnt > 0:
            label = f"다음 Stage 준비 ({done_cnt}/{len(plan)})"
        else:
            label = "준비 중"
        md = (
            f"**{label}** · 경과 `{_fmt_s(elapsed)}` / 예상 `{_fmt_s(total_est)}`"
            f" · 남은 `~{_fmt_s(remaining)}`"
        )
        short = f"⏳ {_fmt_s(elapsed)} / ~{_fmt_s(total_est)}"
        return md, short

    def worker():
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _StreamTee(orig_out, latest_fh, job_fh, buffer=full_lines)
        sys.stderr = _StreamTee(orig_err, latest_fh, job_fh, buffer=full_lines)
        try:
            job.run(from_stage=from_stage, to_stage=to_stage, ctx=ctx, force_from=force_from)
        except Cancelled:
            ctx.log("취소됨")
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            tb = traceback.format_exc()
            ctx.log(f"오류: {type(e).__name__}: {e}")
            for fh in (latest_fh, job_fh):
                try:
                    fh.write(tb + "\n")
                except Exception:
                    pass
            full_lines.extend(tb.splitlines())
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while th.is_alive():
            _scan(len(ctx.lines))
            md, short = _progress()
            yield (
                "\n".join(ctx.lines[-200:]),
                "\n".join(full_lines[-800:]),
                status_md(job.id),
                md,
                short,
                _tab_update_if_newly_done(),
            )
            time.sleep(0.5)
        th.join()
        _scan(len(ctx.lines))
    finally:
        _running.pop(job.id, None)
        for fh in (latest_fh, job_fh):
            try:
                fh.close()
            except Exception:
                pass
    md, short = _progress()
    yield (
        "\n".join(ctx.lines[-200:]),
        "\n".join(full_lines[-800:]),
        status_md(job.id),
        md,
        short,
        _tab_update_if_newly_done(),
    )
    if err:
        raise gr.Error(f"{type(err[0]).__name__}: {err[0]}")


# 상단 hfov 슬라이더 레인지. 자동 추정 결과를 반영할 때 반올림/클램프 기준.
HFOV_MIN = 20.0
HFOV_MAX = 120.0
HFOV_STEP = 0.5


def draw_pitch_guide(rgb, hfov_deg: float, pitch_deg: float, max_side: int = 640):
    """입력 썸네일 위에 노란 가로선 1개 — '현재 pitch 값이 가리키는 세로 위치'.

    선을 얼굴에 걸치면 그 pitch 가 'VR 적도=얼굴'을 만드는 값. 수식:
      focal_px = w / (2·tan(hfov/2));   row = h/2 − focal_px · tan(pitch_rad)
    """
    import math

    import cv2

    if rgb is None:
        return None
    img = imageio.downscale_for_preview(rgb, max_side).copy()
    h, w = img.shape[:2]
    f_px = w / (2.0 * math.tan(math.radians(float(hfov_deg)) / 2.0))
    row = int(round(h / 2.0 - f_px * math.tan(math.radians(float(pitch_deg)))))
    row = max(0, min(h - 1, row))
    cv2.line(img, (0, row), (w - 1, row), (255, 220, 0), 2)
    cv2.putText(
        img,
        f"pitch {pitch_deg:+.1f}°  (hfov {hfov_deg:g}°)",
        (10, max(18, row - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def input_preview_for_guide(job_id: str | None, max_side: int = 640):
    """Job 의 입력 raw 이미지(sbs 는 좌반) 를 downsample 해서 돌려준다. 가늠자 재렌더용 State."""
    job = get_job(job_id)
    if job is None:
        return None
    try:
        raw = imageio.read_rgba(job.input_path)[..., :3]
    except FileNotFoundError:
        return None
    if job.input_mode == "sbs":
        raw = raw[:, : raw.shape[1] // 2]
    return imageio.downscale_for_preview(raw, max_side)


def snap_hfov(hfov: float) -> float:
    """hfov 값을 슬라이더 step 에 맞춰 반올림하고 [HFOV_MIN, HFOV_MAX] 로 클램프."""
    v = round(float(hfov) / HFOV_STEP) * HFOV_STEP
    return float(min(max(v, HFOV_MIN), HFOV_MAX))


def estimate_hfov_from_job(job_id: str | None) -> tuple[float, float, str]:
    """선택된 FOV estimator (기본 GeoCalib) 를 Job 입력 이미지에 돌려 (hfov_deg_raw, confidence, model) 반환.
       Stage 0 산출물(C.png/L.png)이 있으면 그걸, 없으면 raw input (sbs 는 좌반) 을 쓴다.
       반환값은 raw — 반올림/클램프는 호출측에서 snap_hfov 로. 추정 자체 실패 시 gr.Error."""
    job = get_job(job_id)
    if job is None:
        raise gr.Error("Job 을 먼저 만들거나 여세요")
    from vr180.models.registry import get_fov_estimator

    name = job.stage_params(1).get("estimator") or "geocalib"
    try:
        est = get_fov_estimator(name, job.model_params())
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"FOV estimator '{name}' 로드 실패: {e}") from e
    if est is None:
        raise gr.Error(f"'{name}' 은 수동 모드 — 추정기가 아닙니다")
    src = job.path(0, "C.png" if job.input_mode == "mono" else "L.png")
    if src.exists():
        img = imageio.read_rgb(src)
    else:
        raw = imageio.read_rgba(job.input_path)[..., :3]
        img = raw[:, : raw.shape[1] // 2] if job.input_mode == "sbs" else raw
    try:
        with est:
            hfov, conf = est.estimate_hfov(img)
            model_name = est.info().name
    except ModuleNotFoundError as e:
        hint = {"geocalib": "uv sync --extra geocalib"}.get(name, f"'{name}' 선택 의존성 설치 필요")
        raise gr.Error(f"'{name}' 패키지 미설치 — 터미널에서 `{hint}` 실행 후 GUI 재시작") from e
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"FOV 추정 실패: {e}") from e
    # 극단적으로 비정상인 값만 거부 (model 자체가 수렴 실패한 경우). 슬라이더 범위 안으로 조이는 건 호출측.
    if hfov is None or hfov <= 5 or hfov >= 175:
        raise gr.Error(f"FOV 추정 결과가 비정상 (hfov={hfov}, conf={conf:.2f}) — 모델 수렴 실패")
    return float(hfov), float(conf), model_name


def run_video_stream(
    input_video: str,
    output_video: str,
    *,
    mode: str,
    ref_time: float,
    depth_model: str,
    codec: str,
    crf: int,
    check_motion: bool,
    params_override: dict | None = None,
):
    """gradio generator — vr180.video.run.run_video 를 백그라운드 스레드로 돌리면서 ctx.lines 를
    0.5s 간격으로 yield 한다. 반환: (log_brief, log_full, progress_md, output_video_path | None)."""
    import threading
    import time as _time
    import traceback

    from vr180.util.log import RunContext
    from vr180.video.run import run_video as _run

    full_lines: list[str] = []
    # 영상 변환 로그도 파일로 남긴다 (사진은 latest.log). GUI 가 멈추거나 닫혀도 원인을 볼 수 있게.
    video_log = (log_dir() / "video_latest.log").open("w", encoding="utf-8", buffering=1)

    def _file_log(line: str) -> None:
        full_lines.append(line)
        video_log.write(line + "\n")

    latest_progress: list[str] = [""]  # 프레임 루프 진행률 (ctx.progress) 을 GUI 에 보이기 위한 슬롯

    def _progress(frac: float, msg: str) -> None:
        latest_progress[0] = f"{int(frac * 100)}% · {msg}" if msg else f"{int(frac * 100)}%"

    ctx = RunContext(log_fn=_file_log, progress_fn=_progress)
    err: list[BaseException] = []
    result: dict = {}

    def worker() -> None:
        try:
            r = _run(
                input_video=input_video,
                output_video=output_video,
                input_mode=mode,
                ref_time=ref_time,
                depth_model=depth_model,
                codec=codec,
                crf=int(crf),
                check_motion=check_motion,
                params_override=params_override,
                ctx=ctx,
            )
            result.update(r)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            tb = traceback.format_exc().splitlines()
            full_lines.extend(tb)
            video_log.write("\n".join(tb) + "\n")
        finally:
            video_log.close()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while th.is_alive():
        yield (
            "\n".join(ctx.lines[-200:]),
            "\n".join(full_lines[-800:]),
            ("⏳ 진행 중… " + latest_progress[0]) if latest_progress[0] else "⏳ 진행 중… (기준 프레임 처리)",
            None,
        )
        _time.sleep(0.5)
    th.join()
    msg = "완료" if not err else f"오류: {type(err[0]).__name__}: {err[0]}"
    yield (
        "\n".join(ctx.lines[-200:]),
        "\n".join(full_lines[-800:]),
        msg + (f" · {result.get('frames', '?')} 프레임 → `{result.get('output', output_video)}`" if not err else ""),
        output_video if (not err and Path(output_video).exists()) else None,
    )
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
