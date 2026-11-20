"""인물 AR 탭 — 사진·영상 속 모든 사람을 잘라 입체 + 투명도 영상으로 만들고, 퀘스트 플레이어로 본다.

흐름: 파일 올리기 → (옵션) → '변환 시작' → 로그·진행률 → 미리보기(왼눈 + 애너글리프)·결과 파일
     → '플레이어 켜기' → 퀘스트에서 http://localhost:8080 (USB + adb reverse)
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

import gradio as gr

from vr180.gui import common as C
from vr180.person.pipeline import IMAGE_EXT, person_params
from vr180.util.log import Cancelled, RunContext

_ctx_holder: dict[str, RunContext] = {}
PLAYER_PORT = 8080


def _run_stream(path: str | None, override: dict):
    if not path:
        raise gr.Error("사진이나 영상을 먼저 올리세요.")
    from vr180.person.pipeline import run_person

    full: list[str] = []
    prog = [""]
    ctx = RunContext(
        log_fn=full.append, progress_fn=lambda f, m: prog.__setitem__(0, f"{int(f * 100)}% · {m}")
    )
    _ctx_holder["cur"] = ctx
    result: dict = {}
    err: list[BaseException] = []

    def worker():
        try:
            result.update(run_person(path, params=override, ctx=ctx))
        except Cancelled:
            ctx.log("취소됨")
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            full.extend(traceback.format_exc().splitlines())

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while th.is_alive():
        yield "\n".join(ctx.lines[-200:]), "⏳ " + (prog[0] or "준비 중…"), None, None, None
        time.sleep(0.5)
    _ctx_holder.pop("cur", None)
    if err:
        yield (
            "\n".join(ctx.lines[-200:] + full[-40:]),
            f"오류: {type(err[0]).__name__}: {err[0]}",
            None,
            None,
            None,
        )
        raise gr.Error(f"{type(err[0]).__name__}: {err[0]}")
    if not result:
        yield "\n".join(ctx.lines[-200:]), "취소됨", None, None, None
        return
    files = [result["output"], result["green"]]
    md = (
        f"✅ 완료 ({result['elapsed_s']}s) · 사람 {result['people']}명 · 눈당 {result['eye_width']}×{result['eye_height']}\n\n"
        f"- 퀘스트용 (위 컬러 / 아래 투명도): `{result['output']}`\n"
        f"- 기존 플레이어용 초록 배경 SBS: `{result['green']}`\n"
        f"- 중간 파일: `{result['job_dir']}`"
    )
    video = result["output"] if result["kind"] == "video" else None
    yield "\n".join(ctx.lines[-200:]), md, result["preview"], files, video


def build() -> None:
    p0 = person_params()
    gr.Markdown("### 🧍 인물 AR — 영상 속 모든 사람을 잘라 퀘스트 패스스루로 내 방에 세우기")
    gr.Markdown(
        "2D 사진이나 영상을 올리면 사람을 전부 찾아(RF-DETR) 영상 내내 따라가고(SAM 2.1), 머리카락까지 경계를 다듬어(BiRefNet) "
        "입체로 만듭니다. 결과는 **위 컬러 / 아래 투명도** 영상 하나와 기존 플레이어용 **초록 배경 SBS** 입니다. "
        "모델은 처음 한 번 `모델 받기 (인물 AR).bat` 으로 받으세요."
    )
    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.File(
                label="사진 또는 영상",
                file_types=sorted(IMAGE_EXT | {".mp4", ".mov", ".mkv", ".webm", ".avi"}),
            )
        with gr.Column(scale=2):
            run_btn = gr.Button("▶ 변환 시작", variant="primary", elem_classes=["vr180-run"])
            cancel_btn = gr.Button("■ 취소")
            player_btn = gr.Button("🥽 플레이어 켜기")
            player_md = gr.Markdown("")

    with gr.Group():
        gr.Markdown("#### 🎛 주요 옵션")
        with gr.Row():
            relief = gr.Slider(
                0,
                30,
                value=float(p0["relief_px"]),
                step=0.5,
                label="입체감 (px, 1080 줄 기준)",
                info="2m 앞 사람 몸 두께 ≈ 6px. 크게 하면 과장돼 보입니다.",
                elem_classes=["vr180-tip"],
            )
            convergence = gr.Radio(
                [("가장 앞 사람", "front"), ("사람 평균", "mean")],
                value=p0["convergence"],
                label="초점 (화면 거리에 놓일 사람)",
                elem_classes=["vr180-tip"],
            )
            max_w = gr.Dropdown(
                [960, 1280, 1440, 1920],
                value=int(p0["max_eye_width"]),
                label="눈당 폭",
                info="1920 이면 출력 3840×2160. 퀘스트에서 끊기면 낮추세요.",
                elem_classes=["vr180-tip"],
            )
        with gr.Row():
            tracker = gr.Radio(
                [("SAM 2.1 (권장)", "sam2"), ("매 프레임 검출", "frame_detect")],
                value=p0["tracker"],
                label="따라가기 (영상)",
                elem_classes=["vr180-tip"],
            )
            sam2_size = gr.Dropdown(
                ["tiny", "small", "base_plus", "large"],
                value=p0["sam2_size"],
                label="SAM 2.1 크기",
                info="12GB 면 small 부터.",
                elem_classes=["vr180-tip"],
            )
            matter = gr.Radio(
                [("BiRefNet-matting", "birefnet_matting"), ("모델 없이 부드럽게", "none")],
                value=p0["matter"],
                label="경계 다듬기",
                elem_classes=["vr180-tip"],
            )
        with gr.Accordion("고급 설정", open=False):
            with gr.Row():
                detector = gr.Dropdown(
                    ["rfdetr_seg_nano", "rfdetr_seg_small", "rfdetr_seg_medium", "rfdetr_seg_large"],
                    value=p0["detector"],
                    label="사람 찾기 모델",
                )
                threshold = gr.Slider(
                    0.1, 0.9, value=float(p0["detect_threshold"]), step=0.05, label="찾기 기준 점수"
                )
                keyframe = gr.Slider(
                    0.2, 2.0, value=float(p0["keyframe_sec"]), step=0.1, label="새 사람 찾는 간격 (초)"
                )
                max_people = gr.Slider(1, 20, value=int(p0["max_people"]), step=1, label="최대 인원")
            with gr.Row():
                min_confirm = gr.Slider(
                    1,
                    4,
                    value=int(p0["min_confirm"]),
                    step=1,
                    label="등록에 필요한 연속 검출 횟수",
                    info="포스터·마네킹 오검출 거르기",
                    elem_classes=["vr180-tip"],
                )
                depth_v = gr.Dropdown(
                    ["vda_small", "da2_video_small", "none"],
                    value=p0["video_depth_model"],
                    label="깊이 (영상)",
                )
                depth_i = gr.Dropdown(
                    ["da2_small", "none"], value=p0["image_depth_model"], label="깊이 (사진)"
                )
                crf = gr.Slider(12, 30, value=int(p0["crf"]), step=1, label="CRF (낮을수록 고화질)")
            mask_file = gr.File(
                label="(사진) 사람 마스크 직접 지정 — 흰색 = 사람. 지정하면 사람 찾기 모델을 쓰지 않습니다",
                file_types=["image"],
            )

    progress = gr.Markdown("")
    with gr.Row():
        preview = gr.Image(
            label="미리보기: 왼눈(체크무늬 = 투명) | 애너글리프(빨강=왼눈, 청록=오른눈)",
            type="filepath",
            interactive=False,
        )
        out_video = gr.Video(label="결과 영상 (위 컬러 / 아래 투명도)", interactive=False)
    files = gr.File(label="결과 파일", file_count="multiple", interactive=False)
    log = gr.Textbox(label="로그", lines=10, max_lines=24, interactive=False)

    def on_run(
        f,
        relief_v,
        conv_v,
        max_w_v,
        tracker_v,
        sam2_v,
        matter_v,
        det_v,
        thr_v,
        kf_v,
        maxp_v,
        conf_v,
        dv_v,
        di_v,
        crf_v,
        mask_v,
    ):
        override = {
            "relief_px": float(relief_v),
            "convergence": conv_v,
            "max_eye_width": int(max_w_v),
            "tracker": tracker_v,
            "sam2_size": sam2_v,
            "matter": matter_v,
            "detector": det_v,
            "detect_threshold": float(thr_v),
            "keyframe_sec": float(kf_v),
            "max_people": int(maxp_v),
            "min_confirm": int(conf_v),
            "video_depth_model": dv_v,
            "image_depth_model": di_v,
            "crf": int(crf_v),
            "mask_path": mask_v if (mask_v and Path(str(f)).suffix.lower() in IMAGE_EXT) else None,
        }
        yield from _run_stream(f, override)

    run_btn.click(
        on_run,
        [
            upload,
            relief,
            convergence,
            max_w,
            tracker,
            sam2_size,
            matter,
            detector,
            threshold,
            keyframe,
            max_people,
            min_confirm,
            depth_v,
            depth_i,
            crf,
            mask_file,
        ],
        [log, progress, preview, files, out_video],
    )

    def on_cancel():
        ctx = _ctx_holder.get("cur")
        if ctx is None:
            return "실행 중인 작업 없음"
        ctx.cancel()
        return "취소 요청함 — 지금 단계가 끝나면 멈춥니다"

    cancel_btn.click(on_cancel, None, [progress])

    def on_player():
        from vr180.player_server import start_background

        try:
            url = start_background("127.0.0.1", PLAYER_PORT)
        except OSError as e:
            return f"플레이어를 켜지 못했습니다 ({e}). 이미 다른 프로그램이 {PLAYER_PORT} 포트를 쓰고 있을 수 있습니다."
        return (
            f"플레이어: [{url}]({url}) (PC 브라우저로 미리보기)\n\n"
            f"퀘스트: USB 연결 → 명령창에서 `adb reverse tcp:{PLAYER_PORT} tcp:{PLAYER_PORT}` → 퀘스트 브라우저에서 `{url}` → **AR 시작**"
        )

    player_btn.click(on_player, None, [player_md])


_unused = C  # 공용 CSS 는 상위 Blocks 에서 적용됨
