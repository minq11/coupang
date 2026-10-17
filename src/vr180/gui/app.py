"""Gradio 레이아웃. 상단: 입력 / Job / 썸네일 / 주요 옵션 7개 / 로그.
아래: Stage 0~7 탭은 접힌 Accordion 안 (미리보기·고급용). 실행은 상단 '한번에 진행' 하나로 통일."""

from __future__ import annotations

import os

import gradio as gr

from vr180 import __version__
from vr180.gui import common as C
from vr180.gui import video as video_tab
from vr180.gui.tabs import (
    tab0_input,
    tab1_fov,
    tab2_reproj,
    tab3_outpaint,
    tab4_stereo,
    tab5_depth,
    tab6_right,
    tab7_output,
)
from vr180.models.registry import INPAINTER_CHOICES, INPAINTER_NAMES, inpainter_needs_prompt
from vr180.pipeline.job import load_default_params

TABS = [tab0_input, tab1_fov, tab2_reproj, tab3_outpaint, tab4_stereo, tab5_depth, tab6_right, tab7_output]

RUN_ALL_LABEL = "▶ 한번에 진행 (Stage 0 → 7)"


def _load_main(job_id):
    """상단 '주요 옵션' 10개 위젯의 값을 Job 파라미터에서 읽어 돌려준다 (순서: main_opts 와 동일)."""
    return [
        float(C.get_param(job_id, 1, "hfov_deg", 60.0)),
        float(C.get_param(job_id, 2, "pitch_offset_deg", 0.0)),
        int(C.get_param(job_id, 2, "size", 4096)),
        float(C.get_param(job_id, 2, "extent_h_deg", 180.0)),
        float(C.get_param(job_id, 2, "extent_v_deg", 180.0)),
        float(C.get_param(job_id, 2, "extent_fade_deg", 10.0)),
        C.get_param(job_id, 5, "estimator", "da2_large"),
        float(C.get_param(job_id, 5, "depth_strength", 0.02)),
        C.get_param(job_id, 3, "inpainter", "opencv"),
        C.get_param(job_id, 3, "prompt", ""),
    ]


def build() -> gr.Blocks:
    with gr.Blocks(title=f"vr180 {__version__}", theme=gr.themes.Soft(), css=C.CSS) as demo:
        gr.Markdown("## VR180 변환 파이프라인")
        with gr.Tabs():
            with gr.Tab("📷 사진"):
                _build_photo()
            with gr.Tab("🎞 영상"):
                video_tab.build()
    return demo


def _build_photo() -> None:
    """사진 흐름 — 기존 로직을 그대로 담은 사진 전용 탭."""
    with gr.Column():
        job_id = gr.State(None)
        raw_preview = gr.State(None)  # 가늠자 재렌더용: 업로드 때 저장한 raw(sbs 는 좌반) 다운샘플

        # ───── 상단: 액션 바 (사진 업로드 → 자동으로 작업 폴더 생성 + 화각 자동 추정)
        with gr.Row():
            with gr.Column(scale=3):
                upload = gr.File(
                    label="입력 이미지 (JPG/PNG) — 업로드하면 자동 시작",
                    file_types=["image"],
                )
                input_mode_radio = gr.Radio(
                    choices=[("📷 2D 사진 (mono)", "mono"), ("👓 3D SBS (스테레오)", "sbs")],
                    value="mono",
                    label="입력 모드",
                    info="mono: 2D 사진 1장 → 깊이 추정으로 입체화. sbs: 좌/우가 붙은 스테레오. 모드를 바꾸려면 변경 후 다시 업로드.",
                    elem_classes=["vr180-tip"],
                )
            with gr.Column(scale=2):
                run_all_btn = gr.Button(
                    RUN_ALL_LABEL,
                    variant="primary",
                    elem_classes=["vr180-run"],
                )
                cancel_btn = gr.Button("⏹ 실행 취소", variant="stop")

        # ───── Job 상태: 썸네일 + status + progress
        with gr.Row():
            input_thumb = gr.Image(
                label="입력 SBS",
                type="numpy",
                interactive=False,
                height=180,
                show_download_button=False,
                elem_classes=["vr180-thumb"],
                scale=1,
            )
            with gr.Column(scale=5):
                status = gr.Markdown(C.status_md(None))
                progress = gr.Markdown("")

        # ───── 주요 옵션 (상단으로 뺀 핵심 7개 — 각 Stage 가 아니라 여기서 설정)
        with gr.Group(elem_classes=["vr180-main-opts"]):
            gr.Markdown("### 🎛 주요 옵션 — 결과를 가장 크게 좌우")
            with gr.Row():
                hfov = gr.Slider(
                    C.HFOV_MIN,
                    C.HFOV_MAX,
                    value=60,
                    step=C.HFOV_STEP,
                    label="hfov_deg (수평 화각)",
                    info="업로드 때 자동 추정된 값. 필요하면 수동 조정. 광각 액션캠은 100°+, 일반 사진 70~90°.",
                    elem_classes=["vr180-tip"],
                )
                pitch = gr.Slider(
                    -60,
                    60,
                    value=0,
                    step=0.5,
                    label="pitch_offset_deg (얼굴 올리기)",
                    info="+ = VR 적도를 사진 위쪽(얼굴 방향) 으로. 썸네일의 노란 가로선이 얼굴에 걸치는 값이 정답. 전신 세로샷 보통 +20~+30°.",
                    elem_classes=["vr180-tip"],
                )
                size = gr.Dropdown(
                    [2048, 3072, 4096, 5120, 6144],
                    value=4096,
                    label="눈당 해상도 S",
                    info="S×S half-equirect 캔버스. Quest 2/3 는 4096~5120 권장. 6144 는 PC 뷰어용.",
                    elem_classes=["vr180-tip"],
                )
            with gr.Row():
                ext_h = gr.Number(
                    value=180,
                    label="extent_h_deg (생성 수평°)",
                    info="정면에서 좌우로 몇 ° 까지 생성할지. 180 = 반구 전체 (페이드 꺼짐, 기본). 좁히면 바깥은 어둡게 페이드.",
                    elem_classes=["vr180-tip"],
                )
                ext_v = gr.Number(
                    value=180,
                    label="extent_v_deg (생성 수직°)",
                    info="상하로 몇 ° 까지 생성할지. 180 = 반구 전체 (기본). Quest 시야(≈100°)만 쓰려면 100~120 권장.",
                    elem_classes=["vr180-tip"],
                )
                fade = gr.Slider(
                    0,
                    30,
                    value=10,
                    step=0.5,
                    label="페이드 폭 (°)",
                    info="생성 범위 바깥을 어둡게 섞는 폭. 8° 미만이면 깊이 벽이 느껴짐.",
                    elem_classes=["vr180-tip"],
                )
            with gr.Row():
                estimator = gr.Dropdown(
                    ["none", "da2_small", "da2_base", "da2_large"],
                    value="da2_large",
                    label="mono depth 모델",
                    info="Depth-Anything v2. large=품질 최고/느림, none=모델 없이 (sbs: stereo 외삽, mono: 평면).",
                    elem_classes=["vr180-tip"],
                )
                depth_strength = gr.Slider(
                    0.0,
                    0.06,
                    value=0.02,
                    step=0.002,
                    label="depth_strength (mono 전용, rad)",
                    info="mono 모드에서 각도 disparity 최댓값. ≈1.1° (0.02rad) 가 Quest 에서 편한 수준. sbs 는 자동 피팅이라 영향 없음.",
                    elem_classes=["vr180-tip"],
                )
                inpainter = gr.Dropdown(
                    INPAINTER_NAMES,
                    value="opencv",
                    label="주변부 inpainter (Stage 3)",
                    info=" · ".join(f"{n}={d}" for n, d, _ in INPAINTER_CHOICES),
                    elem_classes=["vr180-tip"],
                )
            # 프롬프트는 opencv 일 때 숨김 (opencv Telea 는 프롬프트 안 씀). inpainter.change 로 토글.
            prompt = gr.Textbox(
                label="프롬프트 (sdxl_lightning / sdxl_lcm 전용)",
                lines=2,
                info="장면 묘사: 실내/실외, 조명, 바닥·천장 재질. 비워 두면 default.yaml 의 기본 프롬프트 사용.",
                elem_classes=["vr180-tip"],
                visible=False,
            )
        main_opts = [hfov, pitch, size, ext_h, ext_v, fade, estimator, depth_strength, inpainter, prompt]

        # ───── 최종 출력 (한번에 진행 완료 시 Stage 7 산출물이 여기 뜸)
        with gr.Group(elem_classes=["vr180-final"]):
            gr.Markdown("### 📸 최종 출력 (VR180 SBS)")
            final_output = gr.Image(
                label=None,
                show_label=False,
                type="numpy",
                interactive=False,
                show_download_button=True,
                height=420,
            )

        # ───── 로그 (간략/상세 탭)
        with gr.Tabs():
            with gr.Tab("📋 간략"):
                log_brief = gr.Textbox(label=None, show_label=False, lines=6, max_lines=14, interactive=False)
            with gr.Tab("🔎 상세 (CMD 원문)"):
                log_full = gr.Textbox(
                    label=None,
                    show_label=False,
                    lines=14,
                    max_lines=30,
                    interactive=False,
                    autoscroll=True,
                )

        shared: dict = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "log": log_brief,
            "log_full": log_full,
            "input_thumb": input_thumb,
            "after_load": [],
            # 탭 안 '고급' 위젯들이 run_all 시 함께 반영되도록 (widget, stage, key, converter) 를 append.
            "param_bindings": [],
        }

        def _load_output(job_id):
            from vr180.pipeline.stage_07_output import output_basename

            job = C.get_job(job_id)
            if job is None:
                return [None]
            png = job.path(7, f"{output_basename(job)}.png")
            return [C.preview(png, 2048) if png.exists() else None]

        # ───── Stage 탭 — 미리보기·고급 설정 (접힘)
        loaders: list[tuple] = [(_load_main, main_opts), (_load_output, [final_output])]
        with gr.Accordion("📂 스테이지별 미리보기 / 고급 설정 (평소에는 접어둠)", open=False):
            with gr.Tabs() as stage_tabs:
                shared["stage_tabs"] = stage_tabs
                for t in TABS:
                    loaders.append(t.build(shared))

        all_outputs = [c for _, outs in loaders for c in outs]

        def load_all(jid):
            vals = []
            for fn, _ in loaders:
                vals += fn(jid)
            return vals

        def _snap_or_default(raw: float) -> tuple[float, str]:
            """raw 가 범위 밖/반올림 필요하면 꼬리표를 함께. UI 로그용."""
            snapped = C.snap_hfov(raw)
            if raw < C.HFOV_MIN:
                return snapped, f" → 하한 {C.HFOV_MIN:g}° 로 클램프"
            if raw > C.HFOV_MAX:
                return snapped, f" → 상한 {C.HFOV_MAX:g}° 로 클램프"
            if abs(snapped - raw) >= C.HFOV_STEP / 2:
                return snapped, f" → {snapped:g}° 로 반올림"
            return snapped, ""

        def do_upload(f, mode):
            """사진 업로드 → Job 자동 생성 + 화각 자동 추정. 실패 시 60° 로 폴백.
            raw_preview State 도 세팅해 pitch 슬라이더 가늠자 재렌더용으로 보관."""
            if not f:
                return None, C.status_md(None), None, None, gr.update(), ""
            j, _, md = C.create_job(f, None, input_mode=mode)
            try:
                raw, conf, name = C.estimate_hfov_from_job(j)
                hfov_v, tag = _snap_or_default(raw)
                log = f"📐 {name}: 추정 {raw:.2f}° (confidence {conf:.2f}){tag} → 슬라이더 반영"
            except Exception as e:  # noqa: BLE001  (geocalib 미설치/수렴 실패 등 모두 60 폴백)
                hfov_v = 60.0
                log = f"⚠️ 화각 자동 추정 실패 ({type(e).__name__}: {e}) — 디폴트 60° 적용"
            # params 에도 즉시 반영 → load_all 재조회 시 일관성 유지. auto_estimate 는 끈 상태로 저장.
            job = C.get_job(j)
            if job is not None:
                job.set_param(1, "hfov_deg", float(hfov_v))
                job.set_param(1, "auto_estimate", False)
                job.save_params()
            # 가늠자 포함 썸네일. pitch 는 업로드 직후 0 (사진 중앙 = VR 적도).
            preview = C.input_preview_for_guide(j)
            thumb = C.draw_pitch_guide(preview, hfov_v, 0.0)
            return j, md, thumb, preview, gr.update(value=hfov_v), log

        upload_evt = upload.upload(
            do_upload,
            [upload, input_mode_radio],
            [job_id, status, input_thumb, raw_preview, hfov, log_brief],
        ).then(load_all, [job_id], all_outputs)

        for fn, ins, outs in shared["after_load"]:
            upload_evt.then(fn, ins, outs)

        # 가늠자 실시간 재렌더: hfov · pitch 가 바뀌면 raw_preview 로부터 input_thumb 다시 그림
        def redraw_guide(preview, hfov_v, pitch_v):
            return C.draw_pitch_guide(preview, hfov_v, pitch_v)

        hfov.change(redraw_guide, [raw_preview, hfov, pitch], [input_thumb])
        pitch.change(redraw_guide, [raw_preview, hfov, pitch], [input_thumb])

        cancel_btn.click(C.cancel_run, [job_id], [log_brief])

        # 프롬프트는 opencv (Telea) 일 때 숨김 — 모델 호출 없는 inpainter 는 프롬프트 안 씀
        inpainter.change(lambda v: gr.update(visible=inpainter_needs_prompt(v)), [inpainter], [prompt])

        # ───── '한번에 진행' — 상단 주요 옵션 + 탭 안 고급 위젯 전부 수집해 param_updates 생성
        bindings = shared["param_bindings"]
        N_MAIN = len(main_opts)

        def run_all_pipeline(jid, *vals):
            main_vals = vals[:N_MAIN]
            tab_vals = vals[N_MAIN:]
            (
                hfov_v,
                pitch_v,
                size_v,
                ext_h_v,
                ext_v_v,
                fade_v,
                estimator_v,
                depth_s_v,
                inpainter_v,
                prompt_v,
            ) = main_vals
            param_updates: dict[int, dict] = {}
            # 탭 고급 위젯 먼저 반영 → 상단 주요 옵션이 뒤이어 덮어씀 (상단이 우선).
            for v, (_, stage, key, conv) in zip(tab_vals, bindings, strict=True):
                param_updates.setdefault(stage, {})[key] = conv(v)
            # Stage 1 은 '한번에 진행' 때 재추정 안 함: 업로드 때 자동 추정하거나 사용자가 수동으로 둔 값을 그대로 쓴다.
            param_updates.setdefault(1, {}).update({"hfov_deg": float(hfov_v), "auto_estimate": False})
            param_updates.setdefault(2, {}).update(
                {
                    "size": int(size_v),
                    "pitch_offset_deg": float(pitch_v),
                    "extent_h_deg": float(ext_h_v),
                    "extent_v_deg": float(ext_v_v),
                    "extent_fade_deg": float(fade_v),
                }
            )
            # 프롬프트를 비우면 default.yaml 기본 프롬프트로 되돌린다 (None 을 넣으면 "None" 문자열이 모델에 들어감).
            default_prompt = str(load_default_params()["stage_03_outpaint"].get("prompt", ""))
            param_updates.setdefault(3, {}).update(
                {"inpainter": inpainter_v, "prompt": (prompt_v or "").strip() or default_prompt}
            )
            param_updates.setdefault(5, {}).update(
                {"estimator": estimator_v, "depth_strength": float(depth_s_v)}
            )
            yield from C.run_pipeline(jid, 0, None, param_updates=param_updates, force_from=False)

        binding_widgets = [w for (w, _, _, _) in bindings]
        run_all_btn.click(
            C.with_button_lock(run_all_pipeline, RUN_ALL_LABEL, n_other=0),
            [job_id, *main_opts, *binding_widgets],
            [
                run_all_btn,
                log_brief,
                log_full,
                status,
                progress,
                stage_tabs,
            ],
        ).then(load_all, [job_id], all_outputs)


def _find_free_port(start: int, span: int = 20) -> int:
    """[start, start+span) 범위에서 bind 가능한 첫 포트. 전부 점유되면 start 그대로 반환
    (호출측이 OSError 로 터뜨리게 둠)."""
    import socket

    for p in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            return p
    return start


def launch(port: int | None = None, share: bool = False) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 외부 호출 없음
    requested = port or int(os.environ.get("VR180_GUI_PORT", "7860"))
    actual = _find_free_port(requested)
    if actual != requested:
        print(f"[VR180] 포트 {requested} 이 사용 중 — {actual} 로 폴백")
    demo = build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=actual, share=share, inbrowser=True, show_error=True)
