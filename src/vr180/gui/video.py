"""영상 변환 탭 — vr180 video 를 GUI 로. 사진과 별개 흐름이라 모듈 분리.

흐름: 영상 업로드 → ffprobe 로 duration 획득 → 기준 프레임 시각 슬라이더 → 프리뷰 (ffmpeg -ss)
     → 주요 옵션 (hfov/pitch/size/extent/depth_model/CRF) → '변환 시작' → 로그·진행률 스트리밍
     → 출력 영상 미리보기 (gr.Video).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from vr180 import paths
from vr180.gui import common as C
from vr180.util import imageio


def _probe(video_path: str) -> dict:
    from vr180.video.frames import probe

    return probe(Path(video_path))


def _ref_thumbnail(video_path: str | None, time_sec: float, max_side: int = 640):
    """지정 시각의 1프레임을 PNG 로 뽑아 다운샘플 numpy 로 돌려준다."""
    if not video_path:
        return None
    from vr180.video.frames import extract_frame_png

    tmp = Path(tempfile.gettempdir()) / "vr180_ref_preview.png"
    try:
        extract_frame_png(Path(video_path), time_sec, tmp)
    except Exception:  # noqa: BLE001
        return None
    img = imageio.read_rgba(tmp)[..., :3]
    return imageio.downscale_for_preview(img, max_side)


def _default_output_path(input_video: str, mode: str) -> str:
    stem = Path(input_video).stem
    outdir = paths.work_dir() / "video_output"
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / f"{stem}_VR180_SBS_{mode}.mp4")


def build() -> None:
    """영상 탭의 모든 위젯·이벤트. 상위 Blocks 안쪽에서 호출된다 (gr.Tab 안)."""
    gr.Markdown("### 🎞 영상 → VR180 SBS 변환 (고정 샷 전용)")
    gr.Markdown(
        "입력 영상을 올리면 길이·fps 를 읽어와 기준 프레임을 선택합니다. "
        "'변환 시작' 누르면 기준 프레임으로 주변부를 1회 생성한 뒤 모든 프레임에 재사용. "
        "mono 모드는 VDA (또는 DA2 폴백) 로 프레임마다 깊이 추정."
    )

    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.File(
                label="입력 영상 (mp4/mov/mkv)",
                file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            )
            info_md = gr.Markdown("_영상을 올리세요._")
            mode = gr.Radio(
                choices=[
                    ("📷 2D 영상 (mono — VDA)", "mono"),
                    ("👓 SBS 영상 (iw3 전처리됨)", "sbs"),
                ],
                value="mono",
                label="입력 모드",
                info="mono: 2D 영상 → VDA 깊이. sbs: 이미 좌/우가 붙은 영상.",
                elem_classes=["vr180-tip"],
            )
        with gr.Column(scale=2):
            run_btn = gr.Button("▶ 변환 시작", variant="primary", elem_classes=["vr180-run"])
            output_path = gr.Textbox(
                label="출력 파일 경로",
                info="비워 두면 work/video_output/ 아래 자동 생성.",
                elem_classes=["vr180-tip"],
            )

    with gr.Row():
        ref_slider = gr.Slider(
            0, 10, value=0, step=0.1,
            label="기준 프레임 시각 (초)",
            info="주변부 생성에 쓸 프레임. 피사체가 가장 가운데/선명한 지점이 유리.",
            elem_classes=["vr180-tip"],
        )
        ref_thumb = gr.Image(label="기준 프레임 미리보기", type="numpy", interactive=False, height=240)

    with gr.Group():
        gr.Markdown("### 🎛 주요 옵션 (기준 프레임 처리에 적용)")
        with gr.Row():
            hfov = gr.Slider(
                C.HFOV_MIN, C.HFOV_MAX, value=60, step=C.HFOV_STEP,
                label="hfov_deg", elem_classes=["vr180-tip"],
            )
            pitch = gr.Slider(
                -60, 60, value=0, step=0.5,
                label="pitch_offset_deg", elem_classes=["vr180-tip"],
            )
            size = gr.Dropdown(
                [2048, 3072, 4096, 5120], value=4096,
                label="눈당 해상도 S", elem_classes=["vr180-tip"],
            )
        with gr.Row():
            ext_h = gr.Number(value=180, label="extent_h_deg", elem_classes=["vr180-tip"])
            ext_v = gr.Number(value=180, label="extent_v_deg", elem_classes=["vr180-tip"])
            fade = gr.Slider(0, 30, value=10, step=0.5, label="fade °", elem_classes=["vr180-tip"])
        with gr.Row():
            depth_model = gr.Dropdown(
                ["vda_large", "vda_small", "da2_video", "none"],
                value="vda_large",
                label="영상 깊이 모델 (mono 전용)",
                info="VDA = 시간축 window 로 떨림↓. VDA 미설치 시 자동 DA2 폴백.",
                elem_classes=["vr180-tip"],
            )
            codec = gr.Dropdown(["libx265", "libx264"], value="libx265", label="코덱", elem_classes=["vr180-tip"])
            crf = gr.Slider(14, 30, value=20, step=1, label="CRF", info="낮을수록 고품질/큰 용량.",
                            elem_classes=["vr180-tip"])
            check_motion = gr.Checkbox(value=True, label="고정 샷 확인", info="phaseCorrelate 로 평균 이동 측정 → 2px 초과 시 경고.",
                                       elem_classes=["vr180-tip"])

    with gr.Group():
        gr.Markdown("### 📸 변환 결과")
        output_preview = gr.Video(label=None, show_label=False, interactive=False, height=360)

    with gr.Tabs():
        with gr.Tab("📋 로그 (간략)"):
            log_brief = gr.Textbox(label=None, show_label=False, lines=8, max_lines=16, interactive=False)
        with gr.Tab("🔎 상세"):
            log_full = gr.Textbox(label=None, show_label=False, lines=14, max_lines=30,
                                  interactive=False, autoscroll=True)
    progress = gr.Markdown("")

    # ───── 이벤트
    def on_upload(f, cur_mode):
        if not f:
            return gr.update(), "_영상을 올리세요._", gr.update(), None, gr.update()
        try:
            info = _probe(f)
        except Exception as e:  # noqa: BLE001
            return (
                gr.update(),
                f"⚠️ 영상 정보 읽기 실패: {e}",
                gr.update(),
                None,
                gr.update(),
            )
        dur = max(info["duration"], 0.1)
        md = (
            f"✅ **{Path(f).name}** · {info['width']}×{info['height']} · "
            f"{info['fps']:.2f}fps · {dur:.2f}s ({info['nb_frames']} 프레임)"
        )
        thumb = _ref_thumbnail(f, 0.0)
        out = _default_output_path(f, cur_mode)
        return (
            gr.update(maximum=dur, value=0.0),
            md,
            gr.update(value=out),
            thumb,
            gr.update(),
        )

    upload.upload(
        on_upload,
        [upload, mode],
        [ref_slider, info_md, output_path, ref_thumb, output_preview],
    )

    def on_ref_change(f, t):
        return _ref_thumbnail(f, float(t)) if f else None

    ref_slider.change(on_ref_change, [upload, ref_slider], [ref_thumb])

    def on_mode_change(cur_mode, f):
        if not f:
            return gr.update()
        return gr.update(value=_default_output_path(f, cur_mode))

    mode.change(on_mode_change, [mode, upload], [output_path])

    def run_video(f, m, t, out_p, hfov_v, pitch_v, size_v, eh, ev, fade_v,
                  dm_v, codec_v, crf_v, check_motion_v):
        if not f:
            raise gr.Error("입력 영상을 올리세요")
        if not out_p:
            out_p = _default_output_path(f, m)
        # 주변부 생성 (ref 프레임 → 이미지 파이프라인) 에 반영할 params override.
        params_override = {
            "stage_01_fov": {"hfov_deg": float(hfov_v), "auto_estimate": False},
            "stage_02_reproj": {
                "size": int(size_v),
                "pitch_offset_deg": float(pitch_v),
                "extent_h_deg": float(eh),
                "extent_v_deg": float(ev),
                "extent_fade_deg": float(fade_v),
            },
        }
        yield from C.run_video_stream(
            input_video=f,
            output_video=out_p,
            mode=m,
            ref_time=float(t),
            depth_model=dm_v,
            codec=codec_v,
            crf=int(crf_v),
            check_motion=bool(check_motion_v),
            params_override=params_override,
        )

    run_btn.click(
        run_video,
        [upload, mode, ref_slider, output_path, hfov, pitch, size, ext_h, ext_v, fade,
         depth_model, codec, crf, check_motion],
        [log_brief, log_full, progress, output_preview],
    )
