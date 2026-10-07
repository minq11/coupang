"""Gradio 레이아웃. 상단: 입력 업로드 / Job 선택 / 프리셋 / 취소 / 로그. 아래: Stage 별 탭 (tabs/ 하나당 파일 하나)."""

from __future__ import annotations

import os

import gradio as gr

from vr180 import __version__
from vr180.gui import common as C
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

TABS = [tab0_input, tab1_fov, tab2_reproj, tab3_outpaint, tab4_stereo, tab5_depth, tab6_right, tab7_output]


def build() -> gr.Blocks:
    with gr.Blocks(title=f"vr180 {__version__}", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 3D SBS → VR180 SBS 변환 파이프라인")
        job_id = gr.State(None)
        with gr.Row():
            with gr.Column(scale=2):
                upload = gr.File(label="입력 3D SBS 이미지 (JPG/PNG)", file_types=["image"])
                with gr.Row():
                    new_id = gr.Textbox(label="새 Job id (비우면 자동)", scale=2)
                    new_btn = gr.Button("새 Job", variant="primary", scale=1)
            with gr.Column(scale=2):
                job_dd = gr.Dropdown(choices=C.job_choices(), label="기존 Job", allow_custom_value=False)
                with gr.Row():
                    open_btn = gr.Button("열기")
                    refresh_btn = gr.Button("목록 새로고침")
                    cancel_btn = gr.Button("실행 취소", variant="stop")
            with gr.Column(scale=1):
                preset_dd = gr.Dropdown(choices=C.preset_choices(), label="프리셋", allow_custom_value=True)
                with gr.Row():
                    preset_save = gr.Button("저장")
                    preset_load = gr.Button("불러오기")
        status = gr.Markdown(C.status_md(None))
        log = gr.Textbox(label="로그", lines=6, max_lines=14, interactive=False)

        shared = {"job_id": job_id, "status": status, "log": log, "after_load": []}
        loaders: list[tuple] = []
        with gr.Tabs():
            for t in TABS:
                loaders.append(t.build(shared))

        all_outputs = [c for _, outs in loaders for c in outs]

        def load_all(jid):
            vals = []
            for fn, _ in loaders:
                vals += fn(jid)
            return vals

        def do_new(f, jid):
            j, dd, md = C.create_job(f, jid)
            return j, dd, md

        ev = new_btn.click(do_new, [upload, new_id], [job_id, job_dd, status]).then(
            load_all, [job_id], all_outputs
        )
        ev2 = open_btn.click(lambda j: (j, C.status_md(j)), [job_dd], [job_id, status]).then(
            load_all, [job_id], all_outputs
        )
        for fn, ins, outs in shared["after_load"]:
            ev.then(fn, ins, outs)
            ev2.then(fn, ins, outs)
        refresh_btn.click(lambda: gr.update(choices=C.job_choices()), None, [job_dd])
        cancel_btn.click(C.cancel_run, [job_id], [log])
        preset_save.click(C.save_preset, [job_id, preset_dd], [preset_dd])
        preset_load.click(C.load_preset, [job_id, preset_dd], [status]).then(load_all, [job_id], all_outputs)
    return demo


def launch(port: int | None = None, share: bool = False) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 외부 호출 없음
    port = port or int(os.environ.get("VR180_GUI_PORT", "7860"))
    demo = build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=port, share=share, inbrowser=True, show_error=True)
