"""Gradio 레이아웃. 상단: 입력 업로드 / Job 선택(팝업) / 썸네일 / 프리셋 / 취소 / 로그(간략/상세). 아래: Stage 탭."""

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
    with gr.Blocks(title=f"vr180 {__version__}", theme=gr.themes.Soft(), css=C.CSS) as demo:
        gr.Markdown("## 3D SBS → VR180 SBS 변환 파이프라인")
        job_id = gr.State(None)
        job_list_state = gr.State([])  # 팝업 갤러리 index → job_id 매핑

        # ───── 상단: 액션 바
        with gr.Row():
            with gr.Column(scale=3):
                upload = gr.File(label="입력 3D SBS 이미지 (JPG/PNG)", file_types=["image"])
                new_btn = gr.Button("➕ 새 Job 만들기", variant="primary")
            with gr.Column(scale=2):
                open_btn = gr.Button("📂 기존 Job 열기", size="lg")
                cancel_btn = gr.Button("⏹ 실행 취소", variant="stop")
                run_all_btn = gr.Button(
                    "▶ 한번에 진행 (Stage 0 → 7)",
                    variant="primary",
                    elem_classes=["vr180-run"],
                )
            with gr.Column(scale=2):
                preset_dd = gr.Dropdown(choices=C.preset_choices(), label="프리셋", allow_custom_value=True)
                with gr.Row():
                    preset_save = gr.Button("💾 저장")
                    preset_load = gr.Button("📥 불러오기")

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

        # ───── 로그 (간략/상세 탭)
        with gr.Tabs():
            with gr.Tab("📋 간략"):
                log_brief = gr.Textbox(
                    label=None, show_label=False, lines=6, max_lines=14, interactive=False
                )
            with gr.Tab("🔎 상세 (CMD 원문)"):
                log_full = gr.Textbox(
                    label=None,
                    show_label=False,
                    lines=14,
                    max_lines=30,
                    interactive=False,
                    autoscroll=True,
                )

        # ───── Job 팝업 (visible 토글)
        with gr.Column(visible=False, elem_classes=["vr180-popup"]) as popup:
            gr.Markdown("### 📂 기존 Job 선택 — 썸네일을 클릭하세요")
            job_gallery = gr.Gallery(
                show_label=False,
                columns=4,
                height=440,
                allow_preview=False,
                object_fit="cover",
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 목록 새로고침")
                close_popup_btn = gr.Button("✖ 닫기")

        shared: dict = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "log": log_brief,
            "log_full": log_full,
            "input_thumb": input_thumb,
            "after_load": [],
            # 실행 중 상호 비활성화 대상. run_all_btn 포함, 각 탭 build() 가 자기 run 버튼 append.
            "all_run_buttons": [run_all_btn],
            # 각 탭이 click 와이어링 closure 를 append → 포스트빌드 일괄 실행 (all_run_buttons 완성 후).
            "after_build_wires": [],
        }

        # ───── Stage 탭 (완료 시 자동 다음 탭 전환을 위해 Tabs 참조 공유)
        loaders: list[tuple] = []
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

        def do_new(f):
            j, _, md = C.create_job(f, None)  # 새 Job id 자동
            return j, md, C.input_preview(j)

        def do_open_popup():
            jobs = C.job_choices()
            items = []
            jids = []
            for jid in jobs:
                p = C.job_input_path(jid)
                if p is None:
                    continue
                items.append((str(p), jid))
                jids.append(jid)
            return gr.update(visible=True), items, jids

        def do_pick(jlist, evt: gr.SelectData):
            if not jlist or evt.index is None or evt.index >= len(jlist):
                return gr.update(), gr.update(), gr.update(visible=False), None
            jid = jlist[evt.index]
            return jid, C.status_md(jid), gr.update(visible=False), C.input_preview(jid)

        new_evt = new_btn.click(
            do_new, [upload], [job_id, status, input_thumb]
        ).then(load_all, [job_id], all_outputs)

        pick_evt = job_gallery.select(
            do_pick, [job_list_state], [job_id, status, popup, input_thumb]
        ).then(load_all, [job_id], all_outputs)

        for fn, ins, outs in shared["after_load"]:
            new_evt.then(fn, ins, outs)
            pick_evt.then(fn, ins, outs)

        open_btn.click(do_open_popup, None, [popup, job_gallery, job_list_state])
        refresh_btn.click(do_open_popup, None, [popup, job_gallery, job_list_state])
        close_popup_btn.click(lambda: gr.update(visible=False), None, [popup])

        cancel_btn.click(C.cancel_run, [job_id], [log_brief])
        preset_save.click(C.save_preset, [job_id, preset_dd], [preset_dd])
        preset_load.click(C.load_preset, [job_id, preset_dd], [status]).then(
            load_all, [job_id], all_outputs
        )

        RUN_ALL_LABEL = "▶ 한번에 진행 (Stage 0 → 7)"

        def run_all_pipeline(jid):
            # Stage 0 → 7 쭉 돌림. force_from=False 로 캐시 유효한 Stage 는 재사용.
            yield from C.run_pipeline(jid, 0, None, force_from=False)

        def _wire_run_all():
            other_btns = [b for b in shared["all_run_buttons"] if b is not run_all_btn]
            run_all_btn.click(
                C.with_button_lock(run_all_pipeline, RUN_ALL_LABEL, n_other=len(other_btns)),
                [job_id],
                [
                    run_all_btn,
                    *other_btns,
                    log_brief,
                    log_full,
                    status,
                    progress,
                    stage_tabs,
                ],
            ).then(load_all, [job_id], all_outputs)

        shared["after_build_wires"].append(_wire_run_all)

        # ───── 포스트빌드: 모든 run 버튼 와이어링을 여기서 일괄 실행 (all_run_buttons 완성된 상태)
        for wire in shared["after_build_wires"]:
            wire()
    return demo


def launch(port: int | None = None, share: bool = False) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 외부 호출 없음
    port = port or int(os.environ.get("VR180_GUI_PORT", "7860"))
    demo = build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=port, share=share, inbrowser=True, show_error=True)
