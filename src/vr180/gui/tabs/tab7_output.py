"""탭 7 — 출력: 최종 SBS, 파일 경로. 접미사, JPG 품질, XMP. 저장 / 출력 폴더 열기."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 7
RUN_LABEL = "💾 저장 (Stage 7)"


def load(job_id):
    job = C.get_job(job_id)
    if job is None:
        return ["_180_3dh", 95, True, None, None, "_Job 없음_"]
    p = job.stage_params(STAGE)
    suffix = p.get("suffix", "_180_3dh") or ""
    png = job.path(7, f"output_VR180_SBS{suffix}.png")
    jpg = job.path(7, f"output_VR180_SBS{suffix}.jpg")
    files = [str(x) for x in (png, jpg) if x.exists()]
    return [
        suffix,
        int(p.get("jpg_quality", 95)),
        bool(p.get("write_xmp", True)),
        C.preview(png, 2048),
        files or None,
        ("\n".join(f"- `{f}`" for f in files) if files else "_아직 없음_"),
    ]


def build(shared):
    with gr.Tab("7 출력", id=STAGE):
        with gr.Row():
            suffix = gr.Dropdown(
                ["_180_3dh", "_LR", ""],
                value="_180_3dh",
                label="파일명 접미사",
                allow_custom_value=True,
                info="_180_3dh = Meta Quest 가 자동 인식, _LR = 일반 SBS 플레이어용.",
                elem_classes=["vr180-tip"],
            )
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            with gr.Row():
                q = gr.Slider(
                    70,
                    100,
                    value=95,
                    step=1,
                    label="JPG 품질",
                    info="JPG 저장 품질(%). 95 가 품질/용량 균형, 100 은 무손실 수준이지만 용량 2배.",
                    elem_classes=["vr180-tip"],
                )
                xmp = gr.Checkbox(
                    label="XMP GPano 메타데이터",
                    value=True,
                    info="JPG 에 GPano XMP 삽입 → Quest 가 180° SBS 로 자동 인식. 보통 켜둔다.",
                    elem_classes=["vr180-tip"],
                )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"], scale=2)
            open_btn = gr.Button("📁 출력 폴더 열기", scale=1)
        paths_md = gr.Markdown()
        final = gr.Image(label="최종 VR180 SBS", type="numpy", interactive=False)
        files = gr.File(label="다운로드", file_count="multiple")
        outputs = [suffix, q, xmp, final, files, paths_md]

        def run(job_id, s, qq, x):
            yield from C.run_pipeline(
                job_id, STAGE, STAGE, {STAGE: {"suffix": s, "jpg_quality": int(qq), "write_xmp": bool(x)}}
            )

        def open_dir(job_id):
            job = C.get_job(job_id)
            return C.open_folder(job.stage_dir(7) if job else None)

        shared["all_run_buttons"].append(run_btn)

        def _wire():
            others = [b for b in shared["all_run_buttons"] if b is not run_btn]
            run_btn.click(
                C.with_button_lock(run, RUN_LABEL, n_other=len(others)),
                [shared["job_id"], suffix, q, xmp],
                [
                    run_btn,
                    *others,
                    shared["log"],
                    shared["log_full"],
                    shared["status"],
                    shared["progress"],
                    shared["stage_tabs"],
                ],
            ).then(load, [shared["job_id"]], outputs)

        shared["after_build_wires"].append(_wire)
        open_btn.click(open_dir, [shared["job_id"]], [shared["log"]])
    return load, outputs
