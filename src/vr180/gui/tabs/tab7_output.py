"""탭 7 — 출력: 최종 SBS, 파일 경로. 고급: 접미사, JPG 품질, XMP. 저장은 상단 '한번에 진행' 에 포함."""

from __future__ import annotations

import gradio as gr

from vr180.gui import common as C

STAGE = 7


def load(job_id):
    from vr180.pipeline.stage_07_output import output_basename

    job = C.get_job(job_id)
    if job is None:
        return ["_180_3dh", 95, True, None, None, "_Job 없음_"]
    p = job.stage_params(STAGE)
    suffix = p.get("suffix", "_180_3dh") or ""
    base = output_basename(job)
    png = job.path(7, f"{base}.png")
    jpg = job.path(7, f"{base}.jpg")
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
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            suffix = gr.Dropdown(
                ["_180_3dh", "_LR", ""],
                value="_180_3dh",
                label="파일명 접미사",
                allow_custom_value=True,
                info="_180_3dh = Meta Quest 가 자동 인식, _LR = 일반 SBS 플레이어용.",
                elem_classes=["vr180-tip"],
            )
            with gr.Row():
                q = gr.Slider(
                    70, 100, value=95, step=1,
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
            open_btn = gr.Button("📁 출력 폴더 열기")
        paths_md = gr.Markdown()
        final = gr.Image(label="최종 VR180 SBS", type="numpy", interactive=False)
        files = gr.File(label="다운로드", file_count="multiple")
        outputs = [suffix, q, xmp, final, files, paths_md]

        def open_dir(job_id):
            job = C.get_job(job_id)
            return C.open_folder(job.stage_dir(7) if job else None)

        shared["param_bindings"].extend([
            (suffix, STAGE, "suffix", str),
            (q, STAGE, "jpg_quality", lambda v: int(v)),
            (xmp, STAGE, "write_xmp", bool),
        ])
        open_btn.click(open_dir, [shared["job_id"]], [shared["log"]])
    return load, outputs
