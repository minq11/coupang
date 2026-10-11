"""탭 3 — 주변부 생성: 타일 그리드(전/후), pano_L. 프롬프트/seed/steps/guidance/타일 FOV/페더, 타일 개별 재생성, 외부 수정본."""

from __future__ import annotations

import shutil

import gradio as gr

from vr180.geometry.reproject import make_tile_grid
from vr180.gui import common as C
from vr180.util import imageio

STAGE = 3
RUN_LABEL = "▶ 전체 실행 (Stage 3)"
ONLY_LABEL = "🔁 선택 타일만 재생성"
KEYS = [
    "inpainter",
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "guidance",
    "tile_fov_deg",
    "feather_px",
    "tile_res",
]
DEF = {
    "inpainter": "opencv",
    "prompt": "",
    "negative_prompt": "text, watermark, people",
    "seed": 42,
    "steps": 30,
    "guidance": 30.0,
    "tile_fov_deg": 90.0,
    "feather_px": 64,
    "tile_res": 1024,
}


def tile_names(job_id):
    job = C.get_job(job_id)
    p = job.stage_params(STAGE) if job else {}
    return [
        t.name
        for t in make_tile_grid(
            p.get("tile_yaws_deg", [-60, 0, 60]), p.get("tile_pitches_deg", [-45, 0, 45]), 90, 64
        )
    ]


def load(job_id):
    job = C.get_job(job_id)
    vals = [C.get_param(job_id, STAGE, k, DEF[k]) for k in KEYS]
    if job is None:
        return vals + [gr.update(choices=[], value=[]), [], None]
    td = job.stage_dir(3) / "tiles"
    gallery = []
    for n in tile_names(job_id):
        for suf in ("in", "out"):
            p = td / f"{n}_{suf}.png"
            if p.exists():
                gallery.append((imageio.downscale_for_preview(imageio.read_rgb(p), 512), f"{n} {suf}"))
    return vals + [
        gr.update(choices=tile_names(job_id), value=[]),
        gallery,
        C.preview(job.path(3, "pano_L.png"), 2048),
    ]


def build(shared):
    with gr.Tab("3 주변부 생성", id=STAGE):
        with gr.Row():
            inpainter = gr.Dropdown(
                ["opencv", "sdxl", "flux"],
                value="opencv",
                label="inpainter",
                info="opencv=모델 없이 Telea (빠르고 묘사 없음, 주변시 방해↓, 기본), sdxl=빠름/보통 품질, flux=느리지만 자연스러움.",
                elem_classes=["vr180-tip"],
            )
        prompt = gr.Textbox(
            label="프롬프트",
            lines=2,
            info="장면 묘사: 실내/실외, 조명, 바닥·천장 재질. 비워 두면 default.yaml 의 기본 프롬프트 사용.",
            elem_classes=["vr180-tip"],
        )
        neg = gr.Textbox(
            label="네거티브",
            value=DEF["negative_prompt"],
            info="피하고 싶은 요소. 사람/워터마크/글자 등을 명시하면 결과가 깔끔해진다.",
            elem_classes=["vr180-tip"],
        )
        with gr.Accordion("고급 설정 (잘 안 바꿈)", open=False):
            with gr.Row():
                seed = gr.Number(
                    value=42,
                    label="seed",
                    precision=0,
                    info="같은 seed → 같은 결과 (재현용). 결과가 마음에 안 들면 숫자를 바꿔본다.",
                    elem_classes=["vr180-tip"],
                )
                steps = gr.Slider(
                    1,
                    60,
                    value=30,
                    step=1,
                    label="steps",
                    info="디노이즈 단계 수. 많을수록 품질↑ 속도↓. SDXL 30, FLUX 20~30 권장.",
                    elem_classes=["vr180-tip"],
                )
                guidance = gr.Slider(
                    0,
                    40,
                    value=30.0,
                    step=0.5,
                    label="guidance",
                    info="프롬프트 반영 강도. FLUX 는 30 근처, SDXL 는 7 근처 권장. 현재 기본 inpainter 는 FLUX.",
                    elem_classes=["vr180-tip"],
                )
            with gr.Row():
                tile_fov = gr.Slider(
                    60,
                    120,
                    value=90,
                    step=1,
                    label="타일 FOV (°)",
                    info="각 타일이 포괄하는 가로 시야각. 커질수록 왜곡은 늘고 이음새는 줄어든다.",
                    elem_classes=["vr180-tip"],
                )
                feather = gr.Slider(
                    0,
                    256,
                    value=64,
                    step=4,
                    label="페더 px",
                    info="타일 가장자리 블렌딩 폭. 크면 부드럽지만 디테일이 뭉개질 수 있음.",
                    elem_classes=["vr180-tip"],
                )
                tile_res = gr.Dropdown(
                    [512, 768, 1024],
                    value=1024,
                    label="타일 해상도",
                    info="디퓨전 모델이 받는 입력 해상도. 1024 가 품질/VRAM 균형.",
                    elem_classes=["vr180-tip"],
                )
        with gr.Row(elem_classes=["vr180-actions"]):
            run_btn = gr.Button(RUN_LABEL, variant="primary", elem_classes=["vr180-run"])
        with gr.Accordion("선택 타일만 재생성", open=False):
            with gr.Row():
                only = gr.CheckboxGroup(choices=[], label="재생성할 타일")
            tile_prompts = gr.Dataframe(
                headers=["타일", "타일별 프롬프트 (비우면 공통 프롬프트 사용)"],
                datatype=["str", "str"],
                row_count=(0, "dynamic"),
                col_count=(2, "fixed"),
                interactive=True,
                label=None,
                show_label=False,
                wrap=True,
            )
            with gr.Row(elem_classes=["vr180-actions"]):
                only_btn = gr.Button(ONLY_LABEL, elem_classes=["vr180-run"])
        with gr.Accordion("외부에서 수정한 pano_L.png 불러오기", open=False):
            upload = gr.File(label="pano_L.png (덮어쓰기 → Stage 5 부터 재실행)")
            with gr.Row(elem_classes=["vr180-actions"]):
                upload_btn = gr.Button("📥 불러오기")
        gallery = gr.Gallery(label="타일 (in / out)", columns=4, height=360)
        pano = gr.Image(label="pano_L", type="numpy", interactive=False)
        comps = [inpainter, prompt, neg, seed, steps, guidance, tile_fov, feather, tile_res]
        outputs = comps + [only, gallery, pano]

        def _vals(*v):
            d = dict(zip(KEYS, v, strict=True))
            d["seed"] = int(d["seed"])
            d["steps"] = int(d["steps"])
            d["feather_px"] = int(d["feather_px"])
            d["tile_res"] = int(d["tile_res"])
            return d

        def run_all(job_id, *v):
            d = _vals(*v)
            d["only_tiles"] = []
            d["reuse_existing_tiles"] = False
            yield from C.run_pipeline(job_id, STAGE, None, {STAGE: d})

        def _df_rows(df) -> list[list]:
            """Dataframe (pandas.DataFrame | list[list] | None) 을 list[list] 로 정규화."""
            if df is None:
                return []
            if hasattr(df, "values"):  # pandas DataFrame
                return df.values.tolist()
            return list(df)

        def run_only(job_id, sel, tp_df, *v):
            if not sel:
                raise gr.Error("재생성할 타일을 고르세요")
            d = _vals(*v)
            d["only_tiles"] = list(sel)
            d["reuse_existing_tiles"] = True
            # Dataframe → dict[tile_name, prompt]. 빈 값은 skip (= 공통 프롬프트 사용).
            overrides: dict[str, str] = {}
            for row in _df_rows(tp_df):
                if len(row) < 2 or row[0] is None:
                    continue
                name = str(row[0]).strip()
                txt = str(row[1] or "").strip()
                if name and txt:
                    overrides[name] = txt
            d["tile_prompt_overrides"] = overrides
            yield from C.run_pipeline(job_id, STAGE, STAGE, {STAGE: d})

        def sync_tile_prompts(selected, current):
            """CheckboxGroup 선택이 바뀌면 Dataframe 행을 선택된 타일로만 재구성. 기존 입력은 보존."""
            existing: dict[str, str] = {}
            for row in _df_rows(current):
                if len(row) >= 2 and row[0] is not None and str(row[0]).strip():
                    existing[str(row[0])] = str(row[1] or "")
            return [[t, existing.get(t, "")] for t in (selected or [])]

        only.change(sync_tile_prompts, [only, tile_prompts], [tile_prompts])

        def do_upload(job_id, f):
            job = C.get_job(job_id)
            if job is None or f is None:
                raise gr.Error("Job 과 파일이 필요합니다")
            dst = job.path(3, "pano_L.png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = imageio.read_rgba(f)
            if img.shape[0] != img.shape[1]:
                raise gr.Error("pano_L 은 정사각이어야 합니다")
            shutil.copy2(f, dst)
            return f"덮어씀: {dst} — Stage 5 부터 재실행하면 반영됩니다", C.status_md(job_id)

        shared["all_run_buttons"].extend([run_btn, only_btn])

        def _wire():
            others_run = [b for b in shared["all_run_buttons"] if b is not run_btn]
            others_only = [b for b in shared["all_run_buttons"] if b is not only_btn]
            run_btn.click(
                C.with_button_lock(run_all, RUN_LABEL, n_other=len(others_run)),
                [shared["job_id"], *comps],
                [
                    run_btn,
                    *others_run,
                    shared["log"],
                    shared["log_full"],
                    shared["status"],
                    shared["progress"],
                    shared["stage_tabs"],
                ],
            ).then(load, [shared["job_id"]], outputs)
            only_btn.click(
                C.with_button_lock(run_only, ONLY_LABEL, n_other=len(others_only)),
                [shared["job_id"], only, tile_prompts, *comps],
                [
                    only_btn,
                    *others_only,
                    shared["log"],
                    shared["log_full"],
                    shared["status"],
                    shared["progress"],
                    shared["stage_tabs"],
                ],
            ).then(load, [shared["job_id"]], outputs)

        shared["after_build_wires"].append(_wire)
        upload_btn.click(do_upload, [shared["job_id"], upload], [shared["log"], shared["status"]]).then(
            load, [shared["job_id"]], outputs
        )
    return load, outputs
