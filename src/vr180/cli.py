"""CLI: python -m vr180 <command>. GUI 는 같은 함수를 부른다.

vr180 run --input x.png [--job ID] [--from-stage N] [--to-stage N] [--set stage_03_outpaint.inpainter=sdxl ...]
vr180 status --job ID
vr180 jobs
vr180 gui [--port 7860]
vr180 make-sample samples/synthetic_sbs.png
vr180 fetch-models [--sdxl] [--sdxl-fast] [--lama] [--da2] [--flux] [--raft-stereo] [--vda] [--all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from vr180 import __version__


def _parse_set(items: list[str]) -> dict:
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set 형식: section.key=value  (받은 값: {it})")
        k, v = it.split("=", 1)
        parts = k.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    return out


def cmd_run(a) -> int:
    from vr180.pipeline.job import Job, stages
    from vr180.util.log import RunContext

    override = _parse_set(a.set)
    if a.job:
        job = Job.open(a.job)
        from vr180.pipeline.job import _deep_update

        _deep_update(job.params, override)
        job.save_params()
    else:
        if not a.input:
            raise SystemExit("--input 또는 --job 이 필요합니다")
        job = Job.create(
            a.input,
            job_id=a.job_id,
            config_path=Path(a.config) if a.config else None,
            params_override=override,
            input_mode=a.mode,
        )
        print(f"Job 생성: {job.id} → {job.dir}  mode={job.input_mode}")
    ctx = RunContext()
    to = a.to_stage if a.to_stage is not None else len(stages()) - 1
    res = job.run(from_stage=a.from_stage, to_stage=to, ctx=ctx, force_from=not a.no_force)
    for i, r in enumerate(res):
        print(f"  stage {i} {stages()[i].NAME:9s} {r}")
    print(f"완료: {job.dir}")
    return 0


def cmd_status(a) -> int:
    from vr180.pipeline.job import Job, stages

    job = Job.open(a.job)
    print(f"Job {job.id}  입력 {job.input_path.name}")
    for i, st in enumerate(stages()):
        m = job.meta(i) or {}
        print(f"  {i} {st.NAME:9s} {job.status(i):8s} {m.get('elapsed_s', ''):>8} {m.get('finished_at', '')}")
    return 0


def cmd_jobs(a) -> int:
    from vr180.pipeline.job import Job

    for j in Job.list_jobs():
        print(j)
    return 0


def cmd_gui(a) -> int:
    from vr180.gui.app import launch

    launch(port=a.port, share=a.share)
    return 0


def cmd_make_sample(a) -> int:
    from vr180.sample import make_mono, make_sbs
    from vr180.util import imageio

    if a.mono:
        img = make_mono(a.width, a.height, a.hfov, a.seed)
        label = "mono"
    else:
        img = make_sbs(a.width, a.height, a.hfov, a.baseline, a.seed)
        label = "sbs"
    imageio.write_image(a.output, img)
    print(f"저장: {a.output} ({img.shape[1]}x{img.shape[0]}, {label}, hfov {a.hfov}°)")
    return 0


def cmd_fetch_models(a) -> int:
    from vr180.fetch import fetch_models

    return fetch_models(
        sdxl=a.sdxl or a.all,
        da2=a.da2 or a.all,
        flux=a.flux or a.all,
        raft_stereo=a.raft_stereo or a.all,
        raft_flow=a.raft_flow or a.all,
        vda=a.vda or a.all,
        lama=a.lama or a.all,
        sdxl_fast=a.sdxl_fast or a.all,
    )


def cmd_video(a) -> int:
    from vr180.util.log import RunContext
    from vr180.video.run import run_video

    override = _parse_set(a.set) if a.set else None
    ctx = RunContext()
    res = run_video(
        input_video=a.input,
        output_video=a.output,
        ref_frame=a.ref_frame,
        ref_time=a.ref_time,
        surround_job=a.surround_job,
        input_mode=a.mode,
        codec=a.codec,
        crf=a.crf,
        fps=a.fps,
        check_motion=not a.no_motion_check,
        motion_threshold_px=a.motion_threshold,
        detect_cuts=a.detect_cuts,
        depth_model=a.depth_model,
        vda_chunk=a.vda_chunk,
        vda_overlap=a.vda_overlap,
        params_override=override,
        ctx=ctx,
    )
    print(f"완료: {res['output']}  surround_job={res['surround_job']}  frames={res['frames']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vr180", description="3D SBS → VR180 SBS 변환 파이프라인")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="파이프라인 실행")
    r.add_argument("--input", "-i", help="입력 이미지 (mono = 2D 사진, sbs = 스테레오)")
    r.add_argument("--job", "-j", help="기존 Job id (또는 폴더)")
    r.add_argument("--job-id", help="새 Job 의 id 지정")
    r.add_argument("--config", "-c", help="기본 파라미터 yaml (기본 configs/default.yaml)")
    r.add_argument(
        "--mode",
        choices=["mono", "sbs"],
        default=None,
        help="입력 모드. 생략하면 config/_job.input_mode 기본값 (mono).",
    )
    r.add_argument("--from-stage", type=int, default=0)
    r.add_argument("--to-stage", type=int, default=None)
    r.add_argument("--no-force", action="store_true", help="from-stage 도 캐시가 유효하면 건너뜀")
    r.add_argument("--set", "-s", action="append", default=[], help="파라미터 덮어쓰기: section.key=value")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="Job 의 Stage 상태")
    s.add_argument("--job", "-j", required=True)
    s.set_defaults(fn=cmd_status)

    j = sub.add_parser("jobs", help="Job 목록")
    j.set_defaults(fn=cmd_jobs)

    g = sub.add_parser("gui", help="Gradio GUI")
    g.add_argument("--port", type=int, default=None)
    g.add_argument("--share", action="store_true")
    g.set_defaults(fn=cmd_gui)

    m = sub.add_parser("make-sample", help="합성 SBS/mono 테스트 이미지 생성")
    m.add_argument("output")
    m.add_argument("--width", type=int, default=1920, help="(sbs) 눈당 폭 / (mono) 전체 폭")
    m.add_argument("--height", type=int, default=1080)
    m.add_argument("--hfov", type=float, default=80.0)
    m.add_argument("--baseline", type=float, default=0.065, help="(sbs 전용) 눈 간격 m")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--mono", action="store_true", help="SBS 대신 가운데 카메라 1장만 생성")
    m.set_defaults(fn=cmd_make_sample)

    f = sub.add_parser("fetch-models", help="모델 가중치 내려받기 (models/ 아래)")
    f.add_argument("--sdxl", action="store_true")
    f.add_argument("--da2", action="store_true")
    f.add_argument("--flux", action="store_true")
    f.add_argument("--raft-stereo", action="store_true")
    f.add_argument("--raft-flow", action="store_true")
    f.add_argument(
        "--vda", action="store_true", help="Video Depth Anything 저장소+가중치 (2D 영상 → mono 경로)"
    )
    f.add_argument(
        "--lama",
        action="store_true",
        help="LaMa big-lama.pt 가중치 (inpainter=lama). 첫 실행 때 자동으로도 받는다",
    )
    f.add_argument(
        "--sdxl-fast", action="store_true", help="SDXL 가속 LoRA 두 개 (inpainter=sdxl_lcm / sdxl_lightning)"
    )
    f.add_argument("--all", action="store_true")
    f.set_defaults(fn=cmd_fetch_models)

    v = sub.add_parser(
        "video",
        help="영상을 VR180 SBS 로 변환 (고정 샷 전용). 기준 프레임의 주변부를 1회 만들어 모든 프레임 재사용.",
    )
    v.add_argument(
        "--input", "-i", required=True, help="입력 영상 (sbs 모드: iw3 등으로 미리 SBS 변환된 영상)"
    )
    v.add_argument("--output", "-o", required=True, help="출력 .mp4 경로")
    v.add_argument(
        "--mode",
        choices=["sbs", "mono"],
        default="sbs",
        help="영상 입력 모드. 현재는 sbs 만 (mono 는 #3 VDA 에서).",
    )
    v.add_argument("--ref-frame", type=int, default=None, help="기준 프레임 번호 (기본 0)")
    v.add_argument("--ref-time", type=float, default=None, help="기준 시각(초). ref-frame 과 둘 중 하나만.")
    v.add_argument(
        "--surround-job", default=None, help="이미 만든 Job id 를 재사용 (기준 프레임 재생성 생략)"
    )
    v.add_argument("--codec", default="libx265")
    v.add_argument("--crf", type=int, default=20)
    v.add_argument("--fps", type=float, default=None, help="출력 fps. 생략하면 입력 fps 그대로.")
    v.add_argument("--no-motion-check", action="store_true", help="phaseCorrelate 샷 안정성 확인 끄기")
    v.add_argument("--motion-threshold", type=float, default=2.0, help="평균 이동(px) 임계값")
    v.add_argument("--detect-cuts", action="store_true", help="히스토그램 급변으로 컷 감지(비싸므로 선택)")
    v.add_argument(
        "--depth-model",
        default="da2_video",
        choices=["none", "vda_large", "vda_small", "da2_video"],
        help="(mono 전용) 영상 깊이 모델. VDA 미설치 시 자동으로 DA2 프레임별 폴백.",
    )
    v.add_argument("--vda-chunk", type=int, default=32, help="(mono) VDA chunk 크기")
    v.add_argument("--vda-overlap", type=int, default=8, help="(mono) VDA overlap")
    v.add_argument("--set", "-s", action="append", default=[], help="파라미터 덮어쓰기: section.key=value")
    v.set_defaults(fn=cmd_video)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.fn(args))


def gui_main() -> None:
    main(["gui"])
