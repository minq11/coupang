"""CLI: python -m vr180 <command>. GUI 는 같은 함수를 부른다.

vr180 run --input x.png [--job ID] [--from-stage N] [--to-stage N] [--set stage_03_outpaint.inpainter=sdxl ...]
vr180 status --job ID
vr180 jobs
vr180 gui [--port 7860]
vr180 make-sample samples/synthetic_sbs.png
vr180 fetch-models [--sdxl] [--da2] [--flux] [--raft-stereo] [--all]
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
        )
        print(f"Job 생성: {job.id} → {job.dir}")
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
    from vr180.sample import make_sbs
    from vr180.util import imageio

    img = make_sbs(a.width, a.height, a.hfov, a.baseline, a.seed)
    imageio.write_image(a.output, img)
    print(f"저장: {a.output} ({img.shape[1]}x{img.shape[0]}, hfov {a.hfov}°)")
    return 0


def cmd_fetch_models(a) -> int:
    from vr180.fetch import fetch_models

    return fetch_models(
        sdxl=a.sdxl or a.all,
        da2=a.da2 or a.all,
        flux=a.flux or a.all,
        raft_stereo=a.raft_stereo or a.all,
        raft_flow=a.raft_flow or a.all,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vr180", description="3D SBS → VR180 SBS 변환 파이프라인")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="파이프라인 실행")
    r.add_argument("--input", "-i", help="입력 SBS 이미지")
    r.add_argument("--job", "-j", help="기존 Job id (또는 폴더)")
    r.add_argument("--job-id", help="새 Job 의 id 지정")
    r.add_argument("--config", "-c", help="기본 파라미터 yaml (기본 configs/default.yaml)")
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

    m = sub.add_parser("make-sample", help="합성 SBS 테스트 이미지 생성")
    m.add_argument("output")
    m.add_argument("--width", type=int, default=1920, help="눈당 폭")
    m.add_argument("--height", type=int, default=1080)
    m.add_argument("--hfov", type=float, default=80.0)
    m.add_argument("--baseline", type=float, default=0.065)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(fn=cmd_make_sample)

    f = sub.add_parser("fetch-models", help="모델 가중치 내려받기 (models/ 아래)")
    f.add_argument("--sdxl", action="store_true")
    f.add_argument("--da2", action="store_true")
    f.add_argument("--flux", action="store_true")
    f.add_argument("--raft-stereo", action="store_true")
    f.add_argument("--raft-flow", action="store_true")
    f.add_argument("--all", action="store_true")
    f.set_defaults(fn=cmd_fetch_models)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.fn(args))


def gui_main() -> None:
    main(["gui"])
