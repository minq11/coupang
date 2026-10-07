"""Job: 입력 1장 + params.json + Stage 폴더들. 캐시 키, Stage 순서, from-stage 재실행.

Stage 모듈 규약 (pipeline/stage_XX_name.py):
    INDEX: int, NAME: str, DIRNAME: str ("03_outpaint"), PARAMS_KEY: str ("stage_03_outpaint")
    USES_MODELS: bool                      # params["models"] 가 캐시 키에 들어가는지
    inputs(job) -> list[Path]              # 캐시 키용 입력 파일 (이전 Stage 산출물)
    outputs(job) -> list[Path]             # 존재해야 '완료' 로 보는 파일
    run(job, params, ctx) -> dict          # meta.json 에 합쳐질 추가 정보 (model 등)
Stage 끼리 서로의 내부 함수를 import 하지 않는다. 공유 로직은 geometry/, util/, models/ 에 둔다.
"""

from __future__ import annotations

import importlib
import shutil
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from vr180 import paths
from vr180.util import imageio
from vr180.util.hashing import files_hash, params_hash
from vr180.util.log import RunContext

STAGE_MODULES = [
    "vr180.pipeline.stage_00_split",
    "vr180.pipeline.stage_01_fov",
    "vr180.pipeline.stage_02_reproj",
    "vr180.pipeline.stage_03_outpaint",
    "vr180.pipeline.stage_04_stereo",
    "vr180.pipeline.stage_05_depth",
    "vr180.pipeline.stage_06_right",
    "vr180.pipeline.stage_07_output",
]

_stage_cache: list[ModuleType] | None = None


def stages() -> list[ModuleType]:
    global _stage_cache
    if _stage_cache is None:
        _stage_cache = [importlib.import_module(m) for m in STAGE_MODULES]
        for i, m in enumerate(_stage_cache):
            assert m.INDEX == i, f"{m.__name__}: INDEX {m.INDEX} != {i}"
    return _stage_cache


def load_default_params(config_path: Path | None = None) -> dict[str, Any]:
    p = config_path or paths.default_config_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Job:
    def __init__(self, job_dir: Path):
        self.dir = Path(job_dir)
        self.id = self.dir.name
        self.params: dict[str, Any] = {}
        if self.params_path.exists():
            self.params = imageio.read_json(self.params_path)

    # ------------------------------------------------------------------ 생성 / 열기
    @classmethod
    def create(
        cls,
        input_image: Path | str,
        job_id: str | None = None,
        config_path: Path | None = None,
        params_override: dict | None = None,
    ) -> Job:
        input_image = Path(input_image)
        if not input_image.exists():
            raise FileNotFoundError(input_image)
        if job_id is None:
            job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + input_image.stem[:24] + "_" + uuid.uuid4().hex[:4]
        job_dir = paths.work_dir() / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        # 입력은 원본 확장자를 유지해 복사 (JPG 재압축 방지). input.* 하나만 둔다.
        dst = job_dir / ("input" + input_image.suffix.lower())
        shutil.copy2(input_image, dst)
        job = cls(job_dir)
        job.params = load_default_params(config_path)
        if params_override:
            _deep_update(job.params, params_override)
        job.params["_job"] = {
            "id": job_id,
            "source": str(input_image.resolve()),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        job.save_params()
        return job

    @classmethod
    def open(cls, job_id_or_dir: str | Path) -> Job:
        p = Path(job_id_or_dir)
        if not p.is_dir():
            p = paths.work_dir() / str(job_id_or_dir)
        if not (p / "params.json").exists():
            raise FileNotFoundError(f"Job 을 찾을 수 없음: {job_id_or_dir}")
        return cls(p)

    @staticmethod
    def list_jobs() -> list[str]:
        wd = paths.work_dir()
        return sorted([p.name for p in wd.iterdir() if (p / "params.json").exists()], reverse=True)

    # ------------------------------------------------------------------ 경로
    @property
    def params_path(self) -> Path:
        return self.dir / "params.json"

    @property
    def input_path(self) -> Path:
        for p in sorted(self.dir.glob("input.*")):
            return p
        raise FileNotFoundError(f"입력 이미지가 없음: {self.dir}")

    def stage_dir(self, index: int) -> Path:
        return self.dir / stages()[index].DIRNAME

    def path(self, index: int, name: str) -> Path:
        return self.stage_dir(index) / name

    # ------------------------------------------------------------------ 파라미터
    def save_params(self) -> None:
        imageio.write_json(self.params_path, self.params)

    def stage_params(self, index: int) -> dict[str, Any]:
        return self.params.setdefault(stages()[index].PARAMS_KEY, {})

    def set_param(self, index: int, key: str, value: Any) -> None:
        self.stage_params(index)[key] = value

    def model_params(self) -> dict[str, Any]:
        return self.params.setdefault("models", {})

    # ------------------------------------------------------------------ 캐시 / 상태
    def _cache_key(self, index: int) -> tuple[str, str]:
        st = stages()[index]
        ih = files_hash(st.inputs(self))
        p = dict(self.stage_params(index))
        if getattr(st, "USES_MODELS", False):
            p["__models__"] = self.model_params()
        return ih, params_hash(p)

    def meta(self, index: int) -> dict | None:
        mp = self.stage_dir(index) / "meta.json"
        return imageio.read_json(mp) if mp.exists() else None

    def outputs_exist(self, index: int) -> bool:
        return all(Path(p).exists() for p in stages()[index].outputs(self))

    def status(self, index: int) -> str:
        """'missing' | 'stale' | 'done'. stale = 산출물은 있지만 입력/파라미터가 바뀜."""
        if not self.outputs_exist(index):
            return "missing"
        m = self.meta(index)
        if m is None:
            return "stale"
        try:
            ih, ph = self._cache_key(index)
        except FileNotFoundError:
            return "stale"
        return "done" if (m.get("input_hash") == ih and m.get("params_hash") == ph) else "stale"

    def all_status(self) -> list[str]:
        return [self.status(i) for i in range(len(stages()))]

    # ------------------------------------------------------------------ 실행
    def run_stage(self, index: int, ctx: RunContext | None = None, force: bool = False) -> str:
        """반환: 'cached' | 'ran'."""
        ctx = ctx or RunContext()
        st = stages()[index]
        sd = self.stage_dir(index)
        sd.mkdir(parents=True, exist_ok=True)
        for p in st.inputs(self):
            if not Path(p).exists():
                raise FileNotFoundError(f"Stage {index} 입력이 없음: {p} (이전 Stage 를 먼저 실행)")
        ih, ph = self._cache_key(index)
        if not force and self.status(index) == "done":
            ctx.log(f"Stage {index} {st.NAME}: 캐시 사용")
            return "cached"
        ctx.log(f"Stage {index} {st.NAME}: 실행")
        t0 = time.time()
        extra = st.run(self, dict(self.stage_params(index)), ctx) or {}
        elapsed = time.time() - t0
        meta = {
            "stage": index,
            "name": st.NAME,
            "input_hash": ih,
            "params_hash": ph,
            "params": self.stage_params(index),
            "elapsed_s": round(elapsed, 3),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta.update(extra)
        imageio.write_json(sd / "meta.json", meta)
        ctx.log(f"Stage {index} {st.NAME}: 완료 ({elapsed:.1f}s)")
        return "ran"

    def run(
        self,
        from_stage: int = 0,
        to_stage: int | None = None,
        ctx: RunContext | None = None,
        force_from: bool = True,
    ) -> list[str]:
        """from_stage 부터 to_stage 까지. from_stage 는 강제 재실행, 그 뒤는 캐시가 유효하면 건너뛴다.

        from_stage 이전 Stage 는 산출물이 없을 때만 실행한다 (있으면 stale 이어도 그대로 쓴다 →
        사용자가 외부 툴로 고친 파일을 보존하기 위함).
        """
        ctx = ctx or RunContext()
        n = len(stages())
        to_stage = n - 1 if to_stage is None else to_stage
        results: list[str] = []
        for i in range(0, from_stage):
            if not self.outputs_exist(i):
                results.append(self.run_stage(i, ctx))
            else:
                results.append("kept")
        for i in range(from_stage, to_stage + 1):
            results.append(self.run_stage(i, ctx, force=(force_from and i == from_stage)))
        self.save_params()
        return results
