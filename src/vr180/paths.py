"""저장소 루트, 작업 폴더, 모델 폴더 경로. .env 로 바꿀 수 있다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


def work_dir() -> Path:
    p = _env_path("VR180_WORK_DIR", REPO_ROOT / "work")
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = _env_path("VR180_MODELS_DIR", REPO_ROOT / "models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def third_party_dir() -> Path:
    p = REPO_ROOT / "third_party"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"


def ensure_hf_env() -> None:
    """HF 캐시를 models/ 아래로 돌린다 (HF_HOME 이 .env 에 없을 때)."""
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(models_dir() / "hf")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
