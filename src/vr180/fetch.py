"""모델 가중치 내려받기. 저장소에는 가중치를 넣지 않는다."""

from __future__ import annotations

import io
import subprocess
import urllib.request
import zipfile

from vr180 import paths
from vr180.models.stereo_raft import RAFT_STEREO_WEIGHTS, _raft_stereo_dir
from vr180.pipeline.job import load_default_params


def _snapshot(repo: str, **kw) -> None:
    from huggingface_hub import snapshot_download

    print(f"[hf] {repo}")
    snapshot_download(repo, **kw)


def _fetch_vda(cfg: dict) -> int:
    """Video-Depth-Anything 저장소 clone + 가중치(small/large) 다운로드. 재실행 안전."""
    rc = 0
    repo_dir = paths.third_party_dir() / "Video-Depth-Anything"
    if not (repo_dir / "video_depth_anything").exists():
        print(f"[git] {cfg['vda_repo']} → {repo_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", cfg["vda_repo"], str(repo_dir)], check=True
        )
    else:
        print(f"[skip] VDA 저장소 이미 있음: {repo_dir}")
    # 가중치 — 두 사이즈 모두 (사용 안 할 수도 있지만 선택을 위해 받아둠)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub 가 필요합니다 (uv sync).")
        return 1
    wdir = paths.models_dir() / "vda"
    wdir.mkdir(parents=True, exist_ok=True)
    for size in ("small", "large"):
        hf_repo = cfg[f"vda_{size}_hf"]
        fname = cfg[f"vda_{size}_file"]
        target = wdir / fname
        if target.exists():
            print(f"[skip] {fname} 이미 있음")
            continue
        try:
            print(f"[hf] {hf_repo}/{fname}")
            path = hf_hub_download(hf_repo, fname, local_dir=str(wdir))
            # local_dir 사용 시 HF 가 그 안에 바로 저장함
            if not target.exists():  # 혹시 다른 경로에 들어갔으면 복사
                import shutil as _sh
                _sh.copy2(path, target)
        except Exception as e:  # noqa: BLE001
            print(f"VDA {size} 가중치 다운로드 실패 ({e}). HF 토큰·네트워크 확인 후 재시도.")
            rc = 1
    return rc


def fetch_models(sdxl=False, da2=False, flux=False, raft_stereo=False, raft_flow=False, vda=False) -> int:
    paths.ensure_hf_env()
    cfg = load_default_params()["models"]
    rc = 0
    if da2:
        _snapshot(cfg["da2_large_id"])
    if sdxl:
        _snapshot(
            cfg["sdxl_inpaint_id"],
            allow_patterns=["*.json", "*.txt", "*fp16*", "tokenizer*/*", "scheduler/*"],
        )
    if flux:
        from huggingface_hub import hf_hub_download

        print(f"[hf] {cfg['flux_fill_gguf_repo']}/{cfg['flux_fill_gguf_file']}")
        hf_hub_download(cfg["flux_fill_gguf_repo"], cfg["flux_fill_gguf_file"])
        # text encoder / VAE / 설정은 base repo (게이트: HF_TOKEN 필요)
        _snapshot(
            cfg["flux_fill_id"],
            allow_patterns=[
                "*.json",
                "text_encoder/*",
                "text_encoder_2/*",
                "tokenizer*/*",
                "vae/*",
                "scheduler/*",
            ],
        )
    if raft_stereo:
        repo = _raft_stereo_dir()
        if not (repo / "core").exists():
            print(f"[git] {cfg['raft_stereo_repo']} → {repo}")
            subprocess.run(["git", "clone", "--depth", "1", cfg["raft_stereo_repo"], str(repo)], check=True)
        wdir = paths.models_dir() / "raft_stereo"
        wdir.mkdir(parents=True, exist_ok=True)
        if not (wdir / "raftstereo-middlebury.pth").exists():
            url = RAFT_STEREO_WEIGHTS["middlebury"]
            print(f"[dl] {url}")
            try:
                # Dropbox 는 User-Agent 가 없으면 HTML 랜딩 페이지만 돌려줌.
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=300).read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for n in z.namelist():
                        if n.endswith(".pth") and not n.startswith("__MACOSX/"):
                            (wdir / n.split("/")[-1]).write_bytes(z.read(n))
                            print("   ", n)
            except Exception as e:
                print(
                    f"RAFT-Stereo 가중치 자동 다운로드 실패 ({e}). README 의 models.zip 을 받아 {wdir} 에 풀어주세요."
                )
                rc = 1
    if raft_flow:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        torch.hub.set_dir(str(paths.models_dir() / "torch_hub"))
        raft_large(weights=Raft_Large_Weights.DEFAULT)
    if vda:
        rc |= _fetch_vda(cfg)
    print("완료")
    return rc
