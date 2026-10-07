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


def fetch_models(sdxl=False, da2=False, flux=False, raft_stereo=False, raft_flow=False) -> int:
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
                data = urllib.request.urlopen(url, timeout=120).read()
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for n in z.namelist():
                        if n.endswith(".pth"):
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
    print("완료")
    return rc
