"""모델 가중치 내려받기. 저장소에는 가중치를 넣지 않는다."""

from __future__ import annotations

import io
import subprocess
import urllib.request
import zipfile

from vr180 import paths
from vr180.models.stereo_raft import RAFT_STEREO_WEIGHTS, _raft_stereo_dir
from vr180.pipeline.job import load_default_params

LAMA_HINT = "https://github.com/enesmsahin/simple-lama-inpainting/releases 의 big-lama.pt 를 models/lama/ 에 넣어주세요."


def _snapshot(repo: str, **kw) -> None:
    from huggingface_hub import snapshot_download

    print(f"[hf] {repo}")
    snapshot_download(repo, **kw)


def _fetch_vda(cfg: dict, sizes: tuple[str, ...] = ("small", "large")) -> int:
    """Video-Depth-Anything 저장소 clone + 가중치(small/large) 다운로드. 재실행 안전."""
    rc = 0
    repo_dir = paths.third_party_dir() / "Video-Depth-Anything"
    if not (repo_dir / "video_depth_anything").exists():
        print(f"[git] {cfg['vda_repo']} → {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", cfg["vda_repo"], str(repo_dir)], check=True)
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
    for size in sizes:
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


def fetch_models(
    sdxl=False,
    da2=False,
    raft_stereo=False,
    raft_flow=False,
    vda=False,
    lama=False,
    person=False,
) -> int:
    paths.ensure_hf_env()
    cfg = load_default_params()["models"]
    rc = 0
    if lama:
        from vr180.models.inpaint_lama import download_lama

        try:
            print(f"[lama] {download_lama()}")
        except Exception as e:  # noqa: BLE001
            print(f"LaMa 가중치 다운로드 실패 ({e}). {LAMA_HINT}")
            rc = 1
    if da2:
        _snapshot(cfg["da2_large_id"])
    if sdxl:
        # SDXL inpainting 본체 + 가속 LoRA 두 개 (inpainter=sdxl_lightning / sdxl_lcm). 첫 실행 때 자동으로도 받는다.
        from huggingface_hub import hf_hub_download

        _snapshot(
            cfg["sdxl_inpaint_id"],
            allow_patterns=["*.json", "*.txt", "*fp16*", "tokenizer*/*", "scheduler/*"],
        )
        _snapshot(cfg["sdxl_lcm_lora_id"], allow_patterns=["*.safetensors", "*.json"])
        print(f"[hf] {cfg['sdxl_lightning_repo']}/{cfg['sdxl_lightning_file']}")
        hf_hub_download(cfg["sdxl_lightning_repo"], cfg["sdxl_lightning_file"])
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
    if person:
        rc |= _fetch_person(cfg)
    print("완료")
    return rc


def _download(url: str, dst, log=print) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"[dl] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)


def _fetch_person(cfg: dict) -> int:
    """인물 AR: SAM 2 코드(고정 커밋) + SAM 2.1 small 가중치, BiRefNet-matting, DA2 Small, VDA Small, RF-DETR-Seg.
    모두 상업 이용 가능한 라이선스 (Apache-2.0 / MIT). 재실행 안전."""
    from vr180.person.models import (
        SAM2_CKPT_URL,
        SAM2_COMMIT,
        SAM2_REPO,
        SAM2_SIZES,
        sam2_ckpt_path,
        sam2_repo_dir,
    )

    rc = 0
    repo = sam2_repo_dir()
    if not (repo / "sam2" / "build_sam.py").exists():
        print(f"[git] {SAM2_REPO}@{SAM2_COMMIT[:10]} → {repo}")
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", SAM2_REPO, SAM2_COMMIT], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "FETCH_HEAD"], check=True)
    else:
        print(f"[skip] SAM 2 코드 이미 있음: {repo}")
    size = str(cfg.get("person_sam2_size", "small"))
    ck = sam2_ckpt_path(size)
    if not ck.exists():
        try:
            _download(SAM2_CKPT_URL.format(file=SAM2_SIZES[size][1]), ck)
        except Exception as e:  # noqa: BLE001
            print(f"SAM 2.1 가중치 다운로드 실패 ({e}). {SAM2_CKPT_URL.format(file=SAM2_SIZES[size][1])} 를 받아 {ck} 에 두세요.")
            rc = 1
    for key, rev_key in (("birefnet_matting_id", "birefnet_matting_rev"), ("da2_small_id", None)):
        try:
            _snapshot(cfg[key], revision=cfg.get(rev_key) if rev_key else None)
        except Exception as e:  # noqa: BLE001
            print(f"{cfg[key]} 다운로드 실패 ({e})")
            rc = 1
    rc |= _fetch_vda(cfg, sizes=("small",))
    try:  # RF-DETR 는 처음 만들 때 가중치를 스스로 받는다
        import rfdetr

        print("[rfdetr] rf-detr-seg-medium")
        rfdetr.RFDETRSegMedium()
    except Exception as e:  # noqa: BLE001
        print(f"RF-DETR 준비 실패 ({e}). `uv sync` 후 다시 시도하세요.")
        rc = 1
    return rc
