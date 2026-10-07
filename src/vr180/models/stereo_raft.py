"""RAFT 계열 stereo matcher.

RaftStereoMatcher: princeton-vl/RAFT-Stereo 저장소 (pip 패키지가 아님) 를 third_party/ 에 받아 쓴다.
    `python -m vr180 fetch-models --raft-stereo` 로 코드와 middlebury 가중치를 받는다.
RaftFlowMatcher: torchvision 의 RAFT optical flow (pip 만으로 됨). 정류된 스테레오에서는
    수평 flow 의 음수가 disparity 다. RAFT-Stereo 를 못 받았을 때의 대안.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vr180.models.base import ModelInfo, StereoMatcher, resolve_device

RAFT_STEREO_WEIGHTS = {
    "middlebury": "https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip?dl=1",  # 공식 README 의 models.zip (raftstereo-middlebury.pth 포함)
}


def _raft_stereo_dir() -> Path:
    from vr180 import paths

    return paths.third_party_dir() / "RAFT-Stereo"


def _pad8(x):
    import torch

    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate"), (h, w)


class RaftStereoMatcher(StereoMatcher):
    def __init__(self, model_cfg: dict, iters: int = 32, weights: str = "middlebury"):
        super().__init__(model_cfg)
        self.iters = int(iters)
        self.weights = weights
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None

    def _weights_path(self) -> Path:
        from vr180 import paths

        return paths.models_dir() / "raft_stereo" / f"raftstereo-{self.weights}.pth"

    def _load(self) -> None:
        import argparse

        import torch

        repo = _raft_stereo_dir()
        core = repo / "core"
        if not core.exists():
            raise FileNotFoundError(
                f"RAFT-Stereo 코드가 없음: {repo}. `python -m vr180 fetch-models --raft-stereo` 실행"
            )
        wp = self._weights_path()
        if not wp.exists():
            raise FileNotFoundError(
                f"RAFT-Stereo 가중치가 없음: {wp}. `python -m vr180 fetch-models --raft-stereo` 실행"
            )
        if str(core) not in sys.path:
            sys.path.insert(0, str(core))
        from raft_stereo import RAFTStereo  # type: ignore

        # middlebury 가중치는 공식 README 의 설정과 같아야 한다
        args = argparse.Namespace(
            hidden_dims=[128] * 3,
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=4,
            corr_radius=4,
            n_downsample=2,
            context_norm="batch",
            slow_fast_gru=True,
            n_gru_layers=3,
            mixed_precision=True,
        )
        if self.weights == "middlebury":
            args.corr_implementation = "alt"
        model = torch.nn.DataParallel(RAFTStereo(args))
        state = torch.load(str(wp), map_location="cpu")
        model.load_state_dict(state)
        self.model = model.module.to(self.device).eval()

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="raft_stereo",
            weights=str(self._weights_path()),
            device=self.device,
            extra={"iters": self.iters},
        )

    def disparity(self, left, right, max_disparity: int = 256):
        import torch

        self.load()

        def to_t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()[None].to(self.device)

        l_t, (h, w) = _pad8(to_t(left))
        r_t, _ = _pad8(to_t(right))
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device == "cuda"):
            _, flow_up = self.model(l_t, r_t, iters=self.iters, test_mode=True)
        disp = (-flow_up[0, 0, :h, :w]).float().cpu().numpy().astype(np.float32)
        valid = np.isfinite(disp) & (disp > 0) & (disp < max_disparity * 1.5)
        disp = np.clip(disp, 0, None)
        return disp, valid


class RaftFlowMatcher(StereoMatcher):
    """torchvision RAFT (optical flow) 를 스테레오에 적용. 수평 flow 의 음수 = disparity."""

    def __init__(self, model_cfg: dict, iters: int = 24, **_):
        super().__init__(model_cfg)
        self.iters = int(iters)
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.model = None

    def _load(self) -> None:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        from vr180 import paths

        torch.hub.set_dir(str(paths.models_dir() / "torch_hub"))
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(self.device).eval()

    def _unload(self) -> None:
        self.model = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="torchvision_raft_large_flow", weights="Raft_Large_Weights.DEFAULT", device=self.device
        )

    def disparity(self, left, right, max_disparity: int = 256):
        import torch

        self.load()

        def to_t(a):
            return (
                torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()[None].to(self.device)
                / 127.5
                - 1.0
            )

        l_t, (h, w) = _pad8(to_t(left))
        r_t, _ = _pad8(to_t(right))
        with torch.inference_mode():
            flows = self.model(l_t, r_t, num_flow_updates=self.iters)
        flow = flows[-1][0]
        disp = (-flow[0, :h, :w]).float().cpu().numpy().astype(np.float32)
        dy = flow[1, :h, :w].abs().float().cpu().numpy()
        valid = (disp > 0) & (disp < max_disparity * 1.5) & (dy < 3.0)
        return np.clip(disp, 0, None), valid
