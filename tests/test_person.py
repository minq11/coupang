"""인물 AR — 모델 대신 '색으로 사람을 찾는' 가짜 모델로 전체 흐름을 검증한다 (GPU·가중치 없이)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from vr180.person import matte, stereo
from vr180.person import track as T
from vr180.person.models import Detection, box_from_mask
from vr180.person.pipeline import PersonModels, run_person
from vr180.person.sample import make_person_image, make_person_video
from vr180.util import imageio
from vr180.util.log import RunContext


def _ctx():
    return RunContext(log_fn=lambda s: None)


def _people_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] > 120) & (hsv[..., 2] > 60)


class FakeDetector:
    """채도 높은 덩어리 = 사람. extra[frame_call] 로 오검출을 한 번 끼워 넣을 수 있다."""

    def __init__(self, false_positive_on_call: int | None = None):
        self.calls = 0
        self.fp_call = false_positive_on_call

    def load(self):
        pass

    def unload(self):
        pass

    def detect(self, rgb):
        self.calls += 1
        n, labels = cv2.connectedComponents(_people_mask(rgb).astype(np.uint8))
        out = [
            Detection(mask=labels == k, score=0.7, box=box_from_mask(labels == k))
            for k in range(1, n)
            if (labels == k).sum() > 30
        ]
        if self.fp_call is not None and self.calls == self.fp_call:
            m = np.zeros(rgb.shape[:2], bool)
            m[5:40, 5:30] = True
            out.append(Detection(mask=m, score=0.5, box=box_from_mask(m)))
        return out


class FakeSam2:
    """SAM2VideoPredictor 흉내: 프롬프트 마스크의 평균 색과 가장 가까운 색 덩어리를 매 프레임 '따라간다'."""

    def init_state(self, video_path, **kw):
        files = sorted(Path(video_path).glob("*.jpg"), key=lambda p: int(p.stem))
        return {"files": files, "objs": {}}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        rgb = imageio.read_rgb(state["files"][frame_idx])
        state["objs"][obj_id] = (frame_idx, rgb[mask].mean(axis=0))

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
        n = len(state["files"])
        if start_frame_idx is None:
            start_frame_idx = min(f for f, _ in state["objs"].values())
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, n)
        ids = sorted(state["objs"])
        for f in order:
            rgb = imageio.read_rgb(state["files"][f])
            k, labels = cv2.connectedComponents(_people_mask(rgb).astype(np.uint8))
            comps = [(labels == j, rgb[labels == j].mean(axis=0)) for j in range(1, k)]
            logits = np.full((len(ids), 1) + rgb.shape[:2], -5.0, np.float32)
            for oi, oid in enumerate(ids):
                col = state["objs"][oid][1]
                best = min(comps, key=lambda c: np.abs(c[1] - col).sum(), default=None)
                if best is not None and np.abs(best[1] - col).sum() < 120:
                    logits[oi, 0][best[0]] = 5.0
            yield f, ids, logits

    def reset_state(self, state):
        state.clear()


class FakeMatter:
    def load(self):
        pass

    def unload(self):
        pass

    def alpha(self, rgb):
        return cv2.GaussianBlur(_people_mask(rgb).astype(np.float32), (0, 0), 1.5)


class FakeDepth:
    """사람 = 가깝다(1), 배경 = 멀다(0.1). 사진/영상 인터페이스 둘 다."""

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def inverse_depth(self, rgb):
        return (
            np.where(_people_mask(rgb), 1.0, 0.1).astype(np.float32)
            + np.linspace(0, 0.2, rgb.shape[1])[None, :]
        )

    def inverse_depth_sequence(self, frames):
        return [self.inverse_depth(f) for f in frames]


def _det(mask, score=0.9):
    return Detection(mask=mask, score=score, box=box_from_mask(mask))


# ----------------------------------------------------------------------------- 키프레임 연결 규칙


def test_link_keyframes_rules():
    h, w = 100, 200

    def rect(x0, x1, y0=20, y1=90):
        m = np.zeros((h, w), bool)
        m[y0:y1, x0:x1] = True
        return m

    a = rect(20, 50)
    b = rect(120, 150)
    fp = rect(170, 190, 5, 30)
    tiny = rect(5, 7, 5, 7)
    kfs = [
        (0, [_det(a), _det(tiny)]),
        (15, [_det(a), _det(fp, 0.5)]),  # 오검출은 한 번만 → 등록 안 됨
        (30, [_det(a), _det(b)]),  # B 처음 등장
        (45, [_det(b)]),  # A 사라짐
        (60, [_det(b)]),
        (75, [_det(b)]),
    ]
    tracks = T.link_keyframes(kfs, n_frames=90, interval=15, frame_area=h * w, max_miss=3)
    firsts = sorted(t.first_frame for t in tracks)
    assert firsts == [0, 30]  # A, B 만. 오검출·너무 작은 것 제외
    ta = next(t for t in tracks if t.first_frame == 0)
    assert ta.end_frame == 45  # 마지막으로 잡힌 30 + 15 프레임까지
    tb = next(t for t in tracks if t.first_frame == 30)
    assert tb.end_frame is None


def test_link_keyframes_single_keyframe_and_cap():
    h, w = 50, 50
    ms = []
    for i in range(5):
        m = np.zeros((h, w), bool)
        m[10:40, i * 10 : i * 10 + 6] = True
        ms.append(_det(m))
    tracks = T.link_keyframes([(0, ms)], n_frames=1, interval=15, frame_area=h * w, max_people=3)
    assert len(tracks) == 3  # 키프레임 하나면 확인 규칙 없이 등록, 인원 상한


# ----------------------------------------------------------------------------- SAM 2 양방향 전파


def test_track_sam2_reverse_fills_person_entering_between_keyframes(tmp_path):
    frames, gts = make_person_video(w=320, h=180, n=60, enter_frame=20)
    d = tmp_path / "frames"
    d.mkdir()
    for i, f in enumerate(frames):
        imageio.write_image(d / f"{i:05d}.jpg", f, jpg_quality=98)
    det = FakeDetector()
    kf = T.detect_keyframes(det, lambda i: frames[i], 60, 15)
    tracks = T.link_keyframes(kf, n_frames=60, interval=15, frame_area=320 * 180)
    assert len(tracks) == 2
    tb = max(tracks, key=lambda t: t.first_frame)
    assert tb.first_frame == 30  # 20 에 들어왔지만 키프레임 30 에서 처음 검출
    res = T.track_sam2(FakeSam2(), str(d), tracks, n_frames=60, height=180, width=320, interval=15)
    # 뒤 방향 전파 덕분에 키프레임 30 이전(들어오던 22~29)에도 B 가 잡혀 있어야 한다
    for f in (23, 26, 29, 40):
        gt_b = gts[f][1]
        if gt_b.sum() < 20:
            continue
        got = res.mask(f)
        assert (got & gt_b).sum() / gt_b.sum() > 0.8, f
    # A 는 처음부터 끝까지
    assert (res.mask(5) & gts[5][0]).sum() / gts[5][0].sum() > 0.8
    assert 0.6 < res.tallest_frac < 0.85


# ----------------------------------------------------------------------------- 경계 3구역


def test_refine_alpha_zones():
    h, w = 80, 80
    rgb = np.zeros((h, w, 3), np.uint8)
    m = np.zeros((h, w), bool)
    m[20:60, 20:60] = True

    class Half:
        def alpha(self, crop):
            return np.full(crop.shape[:2], 0.5, np.float32)

    a = matte.refine_alpha(rgb, m, Half(), core_px=4, band_px=4)
    assert a[40, 40] == 1.0  # 안쪽: 불투명
    assert a[5, 5] == 0.0  # 바깥: 투명
    assert a[20, 40] == pytest.approx(0.5)  # 경계 띠: 매팅 값
    assert a[17, 40] == pytest.approx(0.5)  # 바깥쪽 띠도
    soft = matte.refine_alpha(rgb, m, None, core_px=4, band_px=4)
    assert soft[40, 40] == 1.0 and soft[5, 5] == 0.0 and 0.0 < soft[20, 40] < 1.0


def test_temporal_smooth_only_in_band():
    cur = np.full((10, 10), 200, np.uint8)
    prev = np.zeros((10, 10), np.uint8)
    nxt = np.zeros((10, 10), np.uint8)
    band = np.zeros((10, 10), bool)
    band[0] = True
    out = matte.temporal_smooth(prev, cur, nxt, band)
    assert out[0, 0] == 66 and out[5, 5] == 200


# ----------------------------------------------------------------------------- 좌우 영상·포장


def test_eye_views_direction_transparency_and_inner_holes():
    h, w = 60, 120
    rgb = np.full((h, w, 3), 50, np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    alpha[10:55, 40:80] = 255  # 몸
    rgb[10:55, 40:80] = (200, 100, 50)
    rgb[25:35, 55:65] = (250, 250, 0)  # 몸 앞으로 뻗은 손 (더 가까움)
    inv = np.full((h, w), 0.5, np.float32)
    inv[25:35, 55:65] = 1.0
    mask = alpha > 0
    st = stereo.depth_stats(inv, mask)
    disp = stereo.disparity_px(inv, mask, lo=st["lo"], hi=st["hi"], t0=0.0, relief_px=8)
    left, right = stereo.eye_views(rgb, alpha, disp)

    def hand_x(e):
        ys, xs = np.nonzero((e[..., 0] > 240) & (e[..., 1] > 240))
        return xs.mean()

    assert hand_x(left) > hand_x(right) + 4  # 가까운 손: 왼눈에선 오른쪽, 오른눈에선 왼쪽
    assert left[5, 5, 3] == 0 and right[5, 115, 3] == 0  # 사람 바깥은 투명
    for e in (left, right):  # 손이 움직여 생긴 몸 안쪽 구멍은 채워져 있어야 한다
        assert (e[12:52, 46:74, 3] > 127).all()


def test_pack_layout_and_bleed():
    h, w = 20, 30
    lf = np.zeros((h, w, 4), np.uint8)
    r = np.zeros((h, w, 4), np.uint8)
    lf[5:15, 10:20] = (255, 0, 0, 255)
    r[5:15, 8:18] = (0, 0, 255, 255)
    p = stereo.pack(lf, r)
    assert p.shape == (2 * h, 2 * w, 3)
    assert (p[h + 10, 15] == 255).all() and (p[h + 1, 1] == 0).all()  # 아래 절반 = 알파
    assert (p[10, w + 12] == (0, 0, 255)).all()  # 위 오른쪽 = 오른눈 컬러
    assert p[2, 2, 0] > 100  # 투명한 곳 컬러는 사람 색으로 번짐 (검은 테두리 방지)
    g = stereo.green_sbs(lf, r)
    assert g.shape == (h, 2 * w, 3) and tuple(g[0, 0]) == stereo.GREEN


# ----------------------------------------------------------------------------- 사진 전체


def test_run_person_image_with_fake_models(tmp_path):
    img, _, _ = make_person_image(320, 180)
    src = tmp_path / "people.png"
    imageio.write_image(src, img)
    res = run_person(
        src,
        job_dir=tmp_path / "job",
        params={"relief_px": 12},
        models=PersonModels(detector=FakeDetector(), matter=FakeMatter(), depth=FakeDepth()),
        ctx=_ctx(),
    )
    packed = imageio.read_rgb(res["output"])
    assert packed.shape == (360, 640, 3)
    media = imageio.read_json(tmp_path / "job" / "out" / "media.json")
    assert media["layout"] == "rgba_sbs_v1" and media["people"] == 2 and media["type"] == "image"
    assert 0.6 < media["tallest_frac"] < 0.85
    assert media["disp_range_px"][1] > 0  # 시차가 실제로 생김
    alpha_l = packed[180:, :320, 0]
    assert alpha_l.max() == 255 and alpha_l[2, 2] == 0


def test_run_person_image_mask_file_without_any_model(tmp_path):
    img, masks, _ = make_person_image(320, 180)
    src = tmp_path / "p.png"
    mpath = tmp_path / "m.png"
    imageio.write_image(src, img)
    imageio.write_image(mpath, (masks[0] | masks[1]).astype(np.uint8) * 255)
    job = tmp_path / "job"
    params = {"mask_path": str(mpath)}
    res = run_person(
        src, job_dir=job, params=params, models=PersonModels(use_matter=False, use_depth=False), ctx=_ctx()
    )
    assert Path(res["output"]).exists() and res["people"] == 2
    # 두 번째 실행: 마스크·알파·깊이 단계는 캐시 사용 (meta key 동일)
    before = (job / "01_masks" / "meta.json").stat().st_mtime_ns
    run_person(
        src, job_dir=job, params=params, models=PersonModels(use_matter=False, use_depth=False), ctx=_ctx()
    )
    assert (job / "01_masks" / "meta.json").stat().st_mtime_ns == before


# ----------------------------------------------------------------------------- 영상 전체


def _ffmpeg_ok() -> bool:
    from vr180.video import frames as F

    return F.have_ffmpeg()


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg 없음")
def test_run_person_video_with_fake_models(tmp_path):
    from vr180.video import frames as F

    frames, _ = make_person_video(w=320, h=180, n=40, enter_frame=12)
    fd = tmp_path / "src_frames"
    fd.mkdir()
    for i, f in enumerate(frames):
        imageio.write_image(fd / f"{i:05d}.png", f)
    src = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            "30",
            "-i",
            str(fd / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    res = run_person(
        src,
        job_dir=tmp_path / "job",
        params={"codec": "libx264", "crf": 20},
        models=PersonModels(detector=FakeDetector(), matter=FakeMatter(), depth=FakeDepth(), sam2=FakeSam2()),
        ctx=_ctx(),
    )
    info = F.probe(Path(res["output"]))
    assert (info["width"], info["height"]) == (640, 360)
    assert info["nb_frames"] in (40, 0)  # 일부 ffprobe 는 nb_frames 를 안 준다
    g = F.probe(Path(res["green"]))
    assert (g["width"], g["height"]) == (640, 180)
    media = imageio.read_json(tmp_path / "job" / "out" / "media.json")
    assert media["type"] == "video" and media["people"] == 2 and media["frames"] == 40
    people = imageio.read_json(tmp_path / "job" / "01_masks" / "people.json")
    assert people["tracker"] == "sam2"
    # 들어오던 사람(B) 이 키프레임(15) 전에도 마스크에 있어야 한다
    m14 = imageio.read_mask(tmp_path / "job" / "01_masks" / "00014.png") > 127
    assert m14[:, 280:].sum() > 50


@pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg 없음")
def test_run_person_video_falls_back_without_sam2(tmp_path, monkeypatch):
    frames, _ = make_person_video(w=160, h=90, n=12, enter_frame=4)
    fd = tmp_path / "f"
    fd.mkdir()
    for i, f in enumerate(frames):
        imageio.write_image(fd / f"{i:05d}.png", f)
    src = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            "24",
            "-i",
            str(fd / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    monkeypatch.setenv("VR180_MODELS_DIR", str(tmp_path / "no_models"))
    logs = []
    run_person(
        src,
        job_dir=tmp_path / "job",
        params={"codec": "libx264"},
        models=PersonModels(detector=FakeDetector(), use_matter=False, use_depth=False),
        ctx=RunContext(log_fn=logs.append),
    )
    assert any("매 프레임 검출" in s for s in logs)
    assert imageio.read_json(tmp_path / "job" / "01_masks" / "people.json")["tracker"] == "frame_detect"


def test_link_keyframes_walking_person_is_one_track():
    """0.5초에 몸 폭보다 많이 움직이는 사람도 한 사람으로 이어져야 (안 그러면 두 번 검출 규칙에 걸려 사라진다)."""
    h, w = 200, 400
    kfs = []
    for k in range(4):
        m = np.zeros((h, w), bool)
        x = 40 + k * 70  # 몸 폭 30px, 키프레임마다 70px 이동
        m[40:190, x : x + 30] = True
        kfs.append((k * 15, [_det(m, 0.6)]))
    tracks = T.link_keyframes(kfs, n_frames=60, interval=15, frame_area=h * w)
    assert len(tracks) == 1 and len(tracks[0].hits) == 4


def test_video_mask_edit_reruns_alpha(tmp_path):
    """영상: 사용자가 01_masks PNG 를 고쳐 덮어쓰면 다음 실행에서 경계(02_alpha) 단계가 다시 돈다."""
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg 없음")
    frames, _ = make_person_video(w=160, h=90, n=8, enter_frame=100)
    fd = tmp_path / "f"
    fd.mkdir()
    for i, f in enumerate(frames):
        imageio.write_image(fd / f"{i:05d}.png", f)
    src = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            "24",
            "-i",
            str(fd / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    job = tmp_path / "job"
    kw = dict(job_dir=job, params={"codec": "libx264", "tracker": "frame_detect"}, ctx=_ctx())
    run_person(src, models=PersonModels(detector=FakeDetector(), use_matter=False, use_depth=False), **kw)
    meta1 = (job / "02_alpha" / "meta.json").read_text()
    m = imageio.read_mask(job / "01_masks" / "00003.png")
    m[:, :20] = 255  # 사용자가 손으로 마스크를 고침
    imageio.write_image(job / "01_masks" / "00003.png", m)
    # 검출기를 안 넘기면 01_masks 는 캐시를 쓰고, 고친 파일 때문에 02_alpha 는 다시 돈다
    run_person(src, models=PersonModels(detector=None, use_matter=False, use_depth=False), **kw)
    assert (job / "02_alpha" / "meta.json").read_text() != meta1
    assert (imageio.read_mask(job / "02_alpha" / "00003.png")[:, :10] > 200).any()
