"""영상 변환 — 고정 샷 (삼각대) 전제. 기준 프레임의 주변부(pano_L/pano_R) 를 1 번만 만들고
모든 프레임에 재사용. 프레임당 작업은 L/R 분리 + remap + 블렌딩 + 인코드.

입력 모드:
- `sbs`: iw3 등으로 미리 SBS 변환된 영상 (기본). 각 프레임이 좌/우 붙은 입력.
- `mono`: 2D 영상 + Video Depth Anything (#3 에서 구현). 아직 미구현.

외부 의존성: ffmpeg 실행 파일 (PATH 에).
"""

from vr180.video.run import run_video  # noqa: F401
