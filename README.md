# vr180 — 3D SBS → VR180 SBS 이미지 변환 파이프라인

일반 3D SBS 이미지 1장을 입력받아 Meta Quest에서 볼 수 있는 VR180 SBS 이미지 1장을 만든다.
원본이 찍은 화각 안쪽은 원본 L/R을 그대로 유지하고, 그 바깥(좌우·상하)만 AI로 생성해 채운다.

설계서: [3D SBS → VR180 SBS 이미지 변환 파이프라인 설계서](https://claude.ai/artifact/5tQi2C83bEbpQp872YACKG)

## 환경

- Windows 11 + RTX 3080 Ti (12GB) 기준. Linux 도 동일하게 동작.
- Python 3.11, [uv](https://docs.astral.sh/uv/). conda / Docker 는 쓰지 않는다.
- 모델 가중치는 저장소에 없고 `models/` 아래로 내려받는다. 외부 서비스 호출 없음.

```powershell
uv sync                      # .venv 생성 + 의존성 설치 (PyTorch CUDA 12.4 휠 포함)
copy .env.example .env       # (선택) HF_HOME, 작업 폴더 등
uv run vr180 make-sample samples/synthetic_sbs.png   # 합성 테스트 입력
uv run vr180 run -i samples/synthetic_sbs.png        # 모델 없이 뼈대 실행 (opencv/sgbm)
uv run vr180-gui                                     # http://127.0.0.1:7860
```

모델 붙이기:

```powershell
uv run vr180 fetch-models --da2 --sdxl           # Depth Anything V2 Large, SDXL inpainting
uv run vr180 fetch-models --flux                 # FLUX.1 Fill dev GGUF Q4 (HF_TOKEN 필요, 비상업 라이선스)
uv run vr180 fetch-models --raft-stereo          # RAFT-Stereo 코드(third_party/) + middlebury 가중치
uv run vr180 run -i x.png -s stage_03_outpaint.inpainter=sdxl -s stage_04_stereo.matcher=raft_stereo -s stage_05_depth.estimator=da2_large
```

## 구조

```
configs/default.yaml      모든 Stage 기본 파라미터, 모델 선택
work/<job_id>/            Job 별 중간 산출물 (params.json + Stage 폴더마다 meta.json)
src/vr180/geometry/       sphere.py (좌표 규약, 여기만), reproject.py, warp.py
src/vr180/models/         Inpainter / DepthEstimator / StereoMatcher / FovEstimator 인터페이스와 구현체
src/vr180/pipeline/       job.py (캐시, from-stage 재실행), stage_00_split … stage_07_output
src/vr180/gui/            Gradio 앱, 탭 하나당 파일 하나
tests/                    기하 단위 테스트 + 더미 end-to-end
```

Stage 와 산출물:

| # | Stage | 산출물 |
|---|-------|--------|
| 0 | split | `L.png` `R.png` `rectify_report.json` |
| 1 | fov | `camera.json` |
| 2 | reproj | `L_equi.png` `R_equi.png` `valid_mask.png` `preview_sbs.png` |
| 3 | outpaint | `pano_L.png` `tiles/` |
| 4 | stereo | `disp_center.npy` `disp_center_equi.npy` + 미리보기 |
| 5 | depth | `mono_invdepth.npy` `disp_full.npy` `fit_report.json` `fit_scatter.png` |
| 6 | right | `pano_R.png` `hole_mask.png` `anaglyph.png` |
| 7 | output | `output_VR180_SBS_180_3dh.png/.jpg` `preview_original_only.png` |

## CLI

```
vr180 run -i x.png                       새 Job, 전체 실행
vr180 run -j <job> --from-stage 3        Stage 3 부터 재실행 (0~2 는 캐시)
vr180 run -j <job> -s stage_03_outpaint.prompt="outdoor, sunny"
vr180 status -j <job>                    Stage 별 done / stale / missing
vr180 jobs
```

중간 산출물을 외부 툴로 고쳐 덮어쓰면(예: `03_outpaint/pano_L.png`) 그 다음 Stage 부터 재실행 시 고친 파일을 그대로 쓴다.

## 테스트

```
uv run pytest
```

## 모델과 라이선스

| 역할 | 기본 (모델 없음) | 모델 |
|------|------------------|------|
| 주변부 inpaint | OpenCV Telea | SDXL inpainting (fp16 ≈ 7GB), FLUX.1 Fill dev GGUF Q4 / NF4 (**비상업 라이선스**) |
| stereo disparity | OpenCV SGBM | RAFT-Stereo (middlebury), torchvision RAFT flow |
| mono depth | stereo 외삽 | Depth Anything V2 Large / Base / Small (Apache-2.0은 Small 만, Base/Large 는 CC-BY-NC-4.0) |
| FOV 추정 | 수동 | GeoCalib (`uv sync --extra geocalib`) |
