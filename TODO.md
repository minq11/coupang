# TODO

설계서(1차 범위) 이후에 추가하기로 한 항목. 각 항목은 "왜 → 무엇을 → 어디를 고치나 → 파라미터 → 검증" 순서로 적는다.
구현 순서는 1 → 2. 2 는 1 의 범위 제한을 그대로 쓴다.

---

## 1. 생성 범위 제한 + 페이드 (기본 수평 120°, 수직 100°)

### 왜
- 180°×180° 전부를 생성하면 가장 어렵고 오류가 많은 곳(모서리, 천장·바닥 끝)까지 모델을 돌려야 한다.
- Quest 시야는 약 100° 라서 정면을 볼 때 ±50° 바깥은 거의 안 보인다.
- 정면 기준 일정 범위까지만 생성하고 그 바깥은 서서히 어둡게 하면 생성량·오류·시간이 절반 가까이 줄고 체감은 비슷하다.

### 범위의 정의
- equirect 각도 (θ, φ) 기준 타원. `r = sqrt((θ / (h/2))² + (φ / (v/2))²)`, `r ≤ 1` 이면 범위 안.
  - h = `extent_h_deg`, v = `extent_v_deg`. 둘 다 180 이면 기능 꺼짐(지금과 동일).
- 페이드 가중치 w: `r ≤ 1 − f` 에서 1, `r = 1` 에서 0, 사이는 smoothstep. `f` 는 `extent_fade_deg` 를 반지름 비율로 환산한 값.
- 범위는 원본 유효 영역(valid_mask)을 페이드 폭만큼 여유 있게 포함해야 한다. 작으면 경고 후 자동으로 키운다.
- 모양을 타원으로 한 이유: 각도 기준 사각형은 공에 감았을 때 모서리가 뾰족하게 튀고, 타원은 비네팅처럼 자연스럽다. (열린 결정: 둥근 사각형 옵션 `extent_shape: ellipse | rounded_rect`)

### 소유 Stage: Stage 2
범위는 캔버스 위의 기하 영역이므로 Stage 2 가 정의하고 **파일로 남긴다**. 뒤 Stage 는 그 파일을 입력으로 읽는다.
→ 파라미터가 바뀌면 Stage 2(수 초)가 다시 돌고, 뒤 Stage 는 입력 해시가 바뀌어 자동으로 stale 이 된다. 별도 캐시 키 규칙이 필요 없다.

- 계산 함수: `src/vr180/geometry/extent.py` 한 곳. `extent_weight(size, h_deg, v_deg, fade_deg) -> (mask: bool S×S, weight: float32 S×S)`.
- 새 산출물 `02_reproj/`:
  - `extent_mask.png` — 255 = 범위 안 (페이드 띠 포함)
  - `extent_fade.png` — 0~255 가중치 (255 = 완전 생성, 0 = 완전 바깥)
  - `preview_sbs.png` 에 범위 경계선을 1px 흰 선으로 그려서 Quest 1차 확인 때 "어디까지 채워질지" 를 같이 본다.
- `stage_02_reproj.outputs()` 에 두 파일 추가. Stage 3/5/6 의 `inputs()` 에 `extent_mask.png`, `extent_fade.png` 추가.

### 파라미터 (`configs/default.yaml` → `stage_02_reproj`)
| 키 | 기본값 | 설명 |
|---|---|---|
| `extent_h_deg` | 120 | 수평 생성 범위 (전체 폭) |
| `extent_v_deg` | 100 | 수직 생성 범위 |
| `extent_fade_deg` | 10 | 페이드 폭. 8° 미만이면 시차가 급히 0 이 돼 "깊이 벽" 이 느껴진다 |
| `extent_color` | `[0, 0, 0]` | 바깥 색. `auto` 면 생성 경계 평균색의 20% 밝기 |
| `extent_shape` | `ellipse` | `ellipse` / `rounded_rect` |

### Stage 별 변경
**Stage 3 (outpaint)**
- 타일 건너뛰기 규칙 (주의: "타일 면적 중 범위 안 비율" 이 아니다. 모서리 타일도 범위 안에 22% 걸친다):
  **그 타일이 새로 채울 픽셀** = `extent_mask ∧ (pano alpha == 0, 즉 원본도 아니고 앞 타일도 못 덮은 곳) ∧ tile inside` 의 개수가
  타일이 덮는 전체 픽셀의 `tile_min_new_fraction`(기본 1%) 미만이면 모델을 돌리지 않고 로그만 남긴다.
  - 실측: 120×100 과 140×120 에서는 좌/우/위/아래 4장이 생성 영역을 100% 덮어 모서리 4장이 빠진다. 180×180 에서는 4장이 99.6% 를 덮고 모서리는 0.4% 만 새로 채운다.
  - 규칙이 "새로 채울 픽셀" 기준이라 범위를 어떻게 잡든 자동으로 맞는다. 순서(좌/우 → 위/아래 → 모서리)는 그대로.
- inpaint 마스크: `(alpha < 1) ∧ extent_mask`. 페이드 띠 끝까지 생성해야 섞을 내용이 있다.
- 마지막 합성: `rgb = gen × w + color × (1 − w)`. 바깥(w = 0)은 `extent_color`. 기존 "못 덮은 곳 Telea" 는 `extent_mask` 안에서만.
- `pano_L.png` 크기·형식은 그대로(180° 전체, 바깥이 어두울 뿐). 뒤 Stage 와 출력 형식 변경 없음.
- `meta.json` 에 실행한 타일 / 건너뛴 타일 목록을 기록한다 (GUI 표시용).

**Stage 5 (depth)**
- mono depth 타일도 같은 규칙으로 건너뛴다 (9장 → 5장). 기준은 `extent_mask ∧ tile inside` 비율.
- `disp_full = disp_full × w`. 바깥은 0(무한히 멀리). 어두운 영역이 양쪽 눈에 똑같이 보이려면 시차 0 이어야 하고, 페이드 폭만큼 서서히 줄여야 깊이 벽이 안 생긴다. 기존 극지방 완화와 곱으로 겹친다.
- 피팅 띠(band)는 원본 경계 안쪽이라 영향 없음.

**Stage 6 (right)**
- 워핑 `src_mask = extent_mask`. hole 계산·채우기도 `extent_mask` 안에서만.
- 합성 후 `pano_R = warped × w + color × (1 − w)`. 왼눈·오른눈의 바깥 픽셀이 **완전히 동일** 해야 한다(시차 0).

**Stage 4, 7**: 변경 없음.

### GUI
- 탭 2: 슬라이더 3개(수평 90~180, 수직 90~180, 페이드 0~30), 색 선택, 모양 선택. ImageSlider 위에 경계선 오버레이. "범위 바깥 비율 n%" 표시.
- 탭 3: 타일 갤러리에 "건너뜀(범위 밖)" 라벨. 선택 재생성 CheckboxGroup 에서도 구분.
- 탭 5: disp_full 미리보기 캡션에 "범위 바깥은 0 이 정상" 표시.

### 검증
- `tests/test_extent.py`: h=120, v=100 에서 (0°,0°) 안 / (70°,0°) 밖 / (0°,55°) 밖, 페이드 띠 중간에서 w≈0.5, 180/180 이면 전부 255·w=1.
- `tests/test_job.py` 추가: 범위 120×100 더미 실행 후 `meta.json` 실행 타일이 4장인지, `pano_L`·`pano_R` 바깥 픽셀이 색과 일치하고 두 눈이 동일한지, `disp_full` 바깥이 0 인지.
- Quest 체크리스트 추가: 페이드 경계에서 깊이 벽이 안 느껴질 것, 고개를 60° 돌렸을 때 어두워지는 시작이 자연스러울 것.

### 예상 효과 / 트레이드오프
- Stage 3 약 절반, Stage 5 약 절반, hole 채우기 감소.
- 고개를 60° 이상 돌리면 어둡다. 둘러보기 용도면 140~160 으로 올리거나 끈다.
- 열린 결정: 바깥을 완전 검정으로 둘지 `auto`(경계색을 어둡게)로 둘지 → Quest 로 보고 결정.

---

## 2. 영상 변환 (`vr180 video`) — 고정 샷 전용

### 왜
- 주변부를 **한 번만** 만들고 모든 프레임에 재사용하면 프레임당 작업은 remap 두 번 + 합성뿐이다. 모델은 영상 전체에서 한 번만 돈다.
- 원본 영역은 진짜 L/R 영상이라 입체감이 이미 있고, Stage 4 도 주변부 눈금 맞출 때 한 번만 쓰면 된다.
- 조건: **카메라가 고정(삼각대)** 이고 주변 환경이 정적일 것. 손으로 든 영상·패닝은 경계가 어긋나므로 1차 범위 밖.

### 구조
```
vr180 video -i in.mp4 -o out.mp4 [--ref-frame 0 | --ref-time 1.5] [--surround-job <job_id>]
            [--extent 120x100] [--eye-size 2880] [--codec hevc] [--crf 20] [--fps same]
```
1. **기준 프레임으로 주변부 생성(한 번)**: `--ref-frame` 을 PNG 로 뽑아 기존 이미지 파이프라인을 그대로 돌린다 (Stage 0~7). 결과 Job 이 "surround job". `--surround-job` 으로 이미 만든 Job 을 지정하면 이 단계를 건너뛴다.
   - 여기서 얻는 고정 자산: `pano_L.png`, `pano_R.png`, `valid_mask.png`, `camera.json`, `rectify_report.json`(수직 보정값), `extent_fade.png`.
2. **프레임 루프**: ffmpeg 로 디코드 → 각 프레임에 대해
   - 좌우 분리 + 첫 프레임의 rectify 보정값(y 이동)을 그대로 적용
   - remap 표(한 번 계산해 메모리에 유지)로 L/R 을 equirect 로
   - 블렌딩 가중치 `w_in`(valid_mask 의 distanceTransform 기반, 한 번 계산) 으로 `frame_L = w_in × L_equi + (1 − w_in) × pano_L`, R 도 동일
   - 좌우 붙여서 인코더 파이프로 전달
3. **인코드**: ffmpeg 서브프로세스(rawvideo stdin → H.265). 오디오는 원본에서 그대로 복사(`-c:a copy`).

- 코드 위치: `src/vr180/video/` (`frames.py`: ffmpeg 디코드/인코드 파이프, `run.py`: 루프). 이미지 파이프라인의 geometry/util 함수를 그대로 부른다. Stage 모듈 내부 함수는 import 하지 않는다.
- 진행률·취소는 `RunContext` 재사용. GUI 에는 "8 영상" 탭 하나: 입력 영상, 기준 프레임 선택(슬라이더로 시간 지정 + 미리보기), surround Job 선택/생성, 출력 설정, 진행률, 첫 프레임 결과 미리보기.

### 프레임당 비용 (눈당 2880 기준 추정)
- remap ×2 ≈ 60~120 ms (CPU), 블렌딩 ≈ 30 ms, 인코드는 별도 스레드. 1분 30fps(1800프레임) ≈ 5~10분.
- 더 빠르게: `cv2.cuda.remap`(OpenCV CUDA 빌드 필요, 선택), 또는 PyTorch `grid_sample` 로 GPU remap. 1차는 CPU.

### 출력 해상도 / 코덱
- 기본 눈당 2880 → 5760×2880 SBS, H.265, 30fps. Quest 2/3 재생이 안전한 선.
- 눈당 4096(8192×4096)은 옵션으로만. Quest 에서 끊길 수 있음을 README 에 명시.
- 파일명 접미사 `_180_3dh` 유지. 영상용 메타데이터(st3d/sv3d 박스)는 열린 결정 — DeoVR/Skybox 에서 180 SBS 수동 설정으로 1차 확인.

### 보완 장치 (1차에 넣을 것 / 미룰 것)
- [1차] **카메라 움직임 경고**: 프레임 간 전역 이동량을 phase correlation(`cv2.phaseCorrelate`)으로 재서 평균이 임계값(기본 2px)을 넘으면 "고정 샷이 아님" 경고.
- [1차] **컷 감지**: 히스토그램 차이가 급변하면 컷으로 보고, 컷마다 새 기준 프레임으로 주변부를 다시 만든다(`--per-shot`).
- [미룸] 천천히 흐르는 샷: N초마다 주변부를 다시 만들어 교차 페이드.
- [미룸] 손으로 든 영상: 프레임마다 생성(비용 수백 배)이라 범위 밖.

### 새 의존성
- `ffmpeg` 실행 파일 (Windows: winget 또는 zip, PATH 에 추가). 파이썬 쪽은 subprocess 파이프만 쓰고 PyAV 는 안 넣는다.
- `pyproject.toml` 에는 추가 없음. ffmpeg 유무를 시작 시 확인해 없으면 설치 안내.

### 검증
- `tests/test_video.py`: 합성 SBS 프레임 10장을 ffmpeg 로 묶어 입력 → 출력 영상의 프레임 수·해상도 일치, 첫 프레임의 원본 영역이 입력과 일치(PSNR > 40), 바깥 영역이 surround 와 일치.
- ffmpeg 없으면 테스트 skip.
- Quest 체크: 경계에서 원본과 주변부 밝기 차이로 깜빡임이 없는지(노출 변동이 있는 영상은 경계 블렌딩 폭을 넓힌다), 재생이 끊기지 않는지.

### 순서
1. 1번(범위 제한) 먼저. 영상에서 주변부가 작을수록 생성 품질·시간 모두 유리하다.
2. `vr180 video` CPU 버전 + 고정 샷 경고 + 컷 감지.
3. GPU remap, 영상 메타데이터, 느린 패닝 보완은 Quest 확인 후.
