# TODO

설계서(1차 범위) 이후에 추가하기로 한 항목. 각 항목은 "왜 → 무엇을 → 어디를 고치나 → 파라미터 → 검증" 순서로 적는다.
구현 순서는 0 → 1 → 2. 1 은 0 의 두 모드 모두에 적용되고, 2 는 0·1 을 그대로 쓴다.

---

## 0. 입력 모드 추가: 2D 사진 직접 입력 (`input_mode: mono`) — 최우선

### 왜
- 실제 입력은 "2D 사진 → 외부 툴로 깊이 추정 → SBS" 로 만든 **합성 스테레오** 다. 이걸 다시 Stage 4 로 매칭하고 Stage 5 로 피팅하는 건
  `깊이 → 워핑 → 측정 → 깊이 → 피팅` 의 왕복이라 정보만 잃는다 (워핑 구멍 메우기·매칭 오류가 섞인 복사본에 맞추는 꼴).
- 2D 를 직접 받으면 Stage 4 가 사라지고, 가운데와 주변부의 깊이가 **같은 모델·같은 그림** 에서 나와 처음부터 연속이다.
  경계에서 입체감이 튀는 문제가 구조적으로 없어지고, 외부 2D→3D 툴도 필요 없어진다.
- 영상(2번)에서 이득이 더 크다: 프레임마다 외부 툴로 SBS 를 만드는 과정이 통째로 빠진다.
- 실제 스테레오 카메라 사진은 여전히 `input_mode: sbs`(지금 구조)로 처리한다. 두 모드는 코드를 대부분 공유한다.

### 파라미터
- `params["_job"]["input_mode"]`: `sbs` | `mono`. Job 생성 시 지정 (`vr180 run -i x.jpg --mode mono`, GUI 라디오). 기본값은 `mono` 로 바꾼다 (실제 사용 입력이 2D 이므로).
- `stage_05_depth.depth_strength`: 각도 disparity 최댓값(rad). mono 모드에서 상대 inverse depth 를 시차로 바꾸는 유일한 눈금.
  기본 0.02 (≈1.1°, Quest 에서 편한 수준). GUI 슬라이더 0~0.06.
- `stage_05_depth.depth_map_path`: 선택. 외부 툴이 만든 깊이 지도(PNG/npy)를 그대로 쓰고 싶을 때. 비우면 Depth Anything.
- `stage_06_right.mono_symmetric`: true 면 가운데 뷰를 양쪽으로 ±d/2 워핑 (기본). false 면 가운데 뷰 = 왼눈, 오른눈만 −d 워핑.

### Stage 별 변경 (mono 모드)
| Stage | sbs (지금) | mono |
|---|---|---|
| 0 split | 가운데 자르기 + SIFT 수직 시차 | 자르기 없음. `C.png` 하나 저장. rectify_report 는 `{"mode":"mono"}` |
| 1 fov | L.png 로 추정 | C.png 로 추정. 변경 없음 |
| 2 reproj | L_equi, R_equi, valid_mask | `C_equi.png`, valid_mask. (L_equi/R_equi 는 만들지 않는다) |
| 3 outpaint | L_equi → pano_L | C_equi → `pano_C.png` |
| 4 stereo | SGBM/RAFT → disp_center_equi | **건너뜀** (outputs 비어 있음, meta 에 skipped 기록) |
| 5 depth | mono depth + RANSAC 피팅 + 블렌딩 | pano_C 전체에 mono depth 한 번(타일 or 전체). 피팅 없음: `disp = depth_strength × normalize(invdepth)` (1~99 백분위로 0..1 정규화). 극지방 완화·범위 페이드는 동일 |
| 6 right | pano_L → 워핑 → pano_R, 원본 영역은 R_equi | pano_C 를 +d/2 로 워핑 → `pano_L.png`, −d/2 로 워핑 → `pano_R.png`. 구멍은 양쪽에 반씩. 원본 영역도 워핑한다 (진짜 R 이 없으므로). 산출물 이름은 sbs 모드와 같게 유지 → Stage 7 변경 없음 |
| 7 output | pano_L + pano_R | 동일 |

- `pipeline/job.py`: Stage 모듈의 `inputs()/outputs()` 가 `job.input_mode` 를 보고 다른 목록을 돌려준다. `run()` 은 outputs 가 빈 Stage 를 "skipped" 로 처리한다.
- `geometry/warp.py`: `forward_warp` 는 그대로. 음수 disparity(반대 방향 워핑)가 들어와도 z-buffer 가 맞도록 "가까운 쪽이 이김" 판정을 `|disp|` 기준으로 바꾼다.
- Stage 6 의 `_fill_holes` 는 양 눈에 각각 호출. 애너글리프는 pano_L/pano_R 로 동일.
- `sample.py`: `make_mono` 추가 (make_sbs 의 가운데 카메라 1대).

### GUI
- 상단 "새 Job" 옆에 입력 모드 라디오 (2D 사진 / 3D SBS).
- 탭 0: mono 면 수직 시차 히스토그램·swap_lr·rectify 숨김, 원본 1장만 표시.
- 탭 4: mono 면 "이 모드에서는 건너뜀" 안내만.
- 탭 5: mono 면 band/blend/a,b 오버라이드 숨기고 `depth_strength` 슬라이더 + 깊이 지도 파일 입력 표시.
- 탭 6: L/R 토글이 pano_L/pano_R (둘 다 워핑 결과).

### 검증
- `tests/test_job.py`: mono 더미 end-to-end. Stage 4 가 skipped, pano_L/pano_R 이 서로 좌우 대칭 방향으로 밀렸는지(한 깊이 평면 샘플에서 L 은 +du, R 은 −du), 최종 SBS 크기.
- `tests/test_warp.py`: 음수 disparity 워핑 방향 테스트.
- Quest 체크: 가운데 원본이 선명한지(양쪽으로 반씩 밀어 보간이 두 번 들어가므로 Lanczos 유지), 경계에서 입체감 연속.

### 설계서 반영
- "목표와 범위 > 입력" 을 "2D 사진(기본) 또는 3D SBS" 로. "8개 Stage… Right eye 주변부는 disparity 워핑" 문장에 mono 모드 설명 추가.
- 구현 순서: 0번을 Stage 4(stereo) 앞에 둔다. mono 모드부터 Quest 로 확인하고, sbs 모드의 Stage 4/5 피팅은 실제 스테레오 입력이 생길 때 검증.

---

## 1. 생성 범위 제한 + 페이드 (기본 수평 120°, 수직 100°)

### 왜
- 180°×180° 전부를 생성하면 가장 어렵고 오류가 많은 곳(모서리, 천장·바닥 끝)까지 모델을 돌려야 한다.
- Quest 시야는 약 100° 라서 정면을 볼 때 ±50° 바깥은 거의 안 보인다.
- 정면 기준 일정 범위까지만 생성하고 그 바깥은 서서히 어둡게 하면 생성량·오류·시간이 절반 가까이 줄고 체감은 비슷하다.

### 범위의 정의
- equirect 각도 (θ, φ) 기준 타원. `r = sqrt((θ / (h/2))² + (φ / (v/2))²)`, `r ≤ 1` 이면 범위 안.
  - **기본은 "원본 화각 + 여유"**: `h = hfov + 2·extent_margin_h_deg`, `v = vfov + 2·extent_margin_v_deg` (기본 여유 좌우 20°, 상하 25°).
  80° 16:9 사진이면 자동으로 약 120×100 이 되고, 110° 4:3 이면 약 150×144 가 된다.
  실측(그림으로 확인): 110° 4:3 사진에 120×100 고정값을 쓰면 원본이 범위 밖으로 삐져나와 가장자리가 페이드로 어두워진다. 그래서 고정값은 옵션으로만 둔다.
- 고정값 옵션: `extent_h_deg`, `extent_v_deg` 를 직접 주면 그 값을 쓴다 (null 이면 위 규칙). 둘 다 180 이면 기능 꺼짐(지금과 동일).
- 어느 경우든 범위가 `valid_mask` + 페이드 폭보다 작으면 경고 후 자동으로 키운다.
- 페이드 가중치 w: `r ≤ 1 − f` 에서 1, `r = 1` 에서 0, 사이는 smoothstep. `f` 는 `extent_fade_deg` 를 반지름 비율로 환산한 값.
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
| `extent_margin_h_deg` | 20 | 원본 수평 화각 바깥으로 더 생성할 여유 (한쪽 기준) |
| `extent_margin_v_deg` | 25 | 원본 수직 화각 바깥 여유 (한쪽 기준). 천장·바닥은 Quest 시야에 더 들어오므로 조금 넓게 |
| `extent_h_deg` | null | 고정 수평 범위. 주면 margin 규칙 대신 사용 |
| `extent_v_deg` | null | 고정 수직 범위 |
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
- 탭 2: 여유 슬라이더 2개(좌우 0~50°, 상하 0~50°) + "고정값 사용" 토글 시 수평/수직 90~180 슬라이더, 페이드 0~30, 색 선택, 모양 선택. 계산된 실제 범위(예: 120×100)를 숫자로 표시. ImageSlider 위에 경계선 오버레이. "범위 바깥 비율 n%" 표시.
- 탭 3: 타일 갤러리에 "건너뜀(범위 밖)" 라벨. 선택 재생성 CheckboxGroup 에서도 구분.
- 탭 5: disp_full 미리보기 캡션에 "범위 바깥은 0 이 정상" 표시.

### 검증
- `tests/test_extent.py`: 80° 16:9 → 자동 범위가 약 120×100 인지. h=120, v=100 에서 (0°,0°) 안 / (70°,0°) 밖 / (0°,55°) 밖, 페이드 띠 중간에서 w≈0.5, 180/180 이면 전부 255·w=1.
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
