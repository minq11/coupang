# samples/

테스트용 3D SBS 이미지를 여기에 넣는다 (git 에는 올라가지 않는다).

- 형식: 좌우 배열 SBS, JPG/PNG. 예: 3840×1080 (눈당 1920×1080)
- 좌=Left eye 가정. 반대면 Stage 0 의 `swap_lr` 를 켠다.
- 합성 테스트 이미지는 `python -m vr180 make-sample samples/synthetic_sbs.png` 로 만들 수 있다.
