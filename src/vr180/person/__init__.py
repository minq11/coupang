"""인물 AR — 2D 사진·영상 속 모든 사람을 잘라 입체로 만들고, 퀘스트 패스스루 위에 띄울 영상을 만든다.

흐름 (사진은 2 를 건너뜀)
  1 사람 찾기     RF-DETR-Seg (Apache-2.0)         키프레임마다 person 마스크
  2 따라가기      SAM 2.1 (Apache-2.0)              키프레임 마스크를 프롬프트로 앞·뒤 방향 전파
  3 경계 다듬기   BiRefNet-matting (MIT)            안쪽 불투명 / 바깥 투명 / 경계 띠만 매팅 값
  4 깊이          Video Depth Anything Small · DA2 Small (Apache-2.0)
  5 좌우 영상     자체 코드 (geometry.warp.splat_horizontal)
  6 인코딩        위 컬러 / 아래 알파, 눈당 좌우 배치 + 초록 배경 SBS (ffmpeg)
재생은 player/ 의 WebXR 페이지 (vr180 player).
"""
