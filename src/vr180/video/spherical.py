"""mp4 에 Spherical Video V2 메타데이터(st3d + sv3d) 를 넣는다. 외부 의존성 없음.

VR 플레이어(Quest 갤러리·DeoVR·Skybox·Pigasus 등)는 두 가지로 VR180 여부를 판단한다.
1. 파일명 토큰 (_180, _LR, _SBS, _3dh …) — 플레이어마다 다르다.
2. moov/trak/mdia/minf/stbl/stsd/<비디오 샘플 엔트리> 안의 st3d(스테레오 배치) + sv3d(투영) 박스 — 표준.
   https://github.com/google/spatial-media/blob/master/docs/spherical-video-v2-rfc.md

여기서는 2 를 처리한다. ffmpeg 기본 출력은 moov 가 mdat 뒤에 있어 삽입해도 청크 오프셋이 안 바뀌지만,
faststart 등으로 moov 가 앞에 있는 파일도 다루도록 stco/co64 를 늘어난 길이만큼 보정한다.

VR180 SBS(눈당 S×S half-equirect) 기준값:
- st3d stereo_mode = 2 (left-right)
- sv3d/proj/equi bounds: top=bottom=0, left=right=0.25 (눈당 프레임이 360° equirect 의 가운데 절반)
"""

from __future__ import annotations

import struct
from pathlib import Path

VIDEO_SAMPLE_ENTRIES = {b"avc1", b"avc3", b"hev1", b"hvc1", b"mp4v", b"vp09", b"av01"}
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}
STEREO_LEFT_RIGHT = 2
STEREO_TOP_BOTTOM = 1
STEREO_MONO = 0


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def _full_box(typ: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _box(typ, struct.pack(">I", (version << 24) | flags) + payload)


def st3d_box(stereo_mode: int = STEREO_LEFT_RIGHT) -> bytes:
    return _full_box(b"st3d", struct.pack(">B", stereo_mode))


def sv3d_box(hfov_deg: float = 180.0, vfov_deg: float = 180.0) -> bytes:
    """equirect 투영. bounds 는 0.32 고정소수점으로 '360×180 전체에서 잘라낸 비율'."""
    crop_h = max(0.0, (360.0 - hfov_deg) / 360.0 / 2.0)
    crop_v = max(0.0, (180.0 - vfov_deg) / 180.0 / 2.0)
    q = lambda x: int(round(x * (1 << 32))) & 0xFFFFFFFF  # noqa: E731
    svhd = _full_box(b"svhd", b"vr180\x00")
    prhd = _full_box(b"prhd", struct.pack(">iii", 0, 0, 0))  # yaw/pitch/roll 16.16 = 0
    equi = _full_box(b"equi", struct.pack(">IIII", q(crop_v), q(crop_v), q(crop_h), q(crop_h)))
    proj = _box(b"proj", prhd + equi)
    return _box(b"sv3d", svhd + proj)


def _iter_boxes(buf: bytes, start: int, end: int):
    """(type, box_start, header_len, box_size) 를 순서대로. size==0 은 '끝까지'."""
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos : pos + 8])
        hlen = 8
        if size == 1:
            size = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            raise ValueError(f"손상된 mp4 박스 {typ!r} @ {pos}")
        yield typ, pos, hlen, size
        pos += size


def _children_offset(typ: bytes, hlen: int) -> int:
    """컨테이너 박스의 자식이 시작하는 헤더 뒤 오프셋 (stsd 는 FullBox + entry_count)."""
    if typ == b"stsd":
        return hlen + 4 + 4
    if typ in VIDEO_SAMPLE_ENTRIES:
        return hlen + 78  # VisualSampleEntry 고정 필드
    return hlen


def _find_video_sample_entry(
    buf: bytes, moov_start: int, moov_end: int
) -> list[tuple[bytes, int, int, int]] | None:
    """moov 안에서 첫 비디오 트랙의 샘플 엔트리까지 경로 [(type, start, hlen, size), ...] 를 돌려준다."""
    for t_typ, t_pos, t_h, t_sz in _iter_boxes(buf, moov_start + 8, moov_end):
        if t_typ != b"trak":
            continue
        path = [(t_typ, t_pos, t_h, t_sz)]
        mdia = _child(buf, path[-1], b"mdia")
        if mdia is None:
            continue
        hdlr = _child(buf, mdia, b"hdlr")
        if hdlr is None or buf[hdlr[1] + hdlr[2] + 8 : hdlr[1] + hdlr[2] + 12] != b"vide":
            continue
        path.append(mdia)
        for name in (b"minf", b"stbl", b"stsd"):
            nxt = _child(buf, path[-1], name)
            if nxt is None:
                return None
            path.append(nxt)
        stsd = path[-1]
        for e in _iter_boxes(buf, stsd[1] + _children_offset(b"stsd", stsd[2]), stsd[1] + stsd[3]):
            path.append(e)
            return path
        return None
    return None


def _child(buf: bytes, parent: tuple[bytes, int, int, int], name: bytes):
    typ, pos, hlen, size = parent
    for c in _iter_boxes(buf, pos + _children_offset(typ, hlen), pos + size):
        if c[0] == name:
            return c
    return None


def _strip_existing(buf: bytearray, entry: tuple[bytes, int, int, int]) -> int:
    """샘플 엔트리 안의 기존 st3d/sv3d 를 지우고 지운 바이트 수를 돌려준다."""
    typ, pos, hlen, size = entry
    removed = 0
    for c in reversed(list(_iter_boxes(bytes(buf), pos + _children_offset(typ, hlen), pos + size))):
        if c[0] in (b"st3d", b"sv3d"):
            del buf[c[1] : c[1] + c[3]]
            removed += c[3]
    return removed


def _add_size(buf: bytearray, pos: int, hlen: int, delta: int) -> None:
    if hlen == 16:
        old = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
        buf[pos + 8 : pos + 16] = struct.pack(">Q", old + delta)
    else:
        old = struct.unpack(">I", buf[pos : pos + 4])[0]
        if old == 0:
            return  # '끝까지' 박스는 그대로
        buf[pos : pos + 4] = struct.pack(">I", old + delta)


def _shift_chunk_offsets(buf: bytearray, moov_start: int, moov_end: int, delta: int, threshold: int) -> int:
    """moov 가 mdat 앞에 있을 때: threshold 이후를 가리키는 stco/co64 항목에 delta 를 더한다."""
    n = 0

    def walk(start: int, end: int) -> None:
        nonlocal n
        for typ, pos, hlen, size in _iter_boxes(bytes(buf), start, end):
            if typ in _CONTAINERS:
                walk(pos + hlen, pos + size)
            elif typ in (b"stco", b"co64"):
                cnt = struct.unpack(">I", buf[pos + hlen + 4 : pos + hlen + 8])[0]
                base = pos + hlen + 8
                fmt, w = (">I", 4) if typ == b"stco" else (">Q", 8)
                for i in range(cnt):
                    o = base + i * w
                    v = struct.unpack(fmt, buf[o : o + w])[0]
                    if v >= threshold:
                        buf[o : o + w] = struct.pack(fmt, v + delta)
                        n += 1

    walk(moov_start + 8, moov_end)
    return n


def inject_vr180(
    path: Path | str,
    *,
    stereo_mode: int = STEREO_LEFT_RIGHT,
    hfov_deg: float = 180.0,
    vfov_deg: float = 180.0,
) -> dict:
    """mp4 에 st3d + sv3d 를 넣는다 (제자리 수정). 반환: {"inserted": bytes, "moov_before_mdat": bool, ...}."""
    path = Path(path)
    buf = bytearray(path.read_bytes())
    top = list(_iter_boxes(bytes(buf), 0, len(buf)))
    moov = next((b for b in top if b[0] == b"moov"), None)
    if moov is None:
        raise ValueError("moov 박스가 없음 (mp4 가 아니거나 미완성 파일)")
    mdats = [b for b in top if b[0] == b"mdat"]
    path_boxes = _find_video_sample_entry(bytes(buf), moov[1], moov[1] + moov[3])
    if not path_boxes:
        raise ValueError("비디오 트랙의 샘플 엔트리(stsd) 를 찾지 못함")
    entry = path_boxes[-1]

    removed = _strip_existing(buf, entry)
    entry = (entry[0], entry[1], entry[2], entry[3] - removed)
    new = st3d_box(stereo_mode) + sv3d_box(hfov_deg, vfov_deg)
    insert_at = entry[1] + entry[3]
    buf[insert_at:insert_at] = new
    delta = len(new) - removed

    # 경로상의 모든 조상 (샘플 엔트리 → stsd → stbl → minf → mdia → trak → moov) 크기 갱신
    for _typ, pos, hlen, _ in path_boxes:
        _add_size(buf, pos, hlen, delta)
    _add_size(buf, moov[1], moov[2], delta)

    moov_before_mdat = any(m[1] > moov[1] for m in mdats)
    shifted = 0
    if moov_before_mdat and delta:
        shifted = _shift_chunk_offsets(buf, moov[1], moov[1] + moov[3] + delta, delta, moov[1] + moov[3])
    path.write_bytes(bytes(buf))
    return {
        "inserted": len(new),
        "removed": removed,
        "moov_before_mdat": moov_before_mdat,
        "chunk_offsets_shifted": shifted,
        "sample_entry": entry[0].decode("latin1"),
    }
