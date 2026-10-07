"""JPEG 에 GPano XMP 메타데이터를 넣는다 (APP1 세그먼트 삽입). 외부 의존성 없음.

half-equirect 눈당 S×S 를 좌우로 붙인 SBS 이미지에 대해, 왼쪽 눈 기준으로
FullPanoWidth = 2S (360°), Cropped = S×S, Left offset = S/2 로 기록한다.
Quest 갤러리가 이걸 읽는지는 기기에서 확인 (설계서 '열린 결정').
"""

from __future__ import annotations

import struct
from pathlib import Path

XMP_NS = b"http://ns.adobe.com/xap/1.0/\x00"


def gpano_xmp(pano_size: int) -> bytes:
    s = pano_size
    xml = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="vr180">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"
    GPano:ProjectionType="equirectangular"
    GPano:UsePanoramaViewer="True"
    GPano:StereoMode="Left-Right"
    GPano:FullPanoWidthPixels="{2 * s}"
    GPano:FullPanoHeightPixels="{s}"
    GPano:CroppedAreaImageWidthPixels="{s}"
    GPano:CroppedAreaImageHeightPixels="{s}"
    GPano:CroppedAreaLeftPixels="{s // 2}"
    GPano:CroppedAreaTopPixels="0"
    GPano:InitialViewHeadingDegrees="0"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xml.encode("utf-8")


def inject_xmp_jpeg(path: Path | str, pano_size: int) -> None:
    path = Path(path)
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError("JPEG 가 아님")
    payload = XMP_NS + gpano_xmp(pano_size)
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    # 기존 XMP APP1 이 있으면 제거
    i = 2
    out = bytearray(data[:2])
    while i + 4 <= len(data) and data[i] == 0xFF and data[i + 1] in range(0xE0, 0xF0):
        ln = struct.unpack(">H", data[i + 2 : i + 4])[0]
        chunk = data[i : i + 2 + ln]
        if not (data[i + 1] == 0xE1 and chunk[4 : 4 + len(XMP_NS)] == XMP_NS):
            out += chunk
        i += 2 + ln
    out += seg
    out += data[i:]
    path.write_bytes(bytes(out))
