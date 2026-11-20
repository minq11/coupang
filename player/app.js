// 인물 AR 플레이어 — layout "rgba_sbs_v1" 영상/사진을 패스스루 위 판넬 두 장(눈마다 하나)에 띄운다.
//   위 절반 [왼눈 컬러 | 오른눈 컬러], 아래 절반 [왼눈 알파 | 오른눈 알파]
// three.js 는 XR 에서 레이어 1 을 왼눈, 레이어 2 를 오른눈 카메라에만 보여 준다.
import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const say = (t) => { statusEl.textContent = t; };

// ---------------------------------------------------------------- 렌더러·장면
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x2a2b2e, 1);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(0, 1.6, 0);
let previewEye = 1; // 1 = 왼눈, 2 = 오른눈 (XR 밖 미리보기)
camera.layers.enable(previewEye);

const floorGrid = new THREE.GridHelper(10, 20, 0x666666, 0x444444); // PC 미리보기에서만
scene.add(floorGrid);

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
// flipY 된 텍스처: v=1 이 원본 위쪽. 위 절반(컬러) v∈[0.5,1], 아래 절반(알파) v∈[0,0.5].
// 반 텍셀 안쪽으로 샘플링해 옆 영역(다른 눈 / 컬러↔알파) 색이 섞이지 않게 한다.
const FRAG = /* glsl */`
varying vec2 vUv;
uniform sampler2D map;
uniform float eyeOffset;
uniform vec2 texel;
void main() {
  float u = eyeOffset + mix(0.5 * texel.x, 0.5 - 0.5 * texel.x, vUv.x);
  float vc = mix(0.5 + 0.5 * texel.y, 1.0 - 0.5 * texel.y, vUv.y);
  float va = mix(0.5 * texel.y, 0.5 - 0.5 * texel.y, vUv.y);
  vec3 c = texture2D(map, vec2(u, vc)).rgb;
  float a = texture2D(map, vec2(u, va)).g;
  if (a < 0.01) discard;
  gl_FragColor = vec4(c, a);
}`;

function eyeMaterial(eyeOffset) {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: null }, eyeOffset: { value: eyeOffset }, texel: { value: new THREE.Vector2(1 / 2048, 1 / 2048) } },
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });
}

const panel = new THREE.Group();
scene.add(panel);
const geom = new THREE.PlaneGeometry(1, 1);
const meshL = new THREE.Mesh(geom, eyeMaterial(0.0));
const meshR = new THREE.Mesh(geom, eyeMaterial(0.5));
meshL.layers.set(1);
meshR.layers.set(2);
panel.add(meshL, meshR);
panel.visible = false;

// ---------------------------------------------------------------- 배치
const TARGET_HEIGHT_M = 1.7; // 가장 크게 나온 사람의 키
const layout = { dist: 2.0, scale: 1.0, yaw: 0.0, lift: 0.0, floor: true };
let meta = null;

function applyLayout() {
  if (!meta) return;
  const frac = meta.tallest_frac > 0.05 ? meta.tallest_frac : 0.8;
  const bottom = meta.tallest_frac > 0.05 ? meta.tallest_bottom : 1.0;
  const h = THREE.MathUtils.clamp(TARGET_HEIGHT_M / frac, 0.3, 12) * layout.scale;
  const w = h * (meta.eye_width / meta.eye_height);
  // 가장 큰 사람의 발끝(이미지 위에서 bottom 비율)이 바닥 y=0 에 오도록
  const yBottom = -h * (1 - bottom) + layout.lift + (layout.floor ? 0 : -1.6);
  meshL.scale.set(w, h, 1);
  meshR.scale.set(w, h, 1);
  panel.position.set(Math.sin(layout.yaw) * -layout.dist, yBottom + h / 2, Math.cos(layout.yaw) * -layout.dist);
  panel.rotation.set(0, layout.yaw, 0);
}

function resetLayout() { Object.assign(layout, { dist: 2.0, scale: 1.0, yaw: 0.0, lift: 0.0 }); applyLayout(); }

// ---------------------------------------------------------------- 미디어
let video = null;
let texture = null;

function setTexture(tex, w, h) {
  tex.colorSpace = THREE.NoColorSpace; // 컬러·알파를 원본 값 그대로 (셰이더가 그대로 출력)
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  for (const m of [meshL, meshR]) {
    m.material.uniforms.map.value = tex;
    m.material.uniforms.texel.value.set(1 / w, 1 / h);
  }
  if (texture && texture !== tex) texture.dispose();
  texture = tex;
}

async function loadItem(item) {
  meta = item;
  panel.visible = false;
  if (video) { video.pause(); video.removeAttribute('src'); video.load(); video = null; }
  const url = item.base + item.file;
  say(`불러오는 중: ${item.id}`);
  if (item.type === 'video') {
    video = document.createElement('video');
    video.src = url; video.loop = true; video.muted = true; video.playsInline = true; video.crossOrigin = 'anonymous';
    await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = () => rej(new Error('영상 로드 실패')); });
    setTexture(new THREE.VideoTexture(video), video.videoWidth, video.videoHeight);
    $('play').disabled = false;
  } else {
    const tex = await new THREE.TextureLoader().loadAsync(url);
    setTexture(tex, tex.image.width, tex.image.height);
    $('play').disabled = true;
  }
  applyLayout();
  panel.visible = true;
  $('ar').disabled = $('vr').disabled = !navigator.xr;
  say(`${item.id}\n사람 ${item.people}명 · ${item.type === 'video' ? `${item.frames}프레임 ${item.fps?.toFixed?.(1)}fps` : '사진'} · 눈당 ${item.eye_width}×${item.eye_height}` + (navigator.xr ? '' : '\n이 브라우저는 WebXR 을 지원하지 않습니다 (퀘스트 브라우저에서 여세요).'));
  window.__playerReady = true;
}

async function refreshList() {
  const sel = $('media');
  const q = new URLSearchParams(location.search);
  let items = [];
  try { items = await (await fetch('/api/media', { cache: 'no-store' })).json(); } catch (e) { /* 서버 없이 ?src= 만 쓸 때 */ }
  if (q.get('src')) items.unshift({ id: q.get('src'), base: '', file: q.get('src'), type: /\.(mp4|webm|mov)$/i.test(q.get('src')) ? 'video' : 'image', eye_width: 16, eye_height: 9, tallest_frac: 0, tallest_bottom: 1, people: '?' });
  sel.innerHTML = '';
  if (!items.length) { sel.innerHTML = '<option value="">결과 없음 — 먼저 인물 AR 변환을 돌리세요</option>'; say(''); return; }
  items.forEach((it, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${it.id} (${it.type === 'video' ? '영상' : '사진'}, ${it.people}명)`; sel.appendChild(o); });
  sel.onchange = () => loadItem(items[+sel.value]).catch((e) => say('오류: ' + e.message));
  await loadItem(items[0]).catch((e) => say('오류: ' + e.message));
}

// ---------------------------------------------------------------- XR 세션
async function startXR(mode) {
  if (!navigator.xr) return say('WebXR 미지원 브라우저');
  if (video) { try { await video.play(); } catch (e) { /* 음소거라 대부분 성공 */ } }
  try {
    const session = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor'] });
    session.addEventListener('end', onSessionEnd);
    renderer.setClearColor(0x000000, mode === 'immersive-ar' ? 0 : 1); // AR 은 투명 배경이어야 패스스루가 보인다
    floorGrid.visible = false;
    camera.layers.set(0); // XR 눈 카메라가 반대쪽 눈 레이어를 물려받지 않게
    await renderer.xr.setSession(session);
  } catch (e) {
    say('XR 시작 실패: ' + e.message);
  }
}

function onSessionEnd() {
  renderer.setClearColor(0x2a2b2e, 1);
  floorGrid.visible = true;
  camera.layers.set(0);
  camera.layers.enable(previewEye);
}

// ---------------------------------------------------------------- 조작
let lastButtons = {};
function pollGamepads(dt) {
  const session = renderer.xr.getSession();
  if (!session) return;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;
    const ax = gp.axes.length >= 4 ? [gp.axes[2], gp.axes[3]] : [gp.axes[0] || 0, gp.axes[1] || 0];
    const dz = (v) => (Math.abs(v) < 0.15 ? 0 : v);
    if (src.handedness === 'right') {
      layout.dist = THREE.MathUtils.clamp(layout.dist - dz(ax[1]) * 1.5 * dt, 0.4, 15);
      layout.scale = THREE.MathUtils.clamp(layout.scale * (1 + dz(ax[0]) * 0.8 * dt), 0.2, 5);
    } else if (src.handedness === 'left') {
      layout.yaw -= dz(ax[0]) * 1.2 * dt;
      layout.lift = THREE.MathUtils.clamp(layout.lift - dz(ax[1]) * 0.8 * dt, -3, 3);
    }
    const key = src.handedness;
    const a = !!gp.buttons[4]?.pressed;
    const b = !!gp.buttons[5]?.pressed;
    const prev = lastButtons[key] || {};
    if (key === 'right' && a && !prev.a) togglePlay();
    if (key === 'right' && b && !prev.b) resetLayout();
    lastButtons[key] = { a, b };
  }
  applyLayout();
}

function togglePlay() {
  if (!video) return;
  if (video.paused) video.play(); else video.pause();
}

window.addEventListener('keydown', (e) => {
  const step = { ArrowUp: () => (layout.dist += 0.2), ArrowDown: () => (layout.dist = Math.max(0.4, layout.dist - 0.2)),
    ArrowRight: () => (layout.scale *= 1.05), ArrowLeft: () => (layout.scale /= 1.05),
    PageUp: () => (layout.lift += 0.05), PageDown: () => (layout.lift -= 0.05), ' ': togglePlay }[e.key];
  if (step) { step(); applyLayout(); e.preventDefault(); }
});

$('ar').onclick = () => startXR('immersive-ar');
$('vr').onclick = () => startXR('immersive-vr');
$('play').onclick = togglePlay;
$('reload').onclick = () => refreshList();
$('eye').onclick = () => {
  camera.layers.disable(previewEye);
  previewEye = previewEye === 1 ? 2 : 1;
  camera.layers.enable(previewEye);
  $('eye').textContent = '미리보기: ' + (previewEye === 1 ? '왼눈' : '오른눈');
};

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// AR 지원 여부 미리 확인 (버튼 문구)
if (navigator.xr) {
  navigator.xr.isSessionSupported('immersive-ar').then((ok) => { if (!ok) $('ar').title = '이 기기는 AR 세션을 지원하지 않습니다'; });
}

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  pollGamepads(Math.min(clock.getDelta(), 0.1));
  renderer.render(scene, camera);
});

refreshList();
