/**
 * popup.js — 팝업 UI 로직
 *
 * 탭 1 (순위 조회): 현재 페이지 또는 1~N페이지를 스캔해 내 상품 순위 표시
 *                   + 조회할 때마다 순위 기록 저장, 이전 조회 대비 변화(▲▼) 표시
 *                   + 페이지 경쟁 분석(가격/리뷰/광고 비중) + 결과 CSV 저장
 * 탭 2 (기록):      키워드×상품별 순위 히스토리, 스파크라인, CSV, 삭제
 * 탭 3 (내 상품):   쿠팡 API — 오늘 주문 현황, 등록상품 목록, 키워드 제안,
 *                   선택 상품 ID를 순위 조회에 자동 추가
 * 탭 4 (설정):      API 연결 상태·테스트, 데이터 현황
 */

const $ = (id) => document.getElementById(id);

const COUPANG_SEARCH_URL = 'https://www.coupang.com/np/search';
const HISTORY_KEY = 'rankHistory';
const HISTORY_MAX_PER_COMBO = 150; // 키워드×상품 조합당 보관 개수
const HISTORY_MAX_COMBOS = 300;    // 조합 수 상한 (초과 시 오래된 것부터 삭제)
const ORDER_CACHE_TTL = 5 * 60 * 1000; // 주문 현황 캐시 5분

// ============================================================
// 공용 헬퍼
// ============================================================
const storageGet = (keys) => new Promise((r) => chrome.storage.local.get(keys, r));
const storageSet = (obj) => new Promise((r) => chrome.storage.local.set(obj, r));

function setLoading(btn, loading) {
  btn.disabled = loading;
  const spinner = btn.querySelector('.spinner');
  if (spinner) spinner.hidden = !loading;
}

let toastTimer = null;
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2200);
}

function sendToBackground(msg) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, (res) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(res || { ok: false, error: '응답 없음' });
    });
  });
}

// 입력창 텍스트에서 상품ID 목록 추출 (중복 제거).
// 쿠팡 상품 URL을 통째로 붙여넣어도 주소에서 ID를 뽑아준다.
function parseIds(raw) {
  const ids = new Set();
  let rest = raw;

  // ① URL에서 추출: /vp/products/{productId} 우선, 없으면 itemId/vendorItemId 쿼리
  for (const url of raw.match(/https?:\/\/[^\s,;"']+/g) || []) {
    const m = url.match(/\/vp\/products\/(\d+)/);
    if (m) {
      ids.add(m[1]);
    } else {
      try {
        const qs = new URL(url).searchParams;
        const alt = qs.get('itemId') || qs.get('vendorItemId');
        if (alt) ids.add(alt);
      } catch (e) {
        // URL 파싱 실패는 무시
      }
    }
    rest = rest.replace(url, ' ');
  }

  // ② 나머지 토큰은 ID 직접 입력으로 처리
  rest.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean).forEach((s) => ids.add(s));

  return [...ids];
}

const fmtDateTime = new Intl.DateTimeFormat('ko-KR', {
  month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit',
});

function ymd() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`;
}

// ---------- CSV ----------
function toCsv(rows) {
  // 엑셀 한글 호환을 위해 UTF-8 BOM을 붙인다
  return '\ufeff' + rows.map((row) =>
    row.map((cell) => {
      const s = cell == null ? '' : String(cell);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    }).join(',')
  ).join('\r\n');
}

function downloadCsv(filename, rows) {
  const blob = new Blob([toCsv(rows)], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ============================================================
// 탭 전환 (기록/주문은 탭 진입 시 lazy 로드)
// ============================================================
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

let ordersLoadedOnce = false;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach((p) => {
    p.classList.toggle('active', p.id === 'tab-' + name);
  });
  if (name === 'history') renderHistoryTab();
  if (name === 'products' && apiConfigured && !ordersLoadedOnce) {
    ordersLoadedOnce = true;
    loadOrderSummary(false);
  }
}

$('goto-settings').addEventListener('click', () => switchTab('settings'));

// ============================================================
// API 연결 상태
// ============================================================
let apiConfigured = false;

async function refreshApiStatus() {
  const res = await sendToBackground({ type: 'API_STATUS' });
  const pill = $('api-pill');

  apiConfigured = !!(res.ok && res.configured);

  if (apiConfigured) {
    pill.textContent = 'API 연결됨';
    pill.className = 'pill pill-on';
    $('cfg-state').textContent =
      res.source === 'env' ? '✅ 키 설정됨 (env.json)' : '✅ 키 설정됨 (직접 입력)';
    $('cfg-vendor-row').hidden = false;
    $('cfg-vendor').textContent = res.vendorId;
    $('cfg-key-row').hidden = false;
    $('cfg-key').textContent = res.accessKeyMasked;
    $('test-btn').hidden = false;
    $('products-empty').hidden = true;
    $('products-main').hidden = false;
  } else {
    pill.textContent = 'API 미설정';
    pill.className = 'pill pill-off';
    $('cfg-state').textContent = '미설정 — 아래에 키를 입력하세요';
    $('products-empty').hidden = false;
    $('products-main').hidden = true;
  }
}

$('test-btn').addEventListener('click', async () => {
  const btn = $('test-btn');
  const out = $('test-result');
  setLoading(btn, true);
  out.hidden = true;

  const res = await sendToBackground({ type: 'API_TEST' });

  setLoading(btn, false);
  out.hidden = false;
  out.className = 'notice ' + (res.ok ? 'success' : 'error');
  out.textContent = res.ok ? res.message : res.error;
});

// ---------- 설정 탭: API 키 직접 입력 ----------
$('save-keys-btn').addEventListener('click', async () => {
  const btn = $('save-keys-btn');
  setLoading(btn, true);
  const res = await sendToBackground({
    type: 'API_SAVE_KEYS',
    accessKey: $('key-access').value,
    secretKey: $('key-secret').value,
    vendorId: $('key-vendor').value,
  });
  setLoading(btn, false);
  if (!res.ok) {
    toast(res.error);
    return;
  }
  $('key-secret').value = ''; // 저장 후 시크릿은 화면에서 지운다
  await refreshApiStatus();
  toast(res.source === 'env'
    ? '저장됨 (단, env.json이 있어 그 값이 우선 적용됩니다)'
    : 'API 키를 저장했습니다');
});

$('clear-keys-btn').addEventListener('click', async () => {
  const res = await sendToBackground({ type: 'API_CLEAR_KEYS' });
  if (!res.ok) {
    toast(res.error);
    return;
  }
  await refreshApiStatus();
  toast('저장된 키를 삭제했습니다');
});

// ---------- 설정 탭: 기록 백업 / 복원 ----------
$('backup-btn').addEventListener('click', async () => {
  const data = await storageGet([HISTORY_KEY, 'lastIds', 'scanDepth']);
  const payload = {
    app: '쿠팡 순위 파인더',
    backupVersion: 1,
    exportedAt: new Date().toISOString(),
    rankHistory: data[HISTORY_KEY] || {},
    lastIds: data.lastIds || '',
    scanDepth: data.scanDepth || '3',
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `쿠팡순위파인더_백업_${ymd()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('백업 파일을 저장했습니다');
});

$('restore-btn').addEventListener('click', () => $('restore-file').click());

$('restore-file').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    const incoming = payload.rankHistory;
    if (!incoming || typeof incoming !== 'object') {
      throw new Error('순위 기록(rankHistory)이 없는 파일입니다.');
    }
    // 기존 기록과 병합: 같은 시각(t) 항목은 중복 제거, 시간순 정렬
    const history = await getHistory();
    let mergedCombos = 0;
    for (const [key, arr] of Object.entries(incoming)) {
      if (!Array.isArray(arr)) continue;
      const existing = history[key] || [];
      const seen = new Set(existing.map((en) => en.t));
      const combined = existing.concat(
        arr.filter((en) => en && typeof en.t === 'number' && !seen.has(en.t))
      );
      combined.sort((a, b) => a.t - b.t);
      if (combined.length > HISTORY_MAX_PER_COMBO) {
        combined.splice(0, combined.length - HISTORY_MAX_PER_COMBO);
      }
      history[key] = combined;
      mergedCombos += 1;
    }
    await storageSet({ [HISTORY_KEY]: history });
    updateHistoryCount();
    toast(`백업에서 ${mergedCombos}개 조합을 불러왔습니다`);
  } catch (err) {
    toast('가져오기 실패: ' + err.message);
  }
});

// ============================================================
// 순위 기록 저장소
//
// 구조: { "<상품ID>|<키워드>": [ {t, position, organicRank, page, isAd,
//         price, reviewCount, rating, name, pagesScanned}, ... ] }
// position이 null이면 "스캔 범위 안에서 못 찾음"(순위권 밖) 기록.
// ============================================================
async function getHistory() {
  return (await storageGet([HISTORY_KEY]))[HISTORY_KEY] || {};
}

/**
 * 조회 결과를 기록하고, 상품ID별 "직전 대비 변화"를 돌려준다.
 * @param {string} keyword
 * @param {number} pagesScanned
 * @param {Array<{id: string, product: Object|null}>} results
 * @returns {Object} id → {kind: 'up'|'down'|'same'|'first'|'reenter'|'out'|'still-out', diff, prevPosition}
 */
async function recordScan(keyword, pagesScanned, results) {
  const history = await getHistory();
  const now = Date.now();
  const deltas = {};

  for (const { id, product } of results) {
    const key = `${id}|${keyword}`;
    const arr = history[key] || (history[key] = []);
    const prev = arr.length > 0 ? arr[arr.length - 1] : null;

    const cur = product ? product.position : null;
    if (!prev) {
      deltas[id] = { kind: 'first', diff: 0, prevPosition: null };
    } else if (prev.position != null && cur != null) {
      const diff = prev.position - cur; // 양수 = 순위 상승
      deltas[id] = {
        kind: diff > 0 ? 'up' : diff < 0 ? 'down' : 'same',
        diff: Math.abs(diff),
        prevPosition: prev.position,
      };
    } else if (prev.position == null && cur != null) {
      deltas[id] = { kind: 'reenter', diff: 0, prevPosition: null };
    } else if (prev.position != null && cur == null) {
      deltas[id] = { kind: 'out', diff: 0, prevPosition: prev.position };
    } else {
      deltas[id] = { kind: 'still-out', diff: 0, prevPosition: null };
    }

    arr.push({
      t: now,
      position: cur,
      organicRank: product ? product.organicRank : null,
      page: product ? product.page : null,
      isAd: product ? !!product.isAd : false,
      delivery: product ? product.delivery || null : null,
      price: product ? product.price : null,
      reviewCount: product ? product.reviewCount : null,
      rating: product ? product.rating : null,
      name: product ? product.name : null,
      pagesScanned,
    });
    if (arr.length > HISTORY_MAX_PER_COMBO) {
      arr.splice(0, arr.length - HISTORY_MAX_PER_COMBO);
    }
  }

  // 조합 수 상한: 마지막 기록이 오래된 조합부터 정리
  const keys = Object.keys(history);
  if (keys.length > HISTORY_MAX_COMBOS) {
    keys
      .sort((a, b) => history[a][history[a].length - 1].t - history[b][history[b].length - 1].t)
      .slice(0, keys.length - HISTORY_MAX_COMBOS)
      .forEach((k) => delete history[k]);
  }

  await storageSet({ [HISTORY_KEY]: history });
  updateHistoryCount();
  return deltas;
}

function deltaBadge(delta) {
  if (!delta) return null;
  const span = document.createElement('span');
  span.className = 'delta';
  switch (delta.kind) {
    case 'up': span.classList.add('up'); span.textContent = `▲${delta.diff}`; break;
    case 'down': span.classList.add('down'); span.textContent = `▼${delta.diff}`; break;
    case 'same': span.classList.add('same'); span.textContent = '–'; break;
    case 'first': span.classList.add('new'); span.textContent = '첫 기록'; break;
    case 'reenter': span.classList.add('up'); span.textContent = '재진입'; break;
    default: return null;
  }
  span.title = delta.prevPosition != null ? `직전 조회: ${delta.prevPosition}위` : '';
  return span;
}

async function updateHistoryCount() {
  const history = await getHistory();
  const combos = Object.keys(history).length;
  const entries = Object.values(history).reduce((s, a) => s + a.length, 0);
  $('cfg-history-count').textContent =
    combos === 0 ? '없음' : `${combos}개 조합 · ${entries}회 조회`;
}

// ============================================================
// 탭 1: 순위 조회
// ============================================================
const idsInput = $('product-ids');
let scanDepth = 3; // 기본 3페이지 — 1페이지만 보면 첫 조회가 대부분 "못 찾음"이 된다
let lastScan = null; // CSV 내보내기용 {keyword, pagesScanned, products, myIds}

storageGet(['lastIds', 'scanDepth']).then((res) => {
  if (res.lastIds) idsInput.value = res.lastIds;
  if (res.scanDepth) setDepth(parseInt(res.scanDepth, 10) || 3);
});

// ---------- 현재 탭 상태: 검색 페이지인지 먼저 보여준다 ----------
let activeTabId = null;

function checkCurrentTab() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs && tabs[0];
    const statusEl = $('tab-status');
    activeTabId = tab ? tab.id : null;

    if (tab && tab.url && tab.url.startsWith(COUPANG_SEARCH_URL)) {
      let keyword = '';
      try {
        keyword = new URL(tab.url).searchParams.get('q') || '';
      } catch (e) { /* 무시 */ }
      statusEl.className = 'tab-status ok';
      statusEl.textContent = `현재 탭: "${keyword}" 검색결과 ✓ 바로 조회할 수 있어요`;
      $('open-search-row').hidden = true;
      $('find-btn').disabled = false;
    } else {
      statusEl.className = 'tab-status warn';
      statusEl.textContent =
        '쿠팡 검색결과 페이지가 아니에요. 키워드를 입력하면 검색 탭을 열어드릴게요.';
      $('open-search-row').hidden = false;
      $('find-btn').disabled = true;
    }
  });
}

function openSearchTab() {
  const kw = $('keyword-input').value.trim();
  if (!kw) {
    toast('키워드를 입력해주세요');
    return;
  }
  chrome.tabs.create({ url: `${COUPANG_SEARCH_URL}?q=${encodeURIComponent(kw)}` });
}

$('open-search-btn').addEventListener('click', openSearchTab);
$('keyword-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') openSearchTab();
});

document.querySelectorAll('#depth-seg .seg').forEach((btn) => {
  btn.addEventListener('click', () => {
    setDepth(parseInt(btn.dataset.depth, 10));
    storageSet({ scanDepth: btn.dataset.depth });
  });
});

function setDepth(depth) {
  scanDepth = depth;
  document.querySelectorAll('#depth-seg .seg').forEach((b) => {
    b.classList.toggle('active', parseInt(b.dataset.depth, 10) === depth);
  });
}

// 다중 페이지 스캔 진행률 수신
chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === 'SCAN_PROGRESS') {
    $('progress-text').textContent = `${msg.current} / ${msg.max} 페이지 조회중…`;
    $('progress-bar').style.width = `${Math.round((msg.current / msg.max) * 100)}%`;
  }
});

$('find-btn').addEventListener('click', onFindRank);

function onFindRank() {
  clearRankOutput();

  const ids = parseIds(idsInput.value);
  if (ids.length === 0) {
    showRankStatus('상품ID를 하나 이상 입력해주세요.', true);
    return;
  }

  storageSet({ lastIds: idsInput.value });
  setLoading($('find-btn'), true);

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs && tabs[0];
    if (!tab || !tab.url || !tab.url.startsWith(COUPANG_SEARCH_URL)) {
      setLoading($('find-btn'), false);
      showRankStatus(
        '쿠팡 검색결과 페이지가 아닙니다.\n' +
        'coupang.com에서 키워드를 검색한 뒤 그 탭에서 다시 눌러주세요.',
        true
      );
      return;
    }

    const multi = scanDepth > 1;
    if (multi) {
      $('scan-progress').hidden = false;
      $('progress-text').textContent = '조회 시작…';
      $('progress-bar').style.width = '0%';
    }

    const message = multi
      ? { type: 'SCAN_PAGES', maxPages: scanDepth }
      : { type: 'PARSE_SEARCH_RESULTS' };

    chrome.tabs.sendMessage(tab.id, message, (response) => {
      setLoading($('find-btn'), false);
      $('scan-progress').hidden = true;

      if (chrome.runtime.lastError) {
        showRankStatus(
          '페이지와 연결하지 못했습니다.\n' +
          '검색결과 페이지를 새로고침(F5)한 뒤 다시 시도해주세요.',
          true
        );
        return;
      }

      if (!response || !response.ok) {
        showRankStatus('조회 중 오류: ' + (response ? response.error : '응답 없음'), true);
        return;
      }

      handleScanResult(response.data, ids, tab.id);
    });
  });
}

$('retry-deep-btn').addEventListener('click', () => {
  setDepth(10);
  storageSet({ scanDepth: '10' });
  onFindRank();
});

async function handleScanResult(data, myIds, tabId) {
  const { keyword, pagesScanned, itemsFound, products } = data;

  if (itemsFound === 0) {
    showRankStatus(
      '상품 목록을 찾지 못했습니다.\n' +
      '쿠팡 DOM 구조가 바뀌었을 수 있습니다. content.js 상단의 SELECTORS를 점검해주세요.',
      true
    );
    return;
  }

  const adCount = products.filter((p) => p.isAd).length;
  $('summary-row').hidden = false;
  $('summary').textContent =
    `"${keyword}" · ${pagesScanned}페이지 스캔 · ` +
    `상품 ${products.length}개 (광고 ${adCount}개)`;

  // 매칭 + 기록 저장 + 변화 계산
  const results = myIds.map((id) => ({
    id,
    product: products.find(
      (p) => p.productId === id || p.itemId === id || p.vendorItemId === id
    ) || null,
  }));

  const deltas = await recordScan(keyword || '(키워드 없음)', pagesScanned, results);

  results.forEach(({ id, product }) => {
    $('results').appendChild(resultCard(id, product, deltas[id]));
  });

  // 경쟁 분석
  renderAnalysis(products, results.filter((r) => r.product).map((r) => r.product));

  // 파싱 검증용 전체 목록
  $('all-products-wrap').hidden = false;
  const allList = $('all-products');
  allList.textContent = '';
  products.forEach((p) => {
    const li = document.createElement('li');
    li.textContent =
      `${p.position}위${p.isAd ? '[AD]' : ''} [${p.page}p] ` +
      `(${p.productId || '?'}) ${p.name || '(상품명 없음)'}`;
    allList.appendChild(li);
  });

  lastScan = { keyword, pagesScanned, products, myIds: new Set(myIds) };

  // 검색결과 페이지 위에 내 상품 하이라이트 (현재 열린 페이지에 있는 것만 표시됨)
  const targets = results
    .filter((r) => r.product)
    .map((r) => ({
      ids: [r.id],
      label: r.product.isAd ? `광고 ${r.product.position}위` : `${r.product.position}위`,
    }));
  if (tabId != null && targets.length > 0) {
    chrome.tabs.sendMessage(tabId, { type: 'HIGHLIGHT_MY_PRODUCTS', targets }, () => {
      void chrome.runtime.lastError; // 실패해도 조회 결과에는 영향 없음
    });
  }

  // 못 찾은 상품이 있으면 더 깊게 재조회 제안
  const anyMissing = results.some((r) => !r.product);
  $('retry-deep-btn').hidden = !(anyMissing && pagesScanned < 10);

  toast('조회 완료 · 기록 탭에 저장됨');
}

function resultCard(id, p, delta) {
  const li = document.createElement('li');
  li.className = 'result-card';

  if (!p) {
    li.classList.add('not-found');
    let text = `상품ID ${id} — 스캔 범위에서 찾지 못했습니다.`;
    if (delta && delta.kind === 'out' && delta.prevPosition != null) {
      text += ` (직전 조회 ${delta.prevPosition}위 → 순위권 밖)`;
    } else {
      text += ' 조회 범위를 늘려보세요.';
    }
    li.textContent = text;
    return li;
  }

  const rankBox = document.createElement('div');
  rankBox.className = 'rank-box';
  const rankNum = document.createElement('span');
  rankNum.className = 'rank-num';
  rankNum.textContent = p.position;
  const rankUnit = document.createElement('span');
  rankUnit.className = 'rank-unit';
  rankUnit.textContent = '위';
  rankBox.appendChild(rankNum);
  rankBox.appendChild(rankUnit);

  const body = document.createElement('div');
  body.className = 'result-body';

  const badge = document.createElement('span');
  badge.className = 'badge ' + (p.isAd ? 'ad' : 'organic');
  badge.textContent = p.isAd ? '광고' : `일반 ${p.organicRank}위`;
  body.appendChild(badge);

  if (p.delivery) {
    const rocket = document.createElement('span');
    rocket.className = 'badge rocket';
    rocket.textContent = p.delivery;
    body.appendChild(rocket);
  }

  const db = deltaBadge(delta);
  if (db) body.appendChild(db);

  const name = document.createElement('div');
  name.className = 'product-name';
  name.textContent = p.name || '(상품명 파싱 실패)';

  const meta = document.createElement('div');
  meta.className = 'meta';
  const parts = [`${p.page}페이지`, `ID ${id}`];
  if (p.price != null) parts.push(`${p.price.toLocaleString()}원`);
  if (p.rating) parts.push(`★${p.rating}`);
  if (p.reviewCount != null) parts.push(`리뷰 ${p.reviewCount.toLocaleString()}`);
  meta.textContent = parts.join(' · ');

  body.appendChild(name);
  body.appendChild(meta);
  li.appendChild(rankBox);
  li.appendChild(body);
  return li;
}

function showRankStatus(text, isError) {
  const el = $('status');
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle('error', !!isError);
}

function clearRankOutput() {
  $('status').hidden = true;
  $('summary-row').hidden = true;
  $('results').textContent = '';
  $('retry-deep-btn').hidden = true;
  $('analysis-wrap').hidden = true;
  $('analysis-body').textContent = '';
  $('all-products-wrap').hidden = true;
  $('all-products').textContent = '';
  $('scan-progress').hidden = true;
  lastScan = null;
}

// ---------- 결과 CSV ----------
$('csv-results-btn').addEventListener('click', () => {
  if (!lastScan) return;
  const rows = [[
    '순위', '일반순위', '광고', '배송', '페이지', 'productId', 'itemId', 'vendorItemId',
    '상품명', '가격', '평점', '리뷰수', '내상품',
  ]];
  lastScan.products.forEach((p) => {
    const mine =
      lastScan.myIds.has(p.productId) || lastScan.myIds.has(p.itemId) ||
      lastScan.myIds.has(p.vendorItemId);
    rows.push([
      p.position, p.organicRank, p.isAd ? 'Y' : '', p.delivery, p.page,
      p.productId, p.itemId, p.vendorItemId,
      p.name, p.price, p.rating, p.reviewCount, mine ? 'Y' : '',
    ]);
  });
  downloadCsv(`쿠팡검색_${lastScan.keyword || '결과'}_${ymd()}.csv`, rows);
  toast('CSV 파일을 저장했습니다');
});

// ============================================================
// 페이지 경쟁 분석
// ============================================================
function median(sorted) {
  const n = sorted.length;
  if (n === 0) return null;
  return n % 2 ? sorted[(n - 1) / 2] : Math.round((sorted[n / 2 - 1] + sorted[n / 2]) / 2);
}

function renderAnalysis(products, myProducts) {
  const body = $('analysis-body');
  body.textContent = '';
  if (products.length === 0) return;

  const prices = products.map((p) => p.price).filter((v) => typeof v === 'number').sort((a, b) => a - b);
  const reviews = products.map((p) => p.reviewCount).filter((v) => typeof v === 'number');
  const ratings = products.map((p) => parseFloat(p.rating)).filter((v) => !isNaN(v));
  const adCount = products.filter((p) => p.isAd).length;

  const avg = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null);
  const avgPrice = avg(prices);
  const avgReviews = avg(reviews);
  const avgRating = avg(ratings);

  // 요약 타일
  const grid = document.createElement('div');
  grid.className = 'stat-grid';
  const tiles = [
    [`${products.length}개`, '스캔 상품'],
    [`${Math.round((adCount / products.length) * 100)}%`, '광고 비중'],
    [avgRating != null ? `★${avgRating.toFixed(1)}` : '-', '평균 평점'],
    [avgPrice != null ? `${Math.round(avgPrice).toLocaleString()}원` : '-', '평균가'],
    [prices.length ? `${median(prices).toLocaleString()}원` : '-', '중앙값 가격'],
    [avgReviews != null ? Math.round(avgReviews).toLocaleString() : '-', '평균 리뷰수'],
  ];
  tiles.forEach(([value, label]) => {
    const tile = document.createElement('div');
    tile.className = 'stat-tile';
    const v = document.createElement('div');
    v.className = 'stat-value';
    v.textContent = value;
    const l = document.createElement('div');
    l.className = 'stat-label';
    l.textContent = label;
    tile.appendChild(v);
    tile.appendChild(l);
    grid.appendChild(tile);
  });
  body.appendChild(grid);

  // 가격 분포 히스토그램 (6구간)
  if (prices.length >= 4) {
    const min = prices[0];
    const max = prices[prices.length - 1];
    if (max > min) {
      const BUCKETS = 6;
      const counts = new Array(BUCKETS).fill(0);
      prices.forEach((v) => {
        const i = Math.min(BUCKETS - 1, Math.floor(((v - min) / (max - min)) * BUCKETS));
        counts[i] += 1;
      });
      const peak = Math.max(...counts);

      const title = document.createElement('div');
      title.className = 'hist-title';
      title.textContent = '가격 분포';
      body.appendChild(title);

      const bars = document.createElement('div');
      bars.className = 'hist-bars';
      counts.forEach((c) => {
        const bar = document.createElement('div');
        bar.className = 'hist-bar';
        bar.title = `${c}개`;
        const fill = document.createElement('div');
        fill.className = 'fill';
        fill.style.height = `${peak ? Math.round((c / peak) * 100) : 0}%`;
        bar.appendChild(fill);
        bars.appendChild(bar);
      });
      body.appendChild(bars);

      const labels = document.createElement('div');
      labels.className = 'hist-labels';
      const lo = document.createElement('span');
      lo.textContent = `${min.toLocaleString()}원`;
      const hi = document.createElement('span');
      hi.textContent = `${max.toLocaleString()}원`;
      labels.appendChild(lo);
      labels.appendChild(hi);
      body.appendChild(labels);
    }
  }

  // 내 상품 인사이트
  myProducts.forEach((mine) => {
    const insight = document.createElement('div');
    insight.className = 'my-insight';

    const lines = [];
    const shortName = (mine.name || `ID ${mine.productId || ''}`).slice(0, 22);
    lines.push(`「${shortName}…」`);

    if (typeof mine.price === 'number' && prices.length > 1 && avgPrice) {
      const diffPct = Math.round(((avgPrice - mine.price) / avgPrice) * 100);
      const cheaperCount = prices.filter((v) => v > mine.price).length;
      const pricePct = Math.round((cheaperCount / prices.length) * 100);
      lines.push(
        diffPct >= 0
          ? `가격: 평균보다 ${diffPct}% 저렴 (페이지에서 저렴한 편 상위 ${100 - pricePct}%)`
          : `가격: 평균보다 ${-diffPct}% 비쌈`
      );
    }
    if (typeof mine.reviewCount === 'number' && reviews.length > 1) {
      const moreCount = reviews.filter((v) => v > mine.reviewCount).length;
      const pct = Math.max(1, Math.round(((moreCount + 1) / reviews.length) * 100));
      lines.push(`리뷰수: 페이지 내 상위 ${pct}%`);
    }
    if (lines.length === 1) lines.push('가격/리뷰 데이터가 부족해 비교를 생략했습니다.');

    lines.forEach((text, i) => {
      const div = document.createElement('div');
      if (i === 0) {
        const b = document.createElement('b');
        b.textContent = text;
        div.appendChild(b);
      } else {
        div.textContent = '· ' + text;
      }
      insight.appendChild(div);
    });
    body.appendChild(insight);
  });

  $('analysis-wrap').hidden = false;
  $('analysis-wrap').open = true;
}

// ============================================================
// 탭 2: 순위 기록
// ============================================================
function sparkline(entries) {
  const pts = entries.filter((e) => e.position != null).slice(-30);
  if (pts.length < 2) return null;

  const NS = 'http://www.w3.org/2000/svg';
  const W = 110, H = 30, PAD = 3;
  const values = pts.map((p) => p.position);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const X = (i) => PAD + (W - 2 * PAD) * (i / (pts.length - 1));
  // 순위가 낮을수록(좋을수록) 위쪽에 그려지도록 y축을 그대로 사용
  const Y = (v) => PAD + (H - 2 * PAD) * ((v - min) / span);

  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('class', 'sparkline');

  const path = document.createElementNS(NS, 'path');
  path.setAttribute('d', pts.map((p, i) =>
    (i ? 'L' : 'M') + X(i).toFixed(1) + ',' + Y(p.position).toFixed(1)
  ).join(' '));
  svg.appendChild(path);

  const dot = document.createElementNS(NS, 'circle');
  dot.setAttribute('cx', X(pts.length - 1).toFixed(1));
  dot.setAttribute('cy', Y(pts[pts.length - 1].position).toFixed(1));
  dot.setAttribute('r', '2.5');
  svg.appendChild(dot);

  return svg;
}

async function renderHistoryTab() {
  const history = await getHistory();
  const keys = Object.keys(history).filter((k) => history[k].length > 0);
  const list = $('history-list');
  list.textContent = '';

  $('history-empty').hidden = keys.length > 0;
  if (keys.length === 0) return;

  // 최근에 조회한 조합부터
  keys.sort((a, b) =>
    history[b][history[b].length - 1].t - history[a][history[a].length - 1].t
  );

  keys.forEach((key) => {
    const sep = key.indexOf('|');
    const id = key.slice(0, sep);
    const keyword = key.slice(sep + 1);
    const entries = history[key];
    const latest = entries[entries.length - 1];
    const prev = entries.length > 1 ? entries[entries.length - 2] : null;

    const li = document.createElement('li');
    li.className = 'history-card';

    // 헤더: 키워드 / 상품명 / 삭제
    const head = document.createElement('div');
    head.className = 'row-between';
    const headLeft = document.createElement('div');
    const kw = document.createElement('div');
    kw.className = 'h-keyword';
    kw.textContent = `"${keyword}"`;
    const nm = document.createElement('div');
    nm.className = 'h-name';
    const lastName = [...entries].reverse().find((e) => e.name);
    nm.textContent = (lastName ? lastName.name + ' · ' : '') + `ID ${id}`;
    headLeft.appendChild(kw);
    headLeft.appendChild(nm);

    const del = document.createElement('button');
    del.className = 'icon-btn';
    del.textContent = '✕';
    del.title = '이 기록 삭제';
    del.addEventListener('click', async () => {
      const h = await getHistory();
      delete h[key];
      await storageSet({ [HISTORY_KEY]: h });
      renderHistoryTab();
      updateHistoryCount();
      toast('기록을 삭제했습니다');
    });

    head.appendChild(headLeft);
    head.appendChild(del);
    li.appendChild(head);

    // 본문: 현재 순위 + 델타 + 스파크라인
    const bodyRow = document.createElement('div');
    bodyRow.className = 'h-body';

    const rank = document.createElement('div');
    rank.className = 'h-rank';
    if (latest.position != null) {
      rank.textContent = `${latest.position}위`;
      if (prev && prev.position != null) {
        const diff = prev.position - latest.position;
        const d = deltaBadge({
          kind: diff > 0 ? 'up' : diff < 0 ? 'down' : 'same',
          diff: Math.abs(diff),
          prevPosition: prev.position,
        });
        if (d) rank.appendChild(d);
      }
    } else {
      const out = document.createElement('span');
      out.className = 'out';
      out.textContent = '순위권 밖';
      rank.appendChild(out);
    }
    bodyRow.appendChild(rank);

    const spark = sparkline(entries);
    if (spark) bodyRow.appendChild(spark);
    li.appendChild(bodyRow);

    // 최근 기록 상세 (최근 10회)
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `최근 기록 ${Math.min(entries.length, 10)}회 보기 (총 ${entries.length}회)`;
    details.appendChild(summary);

    const ul = document.createElement('ul');
    ul.className = 'h-entries';
    [...entries].slice(-10).reverse().forEach((e) => {
      const row = document.createElement('li');
      const when = document.createElement('span');
      when.textContent = fmtDateTime.format(new Date(e.t));
      const what = document.createElement('span');
      what.textContent = e.position != null
        ? `${e.position}위 (${e.page || '?'}p${e.isAd ? '·광고' : ''}, ${e.pagesScanned}p 스캔)`
        : `순위권 밖 (${e.pagesScanned}p 스캔)`;
      row.appendChild(when);
      row.appendChild(what);
      ul.appendChild(row);
    });
    details.appendChild(ul);
    li.appendChild(details);

    list.appendChild(li);
  });
}

// ---------- 기록 CSV / 전체 삭제 ----------
$('history-csv-btn').addEventListener('click', async () => {
  const history = await getHistory();
  const keys = Object.keys(history);
  if (keys.length === 0) {
    toast('내보낼 기록이 없습니다');
    return;
  }
  const rows = [[
    '일시', '키워드', '상품ID', '순위', '일반순위', '페이지', '광고',
    '가격', '평점', '리뷰수', '스캔페이지수', '상품명',
  ]];
  keys.forEach((key) => {
    const sep = key.indexOf('|');
    const id = key.slice(0, sep);
    const keyword = key.slice(sep + 1);
    history[key].forEach((e) => {
      rows.push([
        new Date(e.t).toLocaleString('ko-KR'), keyword, id,
        e.position ?? '순위권밖', e.organicRank, e.page, e.isAd ? 'Y' : '',
        e.price, e.rating, e.reviewCount, e.pagesScanned, e.name,
      ]);
    });
  });
  downloadCsv(`쿠팡순위기록_${ymd()}.csv`, rows);
  toast('CSV 파일을 저장했습니다');
});

$('history-clear-btn').addEventListener('click', async () => {
  if (!confirm('순위 기록을 전부 삭제할까요? 되돌릴 수 없습니다.')) return;
  await storageSet({ [HISTORY_KEY]: {} });
  renderHistoryTab();
  updateHistoryCount();
  toast('기록을 모두 삭제했습니다');
});

// ============================================================
// 탭 3: 내 상품 — 오늘 주문 현황
// ============================================================
$('orders-refresh-btn').addEventListener('click', () => loadOrderSummary(true));

async function loadOrderSummary(force) {
  if (!apiConfigured) return;

  // 캐시(5분) — 팝업을 열 때마다 API를 두드리지 않는다
  if (!force) {
    const { orderCache } = await storageGet(['orderCache']);
    if (orderCache && Date.now() - orderCache.t < ORDER_CACHE_TTL) {
      renderOrderSummary(orderCache.summary, orderCache.t, true);
      return;
    }
  }

  const btn = $('orders-refresh-btn');
  setLoading(btn, true);
  $('order-error').hidden = true;
  if ($('order-stats').hidden) $('order-skeleton').hidden = false;

  const res = await sendToBackground({ type: 'API_ORDER_SUMMARY' });

  setLoading(btn, false);
  $('order-skeleton').hidden = true;

  if (!res.ok) {
    $('order-error').hidden = false;
    $('order-error').textContent = res.error;
    return;
  }

  const now = Date.now();
  await storageSet({ orderCache: { t: now, summary: res.summary } });
  renderOrderSummary(res.summary, now, false);
}

function renderOrderSummary(summary, fetchedAt, fromCache) {
  const stats = $('order-stats');
  stats.textContent = '';
  stats.hidden = false;

  const totalText = summary.total + (summary.hasMore ? '+' : '');
  const tiles = [
    [`${totalText}건`, '총 주문'],
    [`${Math.round(summary.revenue).toLocaleString()}원`, '예상 매출'],
    [summary.date.slice(5).replace('-', '/'), '기준일(오늘)'],
  ];
  tiles.forEach(([value, label]) => {
    const tile = document.createElement('div');
    tile.className = 'stat-tile';
    const v = document.createElement('div');
    v.className = 'stat-value';
    v.textContent = value;
    const l = document.createElement('div');
    l.className = 'stat-label';
    l.textContent = label;
    tile.appendChild(v);
    tile.appendChild(l);
    stats.appendChild(tile);
  });

  const chips = $('order-status-chips');
  chips.textContent = '';
  const labels = summary.labels || {};
  const nonZero = Object.entries(summary.counts || {}).filter(([, c]) => c > 0);
  chips.hidden = nonZero.length === 0;
  nonZero.forEach(([status, count]) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const b = document.createElement('b');
    b.textContent = String(count);
    chip.textContent = (labels[status] || status) + ' ';
    chip.appendChild(b);
    chips.appendChild(chip);
  });

  const updated = $('order-updated');
  updated.hidden = false;
  updated.textContent =
    `${fmtDateTime.format(new Date(fetchedAt))} 기준` +
    (fromCache ? ' · 5분 캐시 (새로고침으로 갱신)' : '');
}

// ============================================================
// 탭 3: 내 상품 목록 + 키워드 제안
// ============================================================
let nextToken = null;
const loadedProducts = new Map(); // sellerProductId → 상품 요약

$('load-products-btn').addEventListener('click', () => loadProducts(true));
$('load-more-btn').addEventListener('click', () => loadProducts(false));

async function loadProducts(reset) {
  const btn = reset ? $('load-products-btn') : $('load-more-btn');
  setLoading(btn, true);
  $('products-error').hidden = true;

  if (reset) {
    nextToken = null;
    loadedProducts.clear();
    $('product-list').textContent = '';
    $('load-more-btn').hidden = true;
    $('add-selected-wrap').hidden = true;
    $('products-skeleton').hidden = false;
  }

  const res = await sendToBackground({
    type: 'API_LIST_PRODUCTS',
    nextToken: nextToken || undefined,
  });

  setLoading(btn, false);
  $('products-skeleton').hidden = true;

  if (!res.ok) {
    $('products-error').hidden = false;
    $('products-error').className = 'notice error';
    $('products-error').textContent = res.error;
    return;
  }

  nextToken = res.nextToken;
  $('load-more-btn').hidden = !nextToken;

  const list = $('product-list');
  (res.products || []).forEach((p) => {
    // 응답 필드명이 문서와 다를 수 있어 알려진 이름을 순서대로 시도한다
    const spId = p.sellerProductId ?? p.sellerProductID ?? p.id;
    if (spId == null || loadedProducts.has(String(spId))) return;
    loadedProducts.set(String(spId), p);
    list.appendChild(productRow(String(spId), p));
  });

  if (loadedProducts.size === 0) {
    $('products-error').hidden = false;
    $('products-error').className = 'notice';
    $('products-error').textContent = '등록된 상품이 없습니다.';
  } else {
    $('add-selected-wrap').hidden = false;
  }
}

function productRow(spId, p) {
  const li = document.createElement('li');
  li.className = 'product-row';

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.value = spId;

  const body = document.createElement('div');
  body.style.flex = '1';
  body.style.minWidth = '0';

  const productName = p.sellerProductName || p.productName || '(이름 없음)';

  const name = document.createElement('div');
  name.className = 'p-name';
  name.textContent = productName;

  const statusName = p.statusName || p.status || '';
  if (statusName) {
    const chip = document.createElement('span');
    chip.className = 'status-chip';
    if (/승인|판매중|approved/i.test(statusName)) chip.classList.add('approved');
    if (/반려|거부|rejected/i.test(statusName)) chip.classList.add('rejected');
    chip.textContent = statusName;
    name.appendChild(chip);
  }

  const meta = document.createElement('div');
  meta.className = 'p-meta';
  const parts = [`등록상품ID ${spId}`];
  if (p.brand) parts.push(p.brand);
  meta.textContent = parts.join(' · ');

  body.appendChild(name);
  body.appendChild(meta);

  // 키워드 제안 토글
  const keywords = suggestKeywords(productName);
  if (keywords.length > 0) {
    const kwBtn = document.createElement('button');
    kwBtn.className = 'kw-toggle-btn';
    kwBtn.textContent = '🔍 키워드 제안';

    const chipsWrap = document.createElement('div');
    chipsWrap.className = 'kw-chips';
    chipsWrap.hidden = true;

    keywords.forEach((kw) => {
      const chip = document.createElement('span');
      chip.className = 'chip clickable';
      chip.textContent = kw;
      chip.title = `쿠팡에서 "${kw}" 검색 열기`;
      chip.addEventListener('click', () => {
        chrome.tabs.create({
          url: `${COUPANG_SEARCH_URL}?q=${encodeURIComponent(kw)}`,
        });
      });
      chipsWrap.appendChild(chip);
    });

    kwBtn.addEventListener('click', () => {
      chipsWrap.hidden = !chipsWrap.hidden;
      kwBtn.textContent = chipsWrap.hidden ? '🔍 키워드 제안' : '🔍 키워드 접기';
    });

    body.appendChild(kwBtn);
    body.appendChild(chipsWrap);
  }

  li.appendChild(checkbox);
  li.appendChild(body);

  // 행 아무데나 눌러도 체크 토글 (버튼/칩 클릭은 제외)
  li.addEventListener('click', (e) => {
    if (e.target === checkbox) return;
    if (e.target.closest('button') || e.target.closest('.chip')) return;
    checkbox.checked = !checkbox.checked;
  });

  return li;
}

/**
 * 상품명에서 검색 키워드 후보 추출.
 * 스펙성 토큰(숫자 시작, 1글자, 홍보 문구)을 걸러내고
 * 인접 2단어 조합(구체적 키워드) + 단일 단어를 섞어 최대 8개.
 */
const KW_STOPWORDS = new Set([
  '무료배송', '당일발송', '당일출고', '로켓배송', '정품', '공식', '국내',
  '국산', '해외', '수입', '대용량', '가성비', '인기', '추천', '신상',
  '신상품', 'best', '세트', '증정', '사은품', '옵션', '택1', '모음', '할인',
]);

function suggestKeywords(name) {
  if (!name) return [];
  const cleaned = name.replace(/[[\](){}/,+~!·×@#&*"'’‘”“]/g, ' ');
  const tokens = cleaned.split(/\s+/).filter((t) => {
    if (t.length < 2 || t.length > 12) return false;
    if (/^\d/.test(t)) return false;               // 100ml, 3개입 같은 스펙
    if (/^[A-Za-z]{1,2}$/.test(t)) return false;   // 의미 없는 짧은 영문
    if (KW_STOPWORDS.has(t.toLowerCase())) return false;
    return true;
  });

  const bigrams = [];
  for (let i = 0; i < tokens.length - 1 && bigrams.length < 4; i++) {
    const bg = tokens[i] + ' ' + tokens[i + 1];
    if (!bigrams.includes(bg)) bigrams.push(bg);
  }
  const singles = [...new Set(tokens)].slice(0, 5);

  return [...new Set([...bigrams, ...singles])].slice(0, 8);
}

// ------------------------------------------------------------
// 선택한 상품 → 순위 조회 입력창에 ID 추가
// 목록 응답에는 검색 매칭용 ID(productId/vendorItemId)가 없을 수 있어서
// 상품 상세를 조회해 ID를 뽑는다.
// ------------------------------------------------------------
$('add-selected-btn').addEventListener('click', async () => {
  const checked = [...document.querySelectorAll('.product-row input:checked')];
  if (checked.length === 0) {
    toast('상품을 먼저 선택해주세요');
    return;
  }

  const btn = $('add-selected-btn');
  setLoading(btn, true);

  const collected = [];
  for (const cb of checked) {
    const spId = cb.value;
    const res = await sendToBackground({ type: 'API_GET_PRODUCT', sellerProductId: spId });
    if (res.ok && res.product) {
      const ids = extractMatchIds(res.product);
      if (ids.length > 0) {
        collected.push(...ids);
        continue;
      }
    }
    // 상세 조회 실패 시 목록에 있던 productId라도 시도
    const summary = loadedProducts.get(spId);
    if (summary && summary.productId) collected.push(String(summary.productId));
  }

  setLoading(btn, false);

  if (collected.length === 0) {
    toast('매칭용 상품ID를 찾지 못했습니다');
    return;
  }

  const merged = [...new Set([...parseIds(idsInput.value), ...collected])];
  idsInput.value = merged.join(', ');
  storageSet({ lastIds: idsInput.value });

  switchTab('rank');
  toast(`상품ID ${collected.length}개를 추가했습니다`);
});

/**
 * 상품 상세 응답에서 검색결과 매칭에 쓸 ID를 재귀적으로 수집.
 * 쿠팡 응답 스키마가 유동적이라 키 이름 기준으로 방어적으로 찾는다.
 * 우선순위: productId > vendorItemId > itemId
 */
function extractMatchIds(obj) {
  const found = { productId: new Set(), vendorItemId: new Set(), itemId: new Set() };

  (function walk(node) {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    for (const [key, val] of Object.entries(node)) {
      if (val && typeof val === 'object') {
        walk(val);
      } else if (val != null && /^(string|number)$/.test(typeof val)) {
        if (key === 'productId') found.productId.add(String(val));
        else if (key === 'vendorItemId') found.vendorItemId.add(String(val));
        else if (key === 'itemId') found.itemId.add(String(val));
      }
    }
  })(obj);

  if (found.productId.size > 0) return [...found.productId];
  if (found.vendorItemId.size > 0) return [...found.vendorItemId];
  return [...found.itemId];
}

// ============================================================
// 초기화
// ============================================================
refreshApiStatus();
updateHistoryCount();
checkCurrentTab();
