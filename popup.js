/**
 * popup.js — 팝업 UI 로직
 *
 * 탭 1 (순위 조회): 콘텐츠 스크립트에 파싱 요청 → 내 상품ID 순위 표시
 * 탭 2 (내 상품):   백그라운드 경유로 쿠팡 API 호출 → 등록상품 목록,
 *                   선택한 상품의 ID를 순위 조회 입력창에 자동 추가
 * 탭 3 (설정):      API 연결 상태 확인, 연결 테스트, 키 설정 가이드
 */

const $ = (id) => document.getElementById(id);

const COUPANG_SEARCH_URL = 'https://www.coupang.com/np/search';

// ============================================================
// 탭 전환
// ============================================================
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach((p) => {
    p.classList.toggle('active', p.id === 'tab-' + name);
  });
}

$('goto-settings').addEventListener('click', () => switchTab('settings'));

// ============================================================
// 공용 헬퍼
// ============================================================
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

// 백그라운드에 메시지 보내고 Promise로 받기
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

// ============================================================
// API 연결 상태 (헤더 pill + 설정 탭 + 내 상품 탭 초기화)
// ============================================================
let apiConfigured = false;

async function refreshApiStatus() {
  const res = await sendToBackground({ type: 'API_STATUS' });
  const pill = $('api-pill');

  apiConfigured = !!(res.ok && res.configured);

  if (apiConfigured) {
    pill.textContent = 'API 연결됨';
    pill.className = 'pill pill-on';
    $('cfg-state').textContent = '✅ 키 설정됨';
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
    $('cfg-state').textContent = '미설정 (env.json 없음)';
    $('products-empty').hidden = false;
    $('products-main').hidden = true;
  }
}

refreshApiStatus();

// ============================================================
// 탭 3: 연결 테스트
// ============================================================
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

// ============================================================
// 탭 1: 순위 조회
// ============================================================
const idsInput = $('product-ids');

chrome.storage?.local?.get?.(['lastIds'], (res) => {
  if (res && res.lastIds) idsInput.value = res.lastIds;
});

$('find-btn').addEventListener('click', onFindRank);

function onFindRank() {
  clearRankOutput();

  const ids = parseIds(idsInput.value);
  if (ids.length === 0) {
    showRankStatus('상품ID를 하나 이상 입력해주세요.', true);
    return;
  }

  chrome.storage?.local?.set?.({ lastIds: idsInput.value });
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

    chrome.tabs.sendMessage(tab.id, { type: 'PARSE_SEARCH_RESULTS' }, (response) => {
      setLoading($('find-btn'), false);

      if (chrome.runtime.lastError) {
        showRankStatus(
          '페이지와 연결하지 못했습니다.\n' +
          '검색결과 페이지를 새로고침(F5)한 뒤 다시 시도해주세요.',
          true
        );
        return;
      }

      if (!response || !response.ok) {
        showRankStatus('파싱 중 오류: ' + (response ? response.error : '응답 없음'), true);
        return;
      }

      renderRank(response.data, ids);
    });
  });
}

// "123, 456\n789" → ['123', '456', '789']
function parseIds(raw) {
  return [...new Set(
    raw.split(/[\s,;]+/).map((s) => s.trim()).filter((s) => s.length > 0)
  )];
}

function renderRank(data, myIds) {
  const { keyword, page, itemsFound, products } = data;

  if (itemsFound === 0) {
    showRankStatus(
      '상품 목록을 찾지 못했습니다.\n' +
      '쿠팡 DOM 구조가 바뀌었을 수 있습니다. content.js 상단의 SELECTORS를 점검해주세요.',
      true
    );
    return;
  }

  const adCount = products.filter((p) => p.isAd).length;
  $('summary').hidden = false;
  $('summary').textContent =
    `키워드 "${keyword}" · ${page}페이지 · 상품 ${products.length}개 파싱 (광고 ${adCount}개)`;

  myIds.forEach((id) => {
    const found = products.find(
      (p) => p.productId === id || p.itemId === id || p.vendorItemId === id
    );
    $('results').appendChild(resultCard(id, found));
  });

  // 파싱 검증용 전체 목록
  $('all-products-wrap').hidden = false;
  const allList = $('all-products');
  allList.textContent = '';
  products.forEach((p) => {
    const li = document.createElement('li');
    li.textContent =
      `${p.position}위${p.isAd ? '[AD]' : ''} (${p.productId || '?'}) ${p.name || '(상품명 없음)'}`;
    allList.appendChild(li);
  });
}

function resultCard(id, p) {
  const li = document.createElement('li');
  li.className = 'result-card';

  if (!p) {
    li.classList.add('not-found');
    li.textContent = `상품ID ${id} — 이 페이지에는 없습니다. 다음 페이지에서 다시 조회해보세요.`;
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

  const name = document.createElement('div');
  name.className = 'product-name';
  name.textContent = p.name || '(상품명 파싱 실패)';

  const meta = document.createElement('div');
  meta.className = 'meta';
  const parts = [`ID ${id}`];
  if (p.price != null) parts.push(`${p.price.toLocaleString()}원`);
  if (p.rating) parts.push(`★${p.rating}`);
  if (p.reviewCount != null) parts.push(`리뷰 ${p.reviewCount.toLocaleString()}`);
  meta.textContent = parts.join(' · ');

  body.appendChild(badge);
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
  $('summary').hidden = true;
  $('results').textContent = '';
  $('all-products-wrap').hidden = true;
  $('all-products').textContent = '';
}

// ============================================================
// 탭 2: 내 상품 (쿠팡 API)
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

  const name = document.createElement('div');
  name.className = 'p-name';
  name.textContent = p.sellerProductName || p.productName || '(이름 없음)';

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
  li.appendChild(checkbox);
  li.appendChild(body);

  // 행 아무데나 눌러도 체크 토글
  li.addEventListener('click', (e) => {
    if (e.target !== checkbox) checkbox.checked = !checkbox.checked;
  });

  return li;
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

  const existing = parseIds(idsInput.value);
  const merged = [...new Set([...existing, ...collected])];
  idsInput.value = merged.join(', ');
  chrome.storage?.local?.set?.({ lastIds: idsInput.value });

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
