/**
 * popup.js — 팝업 UI 로직
 *
 * 1) 입력창의 상품ID들을 파싱하고
 * 2) 현재 활성 탭(쿠팡 검색결과)에 PARSE_SEARCH_RESULTS 메시지를 보내
 * 3) 돌아온 상품 목록에서 내 상품ID를 찾아 순위를 표시한다.
 */

const idsInput = document.getElementById('product-ids');
const findBtn = document.getElementById('find-btn');
const statusEl = document.getElementById('status');
const summaryEl = document.getElementById('summary');
const resultsEl = document.getElementById('results');
const allWrapEl = document.getElementById('all-products-wrap');
const allListEl = document.getElementById('all-products');

const COUPANG_SEARCH_URL = 'https://www.coupang.com/np/search';

// 마지막으로 입력한 상품ID를 기억해두면 매번 다시 입력할 필요가 없다
chrome.storage?.local?.get?.(['lastIds'], (res) => {
  if (res && res.lastIds) idsInput.value = res.lastIds;
});

findBtn.addEventListener('click', onFind);

function onFind() {
  clearOutput();

  const ids = parseIds(idsInput.value);
  if (ids.length === 0) {
    showStatus('상품ID를 하나 이상 입력해주세요.', true);
    return;
  }

  chrome.storage?.local?.set?.({ lastIds: idsInput.value });

  findBtn.disabled = true;
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs && tabs[0];
    if (!tab || !tab.url || !tab.url.startsWith(COUPANG_SEARCH_URL)) {
      findBtn.disabled = false;
      showStatus(
        '쿠팡 검색결과 페이지가 아닙니다.\n' +
        'coupang.com에서 키워드를 검색한 뒤 그 탭에서 다시 눌러주세요.',
        true
      );
      return;
    }

    chrome.tabs.sendMessage(tab.id, { type: 'PARSE_SEARCH_RESULTS' }, (response) => {
      findBtn.disabled = false;

      // 콘텐츠 스크립트가 주입 안 된 상태(확장 설치/갱신 직후 등)
      if (chrome.runtime.lastError) {
        showStatus(
          '페이지와 연결하지 못했습니다.\n' +
          '검색결과 페이지를 새로고침(F5)한 뒤 다시 시도해주세요.',
          true
        );
        return;
      }

      if (!response || !response.ok) {
        showStatus('파싱 중 오류: ' + (response ? response.error : '응답 없음'), true);
        return;
      }

      render(response.data, ids);
    });
  });
}

// "123, 456\n789" → ['123', '456', '789']
function parseIds(raw) {
  return raw
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function render(data, myIds) {
  const { keyword, page, itemsFound, products } = data;

  if (itemsFound === 0) {
    showStatus(
      '상품 목록을 찾지 못했습니다.\n' +
      '쿠팡 DOM 구조가 바뀌었을 수 있습니다. content.js 상단의 SELECTORS를 점검해주세요.',
      true
    );
    return;
  }

  const adCount = products.filter((p) => p.isAd).length;
  summaryEl.hidden = false;
  summaryEl.textContent =
    `키워드 "${keyword}" · ${page}페이지 · ` +
    `상품 ${products.length}개 파싱 (광고 ${adCount}개 포함)`;

  // 내 상품ID를 productId / itemId / vendorItemId 어디서든 매칭
  myIds.forEach((id) => {
    const found = products.find(
      (p) => p.productId === id || p.itemId === id || p.vendorItemId === id
    );
    resultsEl.appendChild(resultCard(id, found));
  });

  // 파싱 검증용 전체 목록
  allWrapEl.hidden = false;
  allListEl.textContent = '';
  products.forEach((p) => {
    const li = document.createElement('li');
    li.textContent =
      `${p.position}위${p.isAd ? '[AD]' : ''} ` +
      `(${p.productId || '?'}) ${p.name || '(상품명 없음)'}`;
    allListEl.appendChild(li);
  });
}

function resultCard(id, p) {
  const li = document.createElement('li');
  li.className = 'result-card';

  if (!p) {
    li.classList.add('not-found');
    li.textContent = `상품ID ${id} — 이 페이지에서 찾지 못했습니다. (다음 페이지를 확인해보세요)`;
    return li;
  }

  const rankLine = document.createElement('div');
  rankLine.className = 'rank-line';

  const rank = document.createElement('span');
  rank.className = 'rank';
  rank.textContent = `${p.position}위`;
  rankLine.appendChild(rank);

  const badge = document.createElement('span');
  badge.className = 'badge ' + (p.isAd ? 'ad' : 'organic');
  badge.textContent = p.isAd ? '광고' : `일반 ${p.organicRank}위`;
  rankLine.appendChild(badge);

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

  li.appendChild(rankLine);
  li.appendChild(name);
  li.appendChild(meta);
  return li;
}

function showStatus(text, isError) {
  statusEl.hidden = false;
  statusEl.textContent = text;
  statusEl.classList.toggle('error', !!isError);
}

function clearOutput() {
  statusEl.hidden = true;
  statusEl.textContent = '';
  summaryEl.hidden = true;
  resultsEl.textContent = '';
  allWrapEl.hidden = true;
  allListEl.textContent = '';
}
