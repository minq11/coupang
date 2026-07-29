/**
 * content.js — 쿠팡 검색결과 페이지 DOM 파서 + 다중 페이지 스캐너
 *
 * 팝업에서 오는 메시지 두 종류에 응답한다:
 *  - PARSE_SEARCH_RESULTS : 현재 페이지만 파싱 (동기)
 *  - SCAN_PAGES           : 1페이지부터 maxPages까지 순서대로 스캔 (비동기)
 *    다른 페이지는 같은 도메인 fetch + DOMParser로 읽으므로
 *    셀러 브라우저의 쿠키/IP 그대로 동작하고, 요청 사이에 랜덤 딜레이를 둔다.
 *
 * 백그라운드 자동 크롤링은 하지 않는다 — 버튼을 누른 순간에만 동작.
 */

// ============================================================
// 셀렉터 모음 — 쿠팡이 DOM 구조를 바꾸면 여기만 고치면 된다.
//
// 각 항목은 "우선순위 순서의 후보 배열"이다. 앞에서부터 시도해서
// 처음으로 매칭되는 셀렉터를 쓴다. 쿠팡은 구버전/신버전 마크업이
// 섞여 나오는 경우가 있어서 알려진 패턴을 전부 후보로 넣어뒀다.
//
// 안 맞을 때 고치는 법:
//   1. 쿠팡 검색결과 페이지에서 F12 → 상품 하나에 우클릭 → 검사
//   2. 상품 <li>(또는 카드 컨테이너)의 태그/클래스를 확인
//   3. 아래 후보 배열 맨 앞에 새 셀렉터를 추가
// ============================================================
const SELECTORS = {
  // 검색결과 상품 목록의 각 아이템 (상품 카드 하나 = 요소 하나)
  productItems: [
    'ul#productList > li.search-product',   // 구버전 쿠팡 검색결과
    'ul#product-list > li',                 // 신버전 (2024~) 목록
    'li.search-product',                    // 구버전 변형
    'li[class*="ProductUnit"]',             // 신버전 CSS 모듈 클래스
  ],

  // 상품 상세 링크 — href에서 productId/itemId/vendorItemId를 뽑는다
  link: [
    'a.search-product-link',
    'a[href*="/vp/products/"]',
  ],

  // 상품명
  name: [
    'div.name',
    '[class*="productName"]',
    '[class*="product-name"]',
  ],

  // 판매가 (숫자 부분)
  price: [
    'strong.price-value',
    '[class*="priceValue"]',
    '[class*="price-value"]',
  ],

  // 별점 (예: "4.5")
  rating: [
    'em.rating',
    '[class*="ProductRating_rating__"]',
    'span.star em',
  ],

  // 리뷰 수 (예: "(1,234)")
  reviewCount: [
    'span.rating-total-count',
    '[class*="ratingCount"]',
    '[class*="rating-total-count"]',
  ],

  // 광고(AD) 뱃지 — 이 요소가 상품 카드 안에 있으면 광고 상품으로 본다
  adBadge: [
    'span.ad-mark-text',
    '[class*="AdMark"]',
    '[class*="ad-badge"]',
    '[data-adsplatform]',
  ],

  // 로켓배송/판매자로켓 등 배송 뱃지 — img의 alt 텍스트로 종류를 구분한다
  deliveryBadge: [
    'img[alt*="로켓"]',
    'img[src*="logo_rocket"]',
    'img[src*="rocket_logo"]',
    'img[src*="rocketwow"]',
    '[class*="rocket"] img',
  ],
};

// 다중 페이지 스캔 시 요청 간 딜레이(ms) — 차단 리스크를 낮춘다
const SCAN_DELAY_MIN = 700;
const SCAN_DELAY_JITTER = 600;

// ------------------------------------------------------------
// 셀렉터 후보 배열을 앞에서부터 시도하는 헬퍼.
// 아무것도 못 찾으면 null / 빈 배열을 돌려주고 절대 throw하지 않는다.
// ------------------------------------------------------------
function queryFirst(root, candidates) {
  for (const sel of candidates) {
    try {
      const el = root.querySelector(sel);
      if (el) return el;
    } catch (e) {
      // 잘못된 셀렉터 문자열이어도 다음 후보로 넘어간다
    }
  }
  return null;
}

function queryAllFirst(root, candidates) {
  for (const sel of candidates) {
    try {
      const els = root.querySelectorAll(sel);
      if (els.length > 0) return Array.from(els);
    } catch (e) {
      // 다음 후보로
    }
  }
  return [];
}

function textOf(root, candidates) {
  const el = queryFirst(root, candidates);
  return el ? el.textContent.trim() : null;
}

// "1,234원" → 1234 같은 숫자 추출. 실패하면 null.
function toNumber(text) {
  if (!text) return null;
  const digits = text.replace(/[^0-9]/g, '');
  return digits ? parseInt(digits, 10) : null;
}

// ------------------------------------------------------------
// 상품 카드 하나 파싱. 어떤 필드를 못 찾아도 죽지 않고
// 찾은 것만 채워서 돌려준다. 링크/ID를 아예 못 찾으면 null.
// ------------------------------------------------------------
function parseItem(item, position, pageNo) {
  const link = queryFirst(item, SELECTORS.link);
  const href = link ? link.getAttribute('href') || '' : '';

  // productId: li의 data 속성 우선, 없으면 href의 /vp/products/{id}에서 추출
  let productId = item.dataset ? (item.dataset.productId || null) : null;
  if (!productId) {
    const m = href.match(/\/vp\/products\/(\d+)/);
    if (m) productId = m[1];
  }

  // itemId / vendorItemId: href 쿼리스트링에서 추출 (없어도 무방)
  let itemId = null;
  let vendorItemId = null;
  try {
    const qs = new URLSearchParams(href.split('?')[1] || '');
    itemId = qs.get('itemId');
    vendorItemId = qs.get('vendorItemId');
  } catch (e) {
    // href가 이상해도 무시
  }

  if (!productId && !itemId && !vendorItemId) {
    // ID를 하나도 못 뽑은 카드는 상품이 아닐 가능성이 높으니 건너뛴다
    return null;
  }

  // 배송 뱃지: 로켓배송/판매자로켓/로켓직구/로켓프레시 등 (alt 텍스트 기준)
  let delivery = null;
  const deliveryEl = queryFirst(item, SELECTORS.deliveryBadge);
  if (deliveryEl) {
    const alt = (deliveryEl.getAttribute('alt') || '').trim();
    delivery = alt && alt.includes('로켓') ? alt : '로켓';
  }

  return {
    position,                                        // 전체 순위 (광고 포함, 페이지 누적, 1부터)
    organicRank: null,                               // 광고 제외 순위 — 호출부에서 채움
    page: pageNo,                                    // 몇 페이지에서 발견됐는지
    isAd: queryFirst(item, SELECTORS.adBadge) !== null,
    delivery,
    productId,
    itemId,
    vendorItemId,
    name: textOf(item, SELECTORS.name),
    price: toNumber(textOf(item, SELECTORS.price)),
    rating: textOf(item, SELECTORS.rating),
    reviewCount: toNumber(textOf(item, SELECTORS.reviewCount)),
  };
}

// ------------------------------------------------------------
// 문서(현재 문서 또는 DOMParser 결과)에서 상품 목록 파싱.
// position/organicRank는 startPosition/startOrganic부터 누적 계산.
// ------------------------------------------------------------
function parseProductsFrom(doc, pageNo, startPosition, startOrganic) {
  const items = queryAllFirst(doc, SELECTORS.productItems);
  const products = [];
  let position = startPosition;
  let organicRank = startOrganic;

  items.forEach((item) => {
    try {
      const p = parseItem(item, position + 1, pageNo);
      if (!p) return;
      position += 1;
      if (!p.isAd) {
        organicRank += 1;
        p.organicRank = organicRank;
      }
      products.push(p);
    } catch (e) {
      // 카드 하나가 실패해도 나머지는 계속 파싱한다
    }
  });

  return { products, itemsFound: items.length, endPosition: position, endOrganic: organicRank };
}

function currentPageNo() {
  try {
    return parseInt(new URL(location.href).searchParams.get('page') || '1', 10) || 1;
  } catch (e) {
    return 1;
  }
}

function currentKeyword() {
  try {
    return new URL(location.href).searchParams.get('q') || '';
  } catch (e) {
    return '';
  }
}

// ------------------------------------------------------------
// 현재 페이지만 파싱 (기존 동작)
// ------------------------------------------------------------
function parseSearchResults() {
  const pageNo = currentPageNo();
  const { products, itemsFound } = parseProductsFrom(document, pageNo, 0, 0);
  return {
    keyword: currentKeyword(),
    pagesScanned: 1,
    firstPage: pageNo,
    pageUrl: location.href,
    parsedAt: new Date().toISOString(),
    itemsFound,
    products,
  };
}

// ------------------------------------------------------------
// 다중 페이지 스캔: 1페이지부터 maxPages까지.
// 현재 열려있는 페이지 번호와 일치하면 fetch 없이 현재 DOM을 쓴다.
// ------------------------------------------------------------
let isScanning = false;

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// 현재 URL의 검색 파라미터를 유지한 채 page만 바꾼 URL 생성
function buildPageUrl(pageNo) {
  const url = new URL(location.href);
  url.searchParams.set('page', String(pageNo));
  return url.toString();
}

function reportProgress(current, max) {
  try {
    chrome.runtime.sendMessage({ type: 'SCAN_PROGRESS', current, max }, () => {
      // 팝업이 닫혀 수신자가 없어도 조용히 무시
      void chrome.runtime.lastError;
    });
  } catch (e) {
    // 무시
  }
}

async function scanPages(maxPages) {
  if (isScanning) throw new Error('이미 스캔이 진행 중입니다. 잠시 후 다시 시도해주세요.');
  isScanning = true;

  try {
    const keyword = currentKeyword();
    const openPageNo = currentPageNo();
    const all = [];
    let position = 0;
    let organic = 0;
    let pagesScanned = 0;
    let totalItemsFound = 0;

    for (let pageNo = 1; pageNo <= maxPages; pageNo++) {
      reportProgress(pageNo, maxPages);

      let doc;
      if (pageNo === openPageNo) {
        doc = document; // 지금 보고 있는 페이지는 fetch 없이 그대로 사용
      } else {
        let res;
        try {
          res = await fetch(buildPageUrl(pageNo), { credentials: 'include' });
        } catch (e) {
          throw new Error(`${pageNo}페이지 요청 실패: ${e.message}`);
        }
        if (!res.ok) {
          // 접근 제한 등으로 실패하면 지금까지 모은 결과로 마무리
          break;
        }
        const html = await res.text();
        doc = new DOMParser().parseFromString(html, 'text/html');
      }

      const { products, itemsFound, endPosition, endOrganic } =
        parseProductsFrom(doc, pageNo, position, organic);

      totalItemsFound += itemsFound;

      // 상품이 하나도 없으면 마지막 페이지를 넘은 것 (또는 셀렉터 문제)
      if (itemsFound === 0) break;

      all.push(...products);
      position = endPosition;
      organic = endOrganic;
      pagesScanned += 1;

      // 다음 페이지 요청 전 랜덤 딜레이 (마지막 페이지 뒤에는 불필요)
      if (pageNo < maxPages) {
        await delay(SCAN_DELAY_MIN + Math.random() * SCAN_DELAY_JITTER);
      }
    }

    return {
      keyword,
      pagesScanned,
      firstPage: 1,
      pageUrl: location.href,
      parsedAt: new Date().toISOString(),
      itemsFound: totalItemsFound,
      products: all,
    };
  } finally {
    isScanning = false;
  }
}

// ------------------------------------------------------------
// 검색결과 페이지 위에 내 상품 하이라이트 (테두리 + 순위 뱃지)
// 팝업 없이도 페이지에서 바로 내 상품 위치가 보이게 한다.
// ------------------------------------------------------------
const HL_STYLE_ID = '__crf-highlight-style';
const HL_CLASS = '__crf-hit';
const HL_BADGE_CLASS = '__crf-badge';

function ensureHighlightStyle() {
  if (document.getElementById(HL_STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = HL_STYLE_ID;
  style.textContent = `
    .${HL_CLASS} {
      outline: 3px solid #e84118 !important;
      outline-offset: -3px;
      border-radius: 8px;
    }
    .${HL_BADGE_CLASS} {
      position: absolute;
      top: 8px;
      left: 8px;
      z-index: 999;
      padding: 3px 10px;
      border-radius: 999px;
      background: #e84118;
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.4;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
      pointer-events: none;
    }
  `;
  document.head.appendChild(style);
}

function clearHighlights() {
  document.querySelectorAll('.' + HL_CLASS).forEach((el) => el.classList.remove(HL_CLASS));
  document.querySelectorAll('.' + HL_BADGE_CLASS).forEach((el) => el.remove());
}

/**
 * @param {Array<{ids: string[], label: string}>} targets
 * @returns {number} 현재 페이지에서 하이라이트된 개수
 */
function highlightMyProducts(targets) {
  clearHighlights();
  if (!Array.isArray(targets) || targets.length === 0) return 0;
  ensureHighlightStyle();

  const items = queryAllFirst(document, SELECTORS.productItems);
  let hits = 0;

  items.forEach((item) => {
    try {
      const p = parseItem(item, 0, 0); // ID 추출 용도로만 사용
      if (!p) return;
      const target = targets.find((t) =>
        (p.productId && t.ids.includes(p.productId)) ||
        (p.itemId && t.ids.includes(p.itemId)) ||
        (p.vendorItemId && t.ids.includes(p.vendorItemId))
      );
      if (!target) return;

      item.classList.add(HL_CLASS);
      if (getComputedStyle(item).position === 'static') {
        item.style.position = 'relative';
      }
      const badge = document.createElement('div');
      badge.className = HL_BADGE_CLASS;
      badge.textContent = target.label;
      item.appendChild(badge);
      hits += 1;

      // 첫 번째 히트로 스크롤 (한 번만)
      if (hits === 1) {
        item.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    } catch (e) {
      // 하나 실패해도 계속
    }
  });

  return hits;
}

// ------------------------------------------------------------
// 팝업에서 오는 메시지 처리
// ------------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg) return;

  if (msg.type === 'PARSE_SEARCH_RESULTS') {
    try {
      sendResponse({ ok: true, data: parseSearchResults() });
    } catch (e) {
      sendResponse({ ok: false, error: e.message });
    }
    return; // 동기 응답
  }

  if (msg.type === 'SCAN_PAGES') {
    const maxPages = Math.min(Math.max(parseInt(msg.maxPages, 10) || 5, 1), 10);
    scanPages(maxPages)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((e) => sendResponse({ ok: false, error: e.message }));
    return true; // 비동기 응답 유지
  }

  if (msg.type === 'HIGHLIGHT_MY_PRODUCTS') {
    try {
      sendResponse({ ok: true, hits: highlightMyProducts(msg.targets) });
    } catch (e) {
      sendResponse({ ok: false, error: e.message });
    }
    return;
  }

  if (msg.type === 'CLEAR_HIGHLIGHTS') {
    try {
      clearHighlights();
      sendResponse({ ok: true });
    } catch (e) {
      sendResponse({ ok: false, error: e.message });
    }
    return;
  }
});
