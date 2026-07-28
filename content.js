/**
 * content.js — 쿠팡 검색결과 페이지 DOM 파서
 *
 * 셀러가 이미 열어둔 쿠팡 검색결과 페이지에서 상품 목록을 읽는다.
 * 팝업(popup.js)이 보내는 { type: 'PARSE_SEARCH_RESULTS' } 메시지에
 * 파싱 결과를 응답하는 것이 전부다. 백그라운드 자동 크롤링은 하지 않는다.
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
};

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
function parseItem(item, position) {
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

  return {
    position,                                        // 페이지 내 전체 순위 (광고 포함, 1부터)
    organicRank: null,                               // 광고 제외 순위 — 아래에서 채움
    isAd: queryFirst(item, SELECTORS.adBadge) !== null,
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
// 페이지 전체 파싱
// ------------------------------------------------------------
function parseSearchResults() {
  const url = new URL(location.href);
  const items = queryAllFirst(document, SELECTORS.productItems);

  const products = [];
  let organicRank = 0;

  items.forEach((item, idx) => {
    try {
      const p = parseItem(item, idx + 1);
      if (!p) return;
      if (!p.isAd) {
        organicRank += 1;
        p.organicRank = organicRank;
      }
      products.push(p);
    } catch (e) {
      // 카드 하나가 실패해도 나머지는 계속 파싱한다
    }
  });

  return {
    keyword: url.searchParams.get('q') || '',
    page: url.searchParams.get('page') || '1',
    pageUrl: location.href,
    parsedAt: new Date().toISOString(),
    itemsFound: items.length,      // 목록 셀렉터가 잡은 카드 수 (0이면 셀렉터 점검 필요)
    products,
  };
}

// ------------------------------------------------------------
// 팝업에서 오는 메시지 처리
// ------------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === 'PARSE_SEARCH_RESULTS') {
    try {
      sendResponse({ ok: true, data: parseSearchResults() });
    } catch (e) {
      sendResponse({ ok: false, error: e.message });
    }
  }
  // 동기 응답이므로 true를 반환하지 않는다
});
