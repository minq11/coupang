/**
 * api/coupang-api.js — 쿠팡 WING Open API 클라이언트
 *
 * 백그라운드 서비스워커에서만 사용한다 (CORS 때문에 팝업/콘텐츠에서 직접 호출 불가).
 * 인증은 쿠팡 CEA(HMAC-SHA256) 서명 방식:
 *   message   = signed-date + method + path + query
 *   signature = HMAC-SHA256(secretKey, message) 의 hex
 *   Authorization: CEA algorithm=HmacSHA256, access-key=..., signed-date=..., signature=...
 *
 * 참고: https://developers.coupangcorp.com (WING Open API 문서)
 * 엔드포인트 경로가 바뀌면 아래 ENDPOINTS 상수만 고치면 된다.
 */

const API_HOST = 'https://api-gateway.coupang.com';

// ============================================================
// 엔드포인트 모음 — 쿠팡이 API 경로를 바꾸면 여기만 고친다
// ============================================================
const ENDPOINTS = {
  // 등록상품 목록 페이징 조회
  listProducts: '/v2/providers/seller_api/apis/api/v1/marketplace/seller-products',
  // 등록상품 단건 상세 조회 (뒤에 /{sellerProductId} 붙여서 사용)
  getProduct: '/v2/providers/seller_api/apis/api/v1/marketplace/seller-products',
  // 발주서(주문) 일단위 조회 — {vendorId} 자리에 판매자ID가 들어간다
  ordersheets: (vendorId) =>
    `/v2/providers/openapi/apis/api/v4/vendors/${vendorId}/ordersheets`,
};

// 주문 상태 코드 → 한국어 라벨 (팝업 표시용으로도 내려준다)
export const ORDER_STATUS_LABELS = {
  ACCEPT: '결제완료',
  INSTRUCT: '상품준비중',
  DEPARTURE: '배송지시',
  DELIVERING: '배송중',
  FINAL_DELIVERY: '배송완료',
};

// 발주서 1건의 예상 매출액. 응답 필드명이 유동적이라 방어적으로 계산:
// orderItems[].orderPrice가 있으면 그 합, 없으면 salesPrice × 수량으로 추정.
function orderRevenue(order) {
  if (!order || !Array.isArray(order.orderItems)) return 0;
  return order.orderItems.reduce((sum, item) => {
    if (!item) return sum;
    if (typeof item.orderPrice === 'number') return sum + item.orderPrice;
    const unit = typeof item.salesPrice === 'number' ? item.salesPrice : 0;
    const qty = typeof item.shippingCount === 'number' ? item.shippingCount : 1;
    return sum + unit * qty;
  }, 0);
}

// signed-date: UTC 기준 "yyMMdd'T'HHmmss'Z'" 형식 (예: 260729T093000Z)
function signedDateNow() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return (
    String(d.getUTCFullYear()).slice(2) +
    p(d.getUTCMonth() + 1) +
    p(d.getUTCDate()) +
    'T' +
    p(d.getUTCHours()) +
    p(d.getUTCMinutes()) +
    p(d.getUTCSeconds()) +
    'Z'
  );
}

// HMAC-SHA256 → hex 문자열 (서비스워커의 Web Crypto API 사용)
async function hmacSha256Hex(secret, message) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(message));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export class CoupangApi {
  /**
   * @param {{accessKey: string, secretKey: string, vendorId: string}} config
   */
  constructor(config) {
    this.accessKey = config.accessKey;
    this.secretKey = config.secretKey;
    this.vendorId = config.vendorId;
  }

  /**
   * 서명 붙여서 API 호출. 실패 시 사람이 읽을 수 있는 에러 메시지로 throw.
   * @param {string} method - 'GET' 등
   * @param {string} path - '/v2/providers/...' 형태
   * @param {Object} [queryParams] - 쿼리스트링 파라미터
   */
  async request(method, path, queryParams = {}) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(queryParams)) {
      if (v !== undefined && v !== null && v !== '') params.set(k, v);
    }
    // 서명에 쓰는 query와 실제 요청 query는 반드시 같은 문자열이어야 한다
    const query = params.toString();

    const signedDate = signedDateNow();
    const message = signedDate + method + path + query;
    const signature = await hmacSha256Hex(this.secretKey, message);

    const authorization =
      `CEA algorithm=HmacSHA256, access-key=${this.accessKey}, ` +
      `signed-date=${signedDate}, signature=${signature}`;

    const url = API_HOST + path + (query ? '?' + query : '');

    let res;
    try {
      res = await fetch(url, {
        method,
        headers: {
          'Authorization': authorization,
          'Content-Type': 'application/json;charset=UTF-8',
        },
      });
    } catch (e) {
      throw new Error('쿠팡 API 서버에 연결하지 못했습니다: ' + e.message);
    }

    const text = await res.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch (e) {
      // JSON이 아닌 에러 페이지가 올 수도 있다
    }

    if (!res.ok) {
      const detail = (body && (body.message || body.errorMessage)) || text.slice(0, 200);
      if (res.status === 401) {
        throw new Error('인증 실패(401): Access Key/Secret Key를 확인해주세요. ' + detail);
      }
      if (res.status === 403) {
        throw new Error('권한 없음(403): Vendor ID 또는 API 사용 권한을 확인해주세요. ' + detail);
      }
      throw new Error(`API 오류(${res.status}): ${detail}`);
    }

    return body;
  }

  /**
   * 등록상품 목록 조회 (페이징)
   * @param {{nextToken?: string, maxPerPage?: number}} [opts]
   * @returns {Promise<{products: Array, nextToken: string|null}>}
   */
  async listSellerProducts(opts = {}) {
    const body = await this.request('GET', ENDPOINTS.listProducts, {
      vendorId: this.vendorId,
      maxPerPage: opts.maxPerPage || 20,
      nextToken: opts.nextToken || undefined,
    });
    return {
      products: (body && body.data) || [],
      nextToken: (body && body.nextToken) || null,
    };
  }

  /**
   * 등록상품 단건 상세 조회 — items 배열에 vendorItemId 등 매칭용 ID가 들어있다
   * @param {string|number} sellerProductId
   */
  async getSellerProduct(sellerProductId) {
    const body = await this.request(
      'GET',
      `${ENDPOINTS.getProduct}/${sellerProductId}`
    );
    return (body && body.data) || null;
  }

  /**
   * 발주서(주문) 일단위 조회 — status별로 페이징
   * @param {{dateFrom: string, dateTo: string, status: string, nextToken?: string, maxPerPage?: number}} opts
   */
  async listOrderSheets(opts) {
    const body = await this.request('GET', ENDPOINTS.ordersheets(this.vendorId), {
      createdAtFrom: opts.dateFrom,
      createdAtTo: opts.dateTo,
      status: opts.status,
      maxPerPage: opts.maxPerPage || 50,
      nextToken: opts.nextToken || undefined,
    });
    return {
      orders: (body && body.data) || [],
      nextToken: (body && body.nextToken) || null,
    };
  }

  /**
   * 오늘(KST) 주문 요약: 상태별 건수 + 예상 매출 합계.
   * 상태 코드는 쿠팡 발주서 API의 진행 단계. 일부 상태 조회가 실패해도
   * 성공한 상태만으로 요약을 만든다 (전부 실패하면 throw).
   */
  async getTodayOrderSummary() {
    // KST(UTC+9) 기준 오늘 날짜 "YYYY-MM-DD"
    const today = new Date(Date.now() + 9 * 3600 * 1000).toISOString().slice(0, 10);

    const statuses = Object.keys(ORDER_STATUS_LABELS);
    const counts = {};
    let total = 0;
    let revenue = 0;
    let hasMore = false;
    let anySuccess = false;
    let lastError = null;

    for (const status of statuses) {
      try {
        let nextToken = null;
        let count = 0;
        let pages = 0;
        do {
          const res = await this.listOrderSheets({
            dateFrom: today,
            dateTo: today,
            status,
            nextToken,
          });
          count += res.orders.length;
          revenue += res.orders.reduce((sum, o) => sum + orderRevenue(o), 0);
          nextToken = res.nextToken;
          pages += 1;
        } while (nextToken && pages < 5); // 상태당 최대 250건까지 (그 이상은 hasMore)
        if (nextToken) hasMore = true;
        counts[status] = count;
        total += count;
        anySuccess = true;
      } catch (e) {
        lastError = e.message;
      }
    }

    if (!anySuccess) {
      throw new Error(lastError || '주문 조회에 실패했습니다.');
    }

    return { date: today, total, revenue, counts, hasMore, labels: ORDER_STATUS_LABELS };
  }

  /** 연결 테스트: 상품 1개만 조회해본다 */
  async testConnection() {
    const { products } = await this.listSellerProducts({ maxPerPage: 1 });
    return {
      ok: true,
      sampleCount: products.length,
      message:
        products.length > 0
          ? '연결 성공! 등록상품 조회가 정상 동작합니다.'
          : '연결 성공! (등록된 상품이 없거나 조회 결과가 비어있습니다)',
    };
  }
}
