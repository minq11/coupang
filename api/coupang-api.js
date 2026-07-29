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
};

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
