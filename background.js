/**
 * background.js — MV3 서비스워커 (모듈)
 *
 * 역할:
 *  1. env.json(API 키 파일) 로드 — env.example.json을 복사해서 만든 파일
 *  2. 팝업에서 오는 API_* 메시지를 받아 쿠팡 WING Open API 호출 후 응답
 *
 * 쿠팡 API는 CORS 때문에 팝업/콘텐츠 스크립트에서 직접 못 부르므로
 * 반드시 여기(서비스워커)를 거친다.
 */

import { CoupangApi } from './api/coupang-api.js';

// ------------------------------------------------------------
// env.json 로드
// - 확장 폴더 안의 env.json을 fetch로 읽는다 (없으면 "미설정" 상태)
// - 키를 바꿨으면 chrome://extensions에서 확장 새로고침(↻) 필요
// ------------------------------------------------------------
let cachedConfig; // undefined = 아직 안 읽음, null = 파일 없음/불완전

async function loadConfig() {
  if (cachedConfig !== undefined) return cachedConfig;
  try {
    const res = await fetch(chrome.runtime.getURL('env.json'));
    if (!res.ok) throw new Error('env.json not found');
    const raw = await res.json();
    const config = {
      accessKey: (raw.COUPANG_ACCESS_KEY || '').trim(),
      secretKey: (raw.COUPANG_SECRET_KEY || '').trim(),
      vendorId: (raw.COUPANG_VENDOR_ID || '').trim(),
    };
    cachedConfig =
      config.accessKey && config.secretKey && config.vendorId ? config : null;
  } catch (e) {
    cachedConfig = null;
  }
  return cachedConfig;
}

async function getApi() {
  const config = await loadConfig();
  if (!config) {
    throw new Error(
      'API 키가 설정되지 않았습니다. env.example.json을 env.json으로 복사해 ' +
      '키를 채운 뒤, chrome://extensions에서 확장을 새로고침해주세요.'
    );
  }
  return new CoupangApi(config);
}

// ------------------------------------------------------------
// 팝업 → 백그라운드 메시지 처리
// ------------------------------------------------------------
const handlers = {
  // 설정 상태 확인 (키 값 자체는 절대 팝업으로 보내지 않는다)
  async API_STATUS() {
    const config = await loadConfig();
    if (!config) return { configured: false };
    return {
      configured: true,
      vendorId: config.vendorId,
      accessKeyMasked: config.accessKey.slice(0, 4) + '****',
    };
  },

  async API_TEST() {
    const api = await getApi();
    return api.testConnection();
  },

  async API_LIST_PRODUCTS(msg) {
    const api = await getApi();
    return api.listSellerProducts({
      nextToken: msg.nextToken,
      maxPerPage: msg.maxPerPage || 20,
    });
  },

  async API_GET_PRODUCT(msg) {
    const api = await getApi();
    return { product: await api.getSellerProduct(msg.sellerProductId) };
  },

  async API_ORDER_SUMMARY() {
    const api = await getApi();
    return { summary: await api.getTodayOrderSummary() };
  },
};

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const handler = msg && handlers[msg.type];
  if (!handler) return; // PARSE_SEARCH_RESULTS 등은 콘텐츠 스크립트 담당

  handler(msg)
    .then((data) => sendResponse({ ok: true, ...data }))
    .catch((e) => sendResponse({ ok: false, error: e.message }));
  return true; // 비동기 응답 유지
});

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('[쿠팡 순위 파인더] 설치 완료');
  }
});
