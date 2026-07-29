/**
 * background.js — MV3 서비스워커 (모듈)
 *
 * 역할:
 *  1. API 키 로드 — 두 가지 소스를 순서대로 시도:
 *     ① env.json (개발자용: env.example.json을 복사해서 만든 파일)
 *     ② chrome.storage.local의 apiKeys (일반 사용자용: 설정 탭에서 직접 입력)
 *  2. 팝업에서 오는 API_* 메시지를 받아 쿠팡 WING Open API 호출 후 응답
 *
 * 쿠팡 API는 CORS 때문에 팝업/콘텐츠 스크립트에서 직접 못 부르므로
 * 반드시 여기(서비스워커)를 거친다.
 */

import { CoupangApi } from './api/coupang-api.js';

// ------------------------------------------------------------
// 설정 로드
// cachedConfig: undefined = 아직 안 읽음, null = 미설정
// env.json을 바꿨으면 chrome://extensions에서 확장 새로고침(↻) 필요.
// 설정 탭에서 저장한 키는 즉시 반영된다.
// ------------------------------------------------------------
let cachedConfig;
let configSource = null; // 'env' | 'storage' | null

function normalizeConfig(accessKey, secretKey, vendorId) {
  const config = {
    accessKey: (accessKey || '').trim(),
    secretKey: (secretKey || '').trim(),
    vendorId: (vendorId || '').trim(),
  };
  return config.accessKey && config.secretKey && config.vendorId ? config : null;
}

async function loadConfig(force) {
  if (!force && cachedConfig !== undefined) return cachedConfig;

  cachedConfig = null;
  configSource = null;

  // ① env.json (있으면 우선)
  try {
    const res = await fetch(chrome.runtime.getURL('env.json'));
    if (res.ok) {
      const raw = await res.json();
      const config = normalizeConfig(
        raw.COUPANG_ACCESS_KEY, raw.COUPANG_SECRET_KEY, raw.COUPANG_VENDOR_ID
      );
      if (config) {
        cachedConfig = config;
        configSource = 'env';
        return cachedConfig;
      }
    }
  } catch (e) {
    // env.json 없음 — 다음 소스로
  }

  // ② 설정 탭에서 저장한 키
  try {
    const { apiKeys } = await chrome.storage.local.get('apiKeys');
    if (apiKeys) {
      const config = normalizeConfig(apiKeys.accessKey, apiKeys.secretKey, apiKeys.vendorId);
      if (config) {
        cachedConfig = config;
        configSource = 'storage';
      }
    }
  } catch (e) {
    // 저장소 오류 — 미설정으로 처리
  }

  return cachedConfig;
}

async function getApi() {
  const config = await loadConfig();
  if (!config) {
    throw new Error(
      'API 키가 설정되지 않았습니다. 설정 탭에서 키를 입력하거나, ' +
      'env.example.json을 env.json으로 복사해 키를 채워주세요.'
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
      source: configSource, // 'env' | 'storage'
      vendorId: config.vendorId,
      accessKeyMasked: config.accessKey.slice(0, 4) + '****',
    };
  },

  // 설정 탭에서 키 저장 (env.json이 있으면 env.json이 계속 우선한다)
  async API_SAVE_KEYS(msg) {
    const config = normalizeConfig(msg.accessKey, msg.secretKey, msg.vendorId);
    if (!config) {
      throw new Error('세 값(Access Key, Secret Key, Vendor ID)을 모두 입력해주세요.');
    }
    await chrome.storage.local.set({ apiKeys: config });
    await loadConfig(true); // 캐시 갱신
    return { saved: true, source: configSource };
  },

  async API_CLEAR_KEYS() {
    await chrome.storage.local.remove('apiKeys');
    await loadConfig(true);
    return { cleared: true };
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
