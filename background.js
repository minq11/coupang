/**
 * background.js — MV3 서비스워커
 *
 * 현재 MVP에서는 팝업이 chrome.tabs.sendMessage로 콘텐츠 스크립트와
 * 직접 통신하므로 백그라운드가 할 일이 거의 없다. 최소한만 둔다.
 *
 * 나중에 여기에 들어갈 것 (지금은 만들지 않음):
 *  - 라이선스 인증 서버 통신
 *  - 순위 기록 저장/집계
 */

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('[쿠팡 순위 파인더] 설치 완료');
  }
});
