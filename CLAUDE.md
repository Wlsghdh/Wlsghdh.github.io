# 프로젝트 규칙

개인 연구자 포트폴리오 + 기록 사이트 (Astro + Tailwind v4). 사양서: `~/cau/history/research-journal-site-plan.md` (v4) — 단, 사용자 프롬프트가 사양서보다 우선.

## Git / 배포

- **push는 Claude가 자동으로 한다.** 의미 있는 작업 단위가 끝나면 커밋 후 `git push origin main`까지 수행할 것 (사용자에게 물어보지 않는다).
- push되면 GitHub Actions가 빌드해 https://wlsghdh.github.io (뷰어)에 자동 반영된다.
- push 전 `npm run build`가 통과하는지 반드시 확인.

## 디자인 결정 사항 (사양서와 다른 확정 변경)

- 좌측 사이드바 ❌ → **상단 sticky 내비바** + 홈 상단 프로필 헤더.
- About 섹션 없음. 활동(잔디)은 내비에 넣지 않고 **홈 맨 아래 "History" 섹션**에 history line으로 배치.
- 활동 카테고리 색 레이블: 논문리딩·코드작업·기업과제·논문작성·세미나 (`src/data/site.ts`의 `activityCategories`, 색 토큰은 `tokens.css`의 `--cat-*`).
- 포인트 컬러는 CAU Blue(#004C97), 이름/로고만 한나체 Air, 본문 Pretendard, 날짜·태그·venue는 JetBrains Mono.

## 데이터 위치

- 개인 정보·내비: `src/data/site.ts` / 이력서·포트폴리오: `src/data/cv.ts` (출처: `~/cau/history/CV_portpolio/`)
- 콘텐츠(Phase 2~): `src/content/logs/`, `src/content/papers/` — 프론트매터 `locked:true`면 본문 비공개, 카드·활동 카운트만.
