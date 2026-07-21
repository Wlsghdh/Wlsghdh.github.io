# 프로젝트 규칙

개인 연구자 포트폴리오 + 기록 사이트 (Astro + Tailwind v4). 사양서: `~/cau/history/research-journal-site-plan.md` (v4) — 단, 사용자 프롬프트가 사양서보다 우선.

## Git / 배포

- **push는 Claude가 자동으로 한다.** 의미 있는 작업 단위가 끝나면 커밋 후 `git push origin main`까지 수행할 것 (사용자에게 물어보지 않는다).
- push되면 GitHub Actions가 빌드해 https://wlsghdh.github.io (뷰어)에 자동 반영된다.
- push 전 `npm run build`가 통과하는지 반드시 확인.

## 디자인 결정 사항 (사양서와 다른 확정 변경)

- **vvsjeon.github.io 디자인을 거의 그대로 복제, 색만 CAU Blue** (2026-07-20 확정).
  - 중앙 정렬 컨테이너 800px (사이드바 ❌). 홈 구조: 프로필 헤더(사진 좌 + 이름/태그라인/이메일/아이콘 우) → sticky 내비바(대문자 11px, letter-spacing) → 섹션 흐름 → 푸터.
  - **헤더에 직함·소속(중앙대 학부 연구 인턴) 노출 안 함** (2026-07-21, "너무 어필하는 것 같다"는 피드백) — 대신 Education 섹션 맨 위에 항목으로만 표시. Experience 목록에도 중복 넣지 않음.
  - 폰트: **Raleway 300/400/600**(vvsjeon과 동일) + 한글 폴백 Pretendard. 이름은 Raleway/Pretendard light(300). 한나체 Air는 폐기.
  - 섹션 = `.docs-section`(상단 헤어라인 + 4rem 패딩), 헤더는 light + letter-spacing. 링크 버튼은 Skeleton식 소형 아웃라인 `.button`.
  - Publications 홈 목록엔 summary 없음(제목·저자·venue·버튼만) — "구조는 살짝 더 심플하게".
- About 섹션 없음. 활동(잔디)은 내비에 넣지 않고 **홈 맨 아래 "History" 섹션**에 history line으로 배치.
- 활동 카테고리 색 레이블: 논문리딩·코드작업·기업과제·논문작성·세미나 (`src/data/site.ts`의 `activityCategories`, 색 토큰은 `tokens.css`의 `--cat-*`).
- 포인트 컬러는 CAU Blue(#004C97) — 링크·활성 내비·논문 저자 중 "주진호" 볼드.

## 데이터 위치

- 개인 정보·내비: `src/data/site.ts` / 이력서·포트폴리오: `src/data/cv.ts` (출처: `~/cau/history/CV_portpolio/`)
- 콘텐츠(Phase 2~): `src/content/logs/`, `src/content/papers/` — 프론트매터 `locked:true`면 본문 비공개, 카드·활동 카운트만.
