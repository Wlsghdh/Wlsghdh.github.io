# 연구 기록 사이트 (wlsghdh.github.io)

개인 연구자 포트폴리오 + 일일 기록 + Reading Papers 사이트.
Astro + Tailwind v4 + CSS 변수 토큰. (사양서: `research-journal-site-plan.md` v4)

## 로컬 실행

```bash
npm install
npm run dev     # http://localhost:4321
npm run build   # 정적 빌드 → dist/
npm run preview # 빌드 결과 미리보기
```

## 구조

```
public/fonts/        셀프 호스팅 폰트 (한나체 Air, Pretendard, JetBrains Mono) — CDN 폴백 있음
public/profile.jpg   증명사진
src/data/site.ts     이름·연락처·내비·활동 카테고리 (여기만 고치면 전체 반영)
src/data/cv.ts       학력·경력·논문·프로젝트·수상·스킬 (항목 추가만 하면 홈/CV 반영)
src/styles/tokens.css  색/타이포/레이아웃 토큰 (CAU Blue #004C97, 카테고리 색 --cat-*)
src/layouts/BaseLayout.astro   상단 sticky 내비바 + 단일 컬럼(720px)
src/components/TopNav.astro    상단바 (이름 로고 + 앵커 내비, 스크롤 스파이)
src/components/ProfileHeader.astro  홈 상단 프로필 (사진·이름·직함·링크)
src/pages/index.astro  홈 (Publications·Projects·Education·Experience·Awards·Skills·Contact·History)
```

## Phase 진행 상황

- [x] **Phase 1** 뼈대 + 디자인 (토큰·폰트·레이아웃·사이드바·홈 스켈레톤)
- [ ] Phase 2 콘텐츠 (logs / papers 컬렉션 + 목록·상세 + locked)
- [ ] Phase 3 잔디 + 스트릭·통계 + `npm run today`
- [ ] Phase 4 포트폴리오 (/cv, /weekly, tags)
- [ ] Phase 5 Decap CMS(/admin) + GitHub Actions 배포 + RSS/OG
