# 연구 기록 사이트 (username.github.io)

개인 연구자 포트폴리오 + 일일 기록 + Reading Papers 사이트.
Astro + Tailwind v4 + CSS 변수 토큰. (사양서: `research-journal-site-plan.md` v4)

## 로컬 실행

```bash
npm install
npm run dev     # http://localhost:4321
npm run build   # 정적 빌드 → dist/
npm run preview # 빌드 결과 미리보기
```

## 처음 할 일 (본인 정보 채우기)

1. `src/data/site.ts` — `githubUsername`, `authorKo`, `authorEn`, (선택) `scholar` 채우기. 나머지(이메일·직함)는 시드값 확인.
2. `public/profile.jpg` — 증명사진 넣기 (없으면 placeholder SVG가 자동 표시됨).
3. 폴더/저장소 이름을 `<본인 username>.github.io`로 변경.

## 구조

```
public/fonts/        셀프 호스팅 폰트 (한나체 Air, Pretendard, JetBrains Mono) — CDN 폴백 있음
src/data/site.ts     이름·연락처·내비 등 전역 정보 (여기만 고치면 전체 반영)
src/styles/tokens.css  색/타이포/레이아웃 토큰 (CAU Blue #004C97)
src/layouts/BaseLayout.astro   2단 레이아웃 (좌 300px 사이드바 + 우 720px 콘텐츠)
src/components/ProfileSidebar.astro  sticky 프로필 + 앵커 내비(스크롤 스파이)
src/pages/index.astro  홈 스켈레톤 (About·활동·Reading·Education·Experience·Awards·Contact)
```

## Phase 진행 상황

- [x] **Phase 1** 뼈대 + 디자인 (토큰·폰트·레이아웃·사이드바·홈 스켈레톤)
- [ ] Phase 2 콘텐츠 (logs / papers 컬렉션 + 목록·상세 + locked)
- [ ] Phase 3 잔디 + 스트릭·통계 + `npm run today`
- [ ] Phase 4 포트폴리오 (/cv, /weekly, tags)
- [ ] Phase 5 Decap CMS(/admin) + GitHub Actions 배포 + RSS/OG
