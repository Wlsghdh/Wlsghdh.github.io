# 연구 기록 사이트 (wlsghdh.github.io)

개인 학술 사이트 + 날짜별 연구 기록(Research Log). Astro 정적 빌드, GitHub Pages 배포.

```bash
npm ci
npm run dev     # http://localhost:4321
npm run build   # dist/ (log.json 스키마 오류 시 실패)
```

- 매일 기록: `src/data/log.json` 에 항목 append
- 프로필·논문·수상 등: `src/data/profile.json`
- 이미지/PDF: `public/assets/{publications,awards,certificates,projects,cv}/`

운영 규칙과 스키마는 [CLAUDE.md](CLAUDE.md) 참고.
