# 프로젝트 규칙 — wlsghdh.github.io

주진호(Jin-Ho Ju)의 개인 학술 사이트 + 일일 연구 기록. Astro 5 + Tailwind v4(리셋 용도), 정적 빌드.

## 파일 역할

| 성격 | 파일 | 갱신 주기 |
|---|---|---|
| 매일 쌓이는 기록 | `src/data/log.json` | 매일 (append만) |
| 고정 정보 (프로필·News·논문·프로젝트·경력·학력·수상·자격증·스킬) | `src/data/profile.json` | 실적 생길 때만 |
| 카테고리 정의 (id·라벨·색) | `src/data/categories.ts` | 거의 안 바뀜 |
| 데이터 로더 + log.json 스키마 검사 | `src/lib/data.ts` | 리디자인 때만 |
| 화면 | `src/pages/index.astro`(섹션 순서), `src/components/*`, `src/layouts/BaseLayout.astro` | 리디자인 때만 |
| 디자인 토큰 / 전역 스타일 | `src/styles/tokens.css`, `src/styles/global.css` | 리디자인 때만 |
| 호환용 래퍼 (고치지 말 것) | `src/data/site.ts`, `src/data/cv.ts` | — |
| 업로드 자료 | `public/assets/{publications,awards,certificates,projects,cv}/` | 자료 생길 때 |

`src/components/TopNav.astro`, `ProfileHeader.astro`와 `public/fonts/`의 Raleway, 한나체, JetBrains Mono는 지금 쓰지 않는다. 삭제는 사용자 지시가 있을 때만 한다.

## 일일 기록 추가 (가장 자주 하는 작업)

- **`src/data/log.json` 에만 항목을 append 한다. 다른 파일은 건드리지 않는다.**
- 기존 항목은 수정하거나 삭제하지 않는다.
- 사용자가 적지 않은 사실은 지어내지 않는다. 모르면 그 필드를 생략한다.
- 링크를 안 적은 논문은 arXiv나 학회 공식 링크를 찾아 채운다. 확실하지 않으면 비우고 사용자에게 묻는다.
- tags 는 2~4개, 영문 소문자.
- **industry(기업과제)는 과제명과 "비공개"만 적는다. 상세 내용은 쓰지 않는다** (대외비).
- 스키마가 틀리면 `npm run build`가 실패하므로 push 전에 반드시 빌드한다.

### log.json 스키마

```json
{
  "date": "2026-11-10",              // 필수, YYYY-MM-DD (실제 활동 날짜)
  "category": "paper-reading",       // 필수, 아래 5개 중 하나
  "title": "RT-2: Vision-Language-Action Models",  // 필수
  "link": "https://arxiv.org/abs/2307.15818",      // 선택
  "venue": "CoRL 2023",             // 선택
  "summary": "한 문장 요약",          // 필수
  "takeaway": "내 연구와의 연결점/의문점",  // 선택
  "tags": ["vla", "generalization"]  // 선택, 2~4개
}
```

카테고리: `paper-reading`(논문리딩) / `coding`(코드작업) / `industry`(기업과제) / `writing`(논문작성) / `seminar`(세미나)

화면 구성:
- 최근 12개월 히트맵. 칸에 마우스를 올리면 그날 기록이 박스로 뜨고, 클릭하면 아래 목록이 그날만 보인다("Show all"로 해제).
- 히트맵은 첫 기록(또는 1년 전)부터 오늘까지 그리고, 길어지면 가로 스크롤한다. 처음 열면 오늘 쪽(오른쪽 끝)이 보인다.
- 날짜 역순 리스트(하루 단위로 묶음), CSS만 쓰는 카테고리 필터(기본 탭 "Recent Activity"), 최근 5일 + "Show more"를 누를 때마다 7일씩.

## 실적 업데이트 (논문 억셉, 수상 등)

- `profile.json`의 해당 배열(`publications`, `awards` 등)에 항목을 추가하고, `news` 맨 앞에도 한 줄 추가한다. News는 화면에 최근 8개만 보인다.
- 필요하면 log.json 에 `writing` 카테고리로 한 줄을 남긴다.
- publications 항목 필드: `id, year, title, authors[], coFirst[], venue, note, summary, pdf, thumb, links[{label,url}]`
  - 내 이름(`person.nameKo`)은 자동으로 bold 처리된다. `coFirst`에 넣은 이름 뒤에는 `*`가 붙는다.
  - 논문 PDF는 `public/assets/publications/P1.pdf`에 두고 `"pdf": "/assets/publications/P1.pdf"`로 적는다. 그러면 제목과 [PDF] 칩이 PDF로 연결된다.
  - 썸네일은 `public/assets/publications/P1.png`에 두고 `"thumb": "/assets/publications/P1.png"`로 적는다. 16:10 비율을 권장한다.
- awards / certificates / projects 의 `image` 필드도 같은 방식으로 `/assets/awards/...` 등을 가리키게 한다. 값이 있으면 제목이 그 파일로 연결된다.
- CV PDF는 `public/assets/cv/cv.pdf`에 두고 `person.cv`에 `/assets/cv/cv.pdf`를 적으면 헤더에 CV 링크가 생긴다. `person.scholar`도 마찬가지다.
- `person.intro`(문자열 배열, 2~4문장)를 채우면 헤더에서 tagline 대신 그 문장들이 보인다.

## 디자인 원칙 (2026-11-10 확정 — 바꾸지 말 것)

- CVPR 논문처럼 아주 단순하게 만든다. 얇은 1px 선, 정갈한 활자, 여백과 활자 크기로만 위계를 만든다. 참고 대상: seokju-cho.github.io, yejich.github.io, 연세대 전해곤 교수 연구실 석사생 사이트.
- **밝은 베이지 배경(`#faf7f0`) 하나만 쓴다. 다크모드와 테마 토글은 넣지 않는다** (2026-11-10 사용자 지시).
- **구조는 chaemin-hwang.github.io를 따른다** (2026-11-10):
  - 상단 sticky 바: 이름 + 섹션 링크.
  - 헤더: 세리프(EB Garamond) 영문 이름, 대문자 소속 줄 "Intelligent Semiconductor Engineering · Chung-Ang University", 소개 문단, 키워드 pill, 아이콘이 붙은 테두리 링크 버튼. 사진은 오른쪽에 테두리 없이 둔다. 태그라인은 넣지 않는다.
  - 섹션 제목: 대문자 + 글자 폭만큼 짧은 밑줄. 항목 사이는 얇은 구분선.
  - 논문과 프로젝트: 왼쪽에 120px 정사각 썸네일.
- **사이트 언어는 영어.** profile.json 내용도 영어로 쓴다. 공동저자 영문 표기는 `authors`, 한글은 `authorsKo`에 둔다. log.json 항목은 사용자가 쓴 언어 그대로 둔다.
- Skills 섹션은 화면에서 뺐다. 데이터는 profile.json 에 남아 있다. Contact 섹션도 없다. 연락처는 헤더 버튼으로 대신한다.
- 모든 publication과 project에 썸네일 칸이 있다. 파일이 없으면 `/assets/thumb-placeholder.svg`가 보인다.
  - 논문: `public/assets/publications/P2.png`처럼 id 이름으로 넣는다.
  - 프로젝트: `public/assets/projects/carnerf.gif`, `etf-with-ai.gif`, `jump-ai-2025.png`, `personal-color.png`.
  - 다른 이름이나 확장자를 쓰려면 profile.json 의 `thumb`/`image` 경로를 바꾼다.
- publication에 `"hidden": true`를 넣으면 데이터에는 남고 화면에서만 빠진다. P1(BMVC 탈락)이 지금 이 상태다. 다시 내려면 `hidden`을 지운다.
- 단일 중앙 컬럼, 본문 폭 740px, 사이드바 없음.
- 폰트는 Pretendard 하나. 라틴 글리프가 Inter 계열이라 한글, 영문, 숫자가 흔들리지 않는다. 본문 15px, 행간 1.65, 숫자는 tabular.
- 색: 본문 `#1a1a1a`, 보조 `#6b7280`, 링크 액센트 1개, 선 `#e7e7e7`. 카테고리 5색은 채도를 낮게 쓴다.
- 그라데이션, 그림자, 카드 테두리, 둥근 큰 박스는 쓰지 않는다.
- 모바일 380px에서 가로 스크롤이 생기면 안 된다.
- 섹션 순서: Header → News → Research Log → Publications → Projects → Experience → Education → Honors & Awards(+Certificates)
- 디자인 수정은 한 번에 한 가지만 한다. 데이터 추가와 디자인 수정을 한 커밋에 섞지 않는다.

## Git / 배포

- **커밋과 push는 Claude가 알아서 한다** (사용자 허가, 2026-11-10). 단 push 전에 `npm run build`가 통과해야 한다.
- 큰 리디자인은 브랜치에서 작업한 뒤 main 에 머지한다.
- main 에 push하면 GitHub Actions(`.github/workflows/deploy.yml`, `withastro/action`)가 https://wlsghdh.github.io 에 배포한다. 반영까지 1~2분 걸린다.
- 커밋 메시지:
  - 일일 기록: `log: 2026-11-10 (paper-reading ×2, coding ×1)`
  - 실적: `profile: <무엇>`
  - 디자인/코드: `design: <무엇>` / `fix: <무엇>`

## 빌드

```
npm ci
npm run dev       # http://localhost:4321
npm run build     # dist/ — log.json 스키마 오류 시 실패
npm run preview
```
