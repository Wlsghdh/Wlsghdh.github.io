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

## 기사 추가 (Press 섹션)

- 사용자가 기사 링크를 주면 `profile.json`의 `press` 배열 **맨 앞**에 추가한다. 화면에서는 Publications 바로 아래 "Press" 섹션에 보인다 (`src/components/Press.astro`).
- 필드: `{ "date": "Sep. 19, 2025", "title": "기사 원제목 그대로", "outlet": "매체 영문명", "note": "한 줄 영문 설명(선택)", "url": "기사 주소", "image": "/assets/press/<yyyy-mm>-<매체>-<키워드>.jpg" }`
- 썸네일은 기사 대표 사진(og:image 또는 본문 사진 원본)을 받아 `public/assets/press/`에 JPEG로 둔다. 화면에서는 175px 폭 16:10 칸에 꽉 채워진다.
- 기사에 없는 내용은 note 에 쓰지 않는다.

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
- **3색만 쓴다** (2026-11-23 사용자 지시, CVPR·NeurIPS 논문 PDF 느낌): 흰 종이 `#ffffff` + 검정에 가까운 먹색 `#16181d` + CVPR 템플릿 링크색 cvprblue `--accent: #3682be`(링크, News 날짜 배지, 섹션 제목 앞 짧은 막대, 내 이름 밑줄). 생성 GIF도 이 색을 쓴다. 검정 배경, 다크모드, 테마 토글은 넣지 않는다.
- **구조는 jiiiisoo.github.io(Jon Barron 템플릿 계열)를 따른다** (2026-11-17, chaemin-hwang 스타일을 대체):
  - 상단 메뉴 바 없음. 본문 폭 940px(2026-11-23 사용자 요청으로 800에서 넓힘).
  - 헤더: 왼쪽에 가운데 정렬 영문 이름만(굵게 38px, 한글 이름은 넣지 않는다). 이름 아래 소속 줄은 2026-11-23 사용자 요청으로 뺐다(`headline` 데이터는 남음) + 소개 문단 + "Email / GitHub / Blog" 텍스트 링크. 오른쪽에 200px 정사각 사진(테두리 없음, 모바일 130px). 키워드 pill, 태그라인은 넣지 않는다.
  - 섹션 제목: 23px 굵게, 앞에 4×18px 코랄 막대. 밑줄·구분선 없음.
  - News: 날짜 pill 배지(포인트 색) + 한 줄, 항목 사이 얇은 선.
  - 논문과 프로젝트: 왼쪽 175px 썸네일(contain), 제목 굵게, 저자(내 이름 굵게+밑줄), 학회 이탤릭, 링크는 "PDF / arXiv / Code" 텍스트. 마우스를 올리면 행 배경이 살짝 어두워진다. 논문 TL;DR 요약은 2026-11-23 사용자 요청으로 화면에서 뺐다(`summary` 데이터는 남아 있다).
  - Experience는 왼쪽에 가로로 긴 네모 로고(132×56), Education은 동그라미 로고(64px). profile.json 항목의 `logo` 필드에 `/assets/logos/...` 경로를 적는다. 로고는 Wikimedia Commons(학교)와 aimmo.co.kr(AIMMO)에서 받았다.
  - 포인트 색은 `--accent` 하나만 쓴다(링크, 날짜 배지).
- **Research Log 섹션은 화면에서 뺐다** (2026-11-17 사용자 지시). `log.json`, `ResearchLog.astro`는 남아 있고 일일 기록 append 규칙도 그대로다.
- **사이트 언어는 영어.** profile.json 내용도 영어로 쓴다. 공동저자 영문 표기는 `authors`, 한글은 `authorsKo`에 둔다. log.json 항목은 사용자가 쓴 언어 그대로 둔다.
- Skills 섹션은 화면에서 뺐다. 데이터는 profile.json 에 남아 있다.
- 모든 publication과 project에 썸네일 칸이 있다. 파일이 없으면 `/assets/thumb-placeholder.svg`가 보인다.
  - 논문: `public/assets/publications/P2.gif`처럼 id 이름으로 넣는다. P4.gif는 사용자가 준 실제 로봇 실험 영상(trial_05_posA05.mp4, mp4v 코덱이라 브라우저 재생 불가)을 잘라 만든 GIF다. P2.gif와 P3.gif는 각 논문 PDF의 실제 그림 여러 장(증강 예시, 파이프라인, 결과 그래프·표)이 차례로 넘어가는 슬라이드 GIF다. P3 첫 장의 센서 신호만 PDF 원본이 흐려서 Claude가 다시 그렸다. 만드는 스크립트는 scratchpad gifgen/slides.mjs(pdfjs-dist로 PDF를 렌더링해 잘라 붙임). 논문 썸네일은 모두 175×175 정사각 칸(cover)이다. 프로젝트 4개는 Claude가 그린 애니메이션 GIF다.
  - 프로젝트: `public/assets/projects/carnerf.gif`, `etf-with-ai.gif`, `jump-ai-2025.gif`, `personal-color.gif`.
  - 다른 이름이나 확장자를 쓰려면 profile.json 의 `thumb`/`image` 경로를 바꾼다.
- publication에 `"hidden": true`를 넣으면 데이터에는 남고 화면에서만 빠진다. P1(BMVC 탈락)이 지금 이 상태다. 다시 내려면 `hidden`을 지운다.
- 단일 중앙 컬럼, 본문 폭 940px, 사이드바 없음.
- 폰트는 **Source Serif 4** (Google Fonts, 2026-11-23 사용자 지시). 논문 느낌 세리프이면서 화면에서 선명하다. Times → Computer Modern 을 거쳐 바꿨다. 한글은 Pretendard. 본문 16px, 행간 1.65.
- **이름 표기는 `Jinho Ju`** (2026-11-23, 하이픈 없이). 논문 저자 목록도 같다.
- **하이픈·대시 기호는 거의 쓰지 않는다** (2026-11-23 사용자 지시). 기간은 `Sep. 2025 ~ Dec. 2025`처럼 물결표, 제목 구분은 쉼표. 고유명사(Chung-Ang, 논문 제목)는 그대로 둔다.
- 썸네일 GIF는 400×400으로 만든다(레티나에서 175px 칸이 흐려지지 않게). PDF는 scale 6으로 렌더링해서 자른다.
- 줄바꿈: 문단은 `text-wrap: pretty`, 제목은 `balance`. 단어 안 하이픈(Jin-Ho, Chung-Ang)은 `src/lib/text.ts`의 `nb()`로 줄이 안 바뀌게 한다.
- 색: 본문 `#16181d`, 보조 `#5c6068`, 포인트 cvprblue 1개, 선 `#e4e6ea`.
- 그라데이션, 그림자, 카드 테두리, 둥근 큰 박스는 쓰지 않는다.
- 모바일 380px에서 가로 스크롤이 생기면 안 된다.
- 섹션 순서: Header → News → Publications → Press → Experience → Education → Honors & Awards(+Certificates) → Contact
- **Contact는 가상 명함 한 장**(2026-11-23 사용자 지시, 참고: fid-design.github.io/Interior 의 CONTACT). `src/components/Contact.astro`. 흰 카드 앞면 한 장뿐이다(뒷면 없음): 이름, "Undergraduate Researcher · LLM / VLA", E./G./B. 연락처(전화번호는 넣지 않는다), 오른쪽 위 로봇 선 아이콘(SVG). 기울이면 글자가 흐려져서 똑바로 두고, 마우스를 올리면 3px 떠오르며 테두리가 포인트 색이 된다. 소속 표기, 사진, 안내 문구, 그림자는 넣지 않는다(2026-11-23 사용자 지시). cqw 단위는 카드 안쪽 `.body`에서만 쓴다.
- **Projects 섹션은 화면에서 뺐다** (2026-11-20 사용자 지시). profile.json `projects`와 `Projects.astro`는 남아 있다.
- 2026-11-20에 "섹션 이름 왼쪽 칸 + 내용 오른쪽" 2단 레이아웃을 시도했지만 사용자가 지금 구조를 더 좋아해서 되돌렸다. 구조는 바꾸지 말 것.
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
