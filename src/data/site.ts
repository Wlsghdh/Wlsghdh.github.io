/**
 * 사이트 전역 정보 — 여기 한 곳만 고치면 헤더·메타·링크에 모두 반영됩니다.
 */
export const site = {
  /** GitHub username (저장소 이름 = `${githubUsername}.github.io`) */
  githubUsername: 'Wlsghdh',

  authorKo: '주진호',
  authorEn: 'Jin-Ho Ju',
  tagline: '데이터로 사람과 문제를 잇는 연구자.',

  /** 직함 */
  role: '중앙대학교 학부 연구 인턴',
  rolePeriod: '2026.07.13 ~ 현재',
  affiliation: '중앙대학교 (Chung-Ang University)',

  /** 연락처 */
  email: 'wlsgh20728@naver.com',
  blog: 'https://edu-data.tistory.com',
  /** (선택) Google Scholar 프로필 URL — 없으면 빈 문자열 유지(링크 숨김) */
  scholar: '',

  /** 사이트 메타 */
  title: '연구 기록',
  description: '주진호 — 일일 연구 기록과 Reading Papers, 학술 포트폴리오',

  get github() {
    return `https://github.com/${this.githubUsername}`;
  },
  get url() {
    return `https://${this.githubUsername.toLowerCase()}.github.io`;
  },
};

/** 상단바 내비게이션 (홈 앵커) */
export const navSections = [
  { id: 'publications', label: 'Publications' },
  { id: 'projects', label: 'Projects' },
  { id: 'education', label: 'Education' },
  { id: 'experience', label: 'Experience' },
  { id: 'awards', label: 'Awards' },
  { id: 'contact', label: 'Contact' },
] as const;

/**
 * History line 활동 카테고리 — 색 레이블.
 * logs/papers 프론트매터의 category 값과 매핑 (Phase 2~3에서 사용).
 */
export const activityCategories = [
  { id: 'paper-reading', label: '논문리딩', color: 'var(--cat-reading)' },
  { id: 'coding', label: '코드작업', color: 'var(--cat-coding)' },
  { id: 'industry', label: '기업과제', color: 'var(--cat-industry)' },
  { id: 'paper-writing', label: '논문작성', color: 'var(--cat-writing)' },
  { id: 'seminar', label: '세미나', color: 'var(--cat-seminar)' },
] as const;
