/**
 * 사이트 전역 정보 — 여기 한 곳만 고치면 사이드바·메타·링크에 모두 반영됩니다.
 * TODO 표시된 값을 본인 정보로 채워 주세요.
 */
export const site = {
  /** GitHub username (저장소 이름 = `${githubUsername}.github.io`) */
  githubUsername: 'Wlsghdh',

  /** TODO: 이름 (사이드바에 한나체 Air + CAU Blue로 표시) */
  authorKo: '이름',
  /** TODO: 영문 이름 (논문 저자 표기 등에 사용 예정) */
  authorEn: 'Your Name',

  /** 직함 */
  role: '중앙대학교 학부 연구 인턴',
  rolePeriod: '2026.07.13 ~ 현재',
  affiliation: '중앙대학교 (Chung-Ang University)',

  /** 연락처 */
  email: 'wlsgh30821@gmail.com',
  /** TODO: (선택) Google Scholar 프로필 URL — 없으면 빈 문자열 유지(링크 숨김) */
  scholar: '',

  /** 사이트 메타 */
  title: '연구 기록',
  description: '일일 연구 기록과 Reading Papers, 학술 포트폴리오',

  get github() {
    return `https://github.com/${this.githubUsername}`;
  },
  get url() {
    return `https://${this.githubUsername}.github.io`;
  },
};

/** 홈 앵커 내비게이션 (사이드바에서 사용) */
export const navSections = [
  { id: 'about', label: 'About' },
  { id: 'activity', label: '활동' },
  { id: 'reading', label: 'Reading Papers' },
  { id: 'education', label: 'Education' },
  { id: 'experience', label: 'Experience' },
  { id: 'awards', label: 'Honors & Awards' },
  { id: 'contact', label: 'Contact' },
] as const;
