/**
 * 호환용 얇은 래퍼 — 실제 데이터는 src/data/profile.json 에 있다.
 * astro.config.mjs 가 site.url 을 쓰므로 남겨 둔다.
 */
import profile from './profile.json';
import { categories } from './categories';

const p = profile.person;

export const site = {
  githubUsername: p.githubUsername,
  authorKo: p.nameKo,
  authorEn: p.nameEn,
  tagline: p.tagline,
  role: p.role,
  rolePeriod: p.rolePeriod,
  affiliation: p.affiliation,
  email: p.email,
  blog: p.blog,
  scholar: p.scholar,
  title: profile.site.title,
  description: profile.site.description,
  github: `https://github.com/${p.githubUsername}`,
  url: `https://${p.githubUsername.toLowerCase()}.github.io`,
};

/** 홈 앵커 내비 (헤더 아래 한 줄) */
export const navSections = [
  { id: 'news', label: 'News' },
  { id: 'log', label: 'Log' },
  { id: 'publications', label: 'Publications' },
  { id: 'projects', label: 'Projects' },
  { id: 'experience', label: 'Experience' },
  { id: 'education', label: 'Education' },
  { id: 'awards', label: 'Honors' },
] as const;

/** 이전 이름 유지 */
export const activityCategories = categories;
