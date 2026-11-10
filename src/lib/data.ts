/**
 * 데이터 로더 — 화면 코드는 이 파일을 통해서만 src/data/*.json 을 읽는다.
 * log.json 은 빌드 시점에 스키마를 검사하고, 틀린 항목이 있으면 빌드를 멈춘다.
 */
import profileJson from '../data/profile.json';
import logJson from '../data/log.json';
import { categoryIds, type CategoryId } from '../data/categories';

export type Link = { label: string; url: string };

export interface LogEntry {
  date: string; // YYYY-MM-DD
  category: CategoryId;
  title: string;
  link?: string;
  venue?: string;
  summary: string;
  takeaway?: string;
  tags?: string[];
}

export const profile = profileJson;
export type Publication = (typeof profileJson.publications)[number];

function validateLog(raw: unknown): LogEntry[] {
  if (!Array.isArray(raw)) throw new Error('log.json: 최상위는 배열이어야 합니다.');
  const errors: string[] = [];
  raw.forEach((e: any, i) => {
    const where = `log.json[${i}]`;
    if (typeof e !== 'object' || e === null) return errors.push(`${where}: 객체가 아님`);
    if (typeof e.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(e.date) || isNaN(Date.parse(e.date)))
      errors.push(`${where}.date: YYYY-MM-DD 형식이어야 함 (${e.date})`);
    if (!categoryIds.includes(e.category))
      errors.push(`${where}.category: ${categoryIds.join(' | ')} 중 하나여야 함 (${e.category})`);
    for (const k of ['title', 'summary'])
      if (typeof e[k] !== 'string' || !e[k].trim()) errors.push(`${where}.${k}: 필수 문자열`);
    for (const k of ['link', 'venue', 'takeaway'])
      if (e[k] !== undefined && typeof e[k] !== 'string') errors.push(`${where}.${k}: 문자열이어야 함`);
    if (e.tags !== undefined && (!Array.isArray(e.tags) || e.tags.some((t: unknown) => typeof t !== 'string')))
      errors.push(`${where}.tags: 문자열 배열이어야 함`);
  });
  if (errors.length) throw new Error('log.json 스키마 오류:\n' + errors.join('\n'));
  return raw as LogEntry[];
}

/** 날짜 역순. 같은 날짜는 파일에 적힌 순서 유지 */
export const log: LogEntry[] = validateLog(logJson)
  .map((e, i) => ({ e, i }))
  .sort((a, b) => (a.e.date === b.e.date ? a.i - b.i : a.e.date < b.e.date ? 1 : -1))
  .map(({ e }) => e);

/** 빌드 시점의 한국 날짜 (YYYY-MM-DD) */
export function todayKST(): string {
  return new Date(Date.now() + 9 * 3600 * 1000).toISOString().slice(0, 10);
}
