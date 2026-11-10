/** Research Log 카테고리 5개 (고정). 색은 tokens.css 의 --cat-* */
export const categories = [
  { id: 'paper-reading', label: '논문리딩', en: 'Reading', color: 'var(--cat-reading)' },
  { id: 'coding', label: '코드작업', en: 'Coding', color: 'var(--cat-coding)' },
  { id: 'industry', label: '기업과제', en: 'Industry', color: 'var(--cat-industry)' },
  { id: 'writing', label: '논문작성', en: 'Writing', color: 'var(--cat-writing)' },
  { id: 'seminar', label: '세미나', en: 'Seminar', color: 'var(--cat-seminar)' },
] as const;

export type CategoryId = (typeof categories)[number]['id'];
export const categoryIds = categories.map((c) => c.id) as readonly string[];
export const categoryById = Object.fromEntries(categories.map((c) => [c.id, c])) as Record<
  CategoryId,
  (typeof categories)[number]
>;
