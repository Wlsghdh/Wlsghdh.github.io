/** 단어 안 하이픈(Jin-Ho, Chung-Ang)에서 줄이 바뀌지 않도록 non-breaking hyphen(U+2011)으로 바꾼다 */
export const nb = (s: string) => s.replace(/(\p{L})-(\p{L})/gu, '$1‑$2');
