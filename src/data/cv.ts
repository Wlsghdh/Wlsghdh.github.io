/**
 * 호환용 얇은 래퍼 — 실제 데이터는 src/data/profile.json 에 있다. 여기는 고치지 말 것.
 */
import profile from './profile.json';

export const { education, experience, publications, projects, awards, skills, certificates } = profile;
