// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import { site } from './src/data/site.ts';

// 저장소가 username.github.io(루트 배포)이므로 base는 '/'
export default defineConfig({
  site: site.url,
  base: '/',
  vite: {
    plugins: [tailwindcss()],
  },
});
