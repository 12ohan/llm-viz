import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? '/',
  publicDir: false,
  css: {
    postcss: {
      plugins: [],
    },
  },
  server: {
    port: 3000,
    host: '0.0.0.0',
  },
  preview: {
    port: 3000,
    host: '0.0.0.0',
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  test: {
    include: ['tests/*.test.ts'],
    exclude: ['tests/e2e/**', '**/node_modules/**', '**/dist/**'],
    environment: 'node',
  },
});
