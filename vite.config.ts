import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const __dirname = dirname(fileURLToPath(import.meta.url));

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
    rollupOptions: {
      input: {
        nano: resolve(__dirname, 'index.html'),
        twotower: resolve(__dirname, 'twotower/index.html'),
        diffusion: resolve(__dirname, 'diffusion/index.html'),
      },
    },
  },
  test: {
    include: ['tests/*.test.ts'],
    exclude: ['tests/e2e/**', '**/node_modules/**', '**/dist/**'],
    environment: 'node',
  },
});
