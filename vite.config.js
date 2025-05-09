import { defineConfig, defaultClientConditions } from 'vite';
import { resolve } from 'path';
import fs from 'fs';

// Use relative paths for all deployments
const base = './';

export default defineConfig({
  root: './',
  base,
  resolve: {
    conditions: defaultClientConditions
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: './index.html'
      },
      output: {
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]'
      }
    }
  },
  server: {
    port: 3000,
    open: true
  },
  plugins: [
    {
      name: 'copy-readme',
      generateBundle() {
        // Read the README.md file
        const readmePath = resolve(__dirname, 'README.md');
        if (fs.existsSync(readmePath)) {
          const readmeContent = fs.readFileSync(readmePath, 'utf-8');
          // Add the README.md file to the build output
          this.emitFile({
            type: 'asset',
            fileName: 'README.md',
            source: readmeContent
          });
        }
      }
    }
  ]
});
