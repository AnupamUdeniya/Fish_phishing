import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'

const currentDirectory = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  base: '',
  plugins: [react()],
  build: {
    outDir: '../../extension/dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        popup: resolve(currentDirectory, 'popup.html')
      }
    }
  }
})