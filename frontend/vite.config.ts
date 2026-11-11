import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built files land in ../static, which FastAPI serves at "/". In dev, API calls are proxied to
// the uvicorn server so the UI can run on its own port with hot reload.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../static", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      "/ask": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/ingest": "http://localhost:8000",
    },
  },
});
