import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    port: 3000,

    proxy: {
      // During local dev, these paths are forwarded to the backend
      // so the React app can call /generate without hitting CORS issues
      "/generate": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/stats": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // WebSocket proxy for /ws/stats
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },

  build: {
    outDir: "dist",
    sourcemap: true,
  },

  // Makes VITE_API_URL available as import.meta.env.VITE_API_URL
  // In production Docker build, this is overridden by the ARG in the Dockerfile
  envPrefix: "VITE_",
});