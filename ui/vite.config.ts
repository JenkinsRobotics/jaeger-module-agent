import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bundle ships inside the Python package: `python -m jaeger_agent.ui` serves it with no node present.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../jaeger_agent/ui/static", emptyOutDir: true },
  server: { proxy: { "/events": "http://127.0.0.1:8791", "/say": "http://127.0.0.1:8791", "/health": "http://127.0.0.1:8791", "/interrupt": "http://127.0.0.1:8791" } },
  test: { environment: "jsdom" },
});
