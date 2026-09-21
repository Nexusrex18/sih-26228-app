import type { NextConfig } from "next";

/**
 * Static export.
 *
 * `output: "export"` produces a directory of plain files that Flask serves and that works
 * with no Node process at runtime. The build needs a network (like `make wheelhouse` does);
 * the *running* system does not, so `docker run --network none` still serves a complete
 * dashboard — which is demo step 0, the thing a judge actually watches.
 *
 * Because there is no server, every route is a static shell and the data arrives from the
 * Flask API on the client. That is why scans are addressed by query string (`/scan?id=...`)
 * rather than by a dynamic segment: a dynamic segment would need `generateStaticParams`,
 * and scan ids do not exist at build time.
 */
const nextConfig: NextConfig = {
  output: "export",
  distDir: "out",
  // Flask serves the export from /app/, so every asset URL has to carry that prefix.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? "/app",
  assetPrefix: process.env.NEXT_PUBLIC_BASE_PATH ?? "/app",
  trailingSlash: true,
  images: {
    // No image optimiser without a server, and no remote loader on an air gap.
    unoptimized: true,
  },
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
