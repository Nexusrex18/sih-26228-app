import type { Metadata, Viewport } from "next";
import { Toaster } from "sonner";
import { AppProvider } from "@/components/shell";
import "./globals.css";

/**
 * Fonts are declared in `globals.css` as `@font-face` over files in `public/fonts/`,
 * fetched once by `scripts/vendor_web_assets.py`.
 *
 * Not `next/font/google`: that reaches out to Google's servers during `next build`, which
 * makes the build itself network-dependent and silently falls back when it cannot. Files on
 * disk make the dependency visible, hashed in a manifest, and reproducible — and the
 * `@font-face` declaration degrades to the system stack if a file is missing, which is a
 * legible failure rather than a mysterious one.
 */

export const metadata: Metadata = {
  title: "CV Integrity Assurance",
  description:
    "Analyst governance for dataset, model and inference-provenance assurance (SIH 26228).",
};

export const viewport: Viewport = {
  themeColor: "#0e1219",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="scroll-slim">
        <AppProvider>{children}</AppProvider>
        <Toaster
          position="bottom-right"
          toastOptions={{
            classNames: {
              toast:
                "!bg-surface-raised !border !border-line-strong !text-ink !rounded",
              description: "!text-ink-muted",
            },
          }}
        />
      </body>
    </html>
  );
}
