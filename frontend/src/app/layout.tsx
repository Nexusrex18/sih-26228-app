import type { Metadata, Viewport } from "next";
import { Toaster } from "sonner";
import { AppProvider } from "@/components/shell";
/* IBM Plex, bundled from npm (@fontsource) so the woff2 files ship inside the static
 * export: no Google Fonts request at build time or at runtime, which the air-gapped demo
 * needs. Plex is an engineering-instrument family — Condensed for display headings, Sans
 * for prose, Mono for digests, ledger seqs and every number that can change. Latin subset
 * and only the weights in use, to keep the bundle small. */
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-sans-condensed/latin-500.css";
import "@fontsource/ibm-plex-sans-condensed/latin-600.css";
import "@fontsource/ibm-plex-sans-condensed/latin-700.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";
import "./globals.css";

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
