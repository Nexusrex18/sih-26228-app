import type { Config } from "tailwindcss";

/**
 * The design system, as tokens.
 *
 * Direction: a calibrated instrument, not a report. The person using this is an acceptance
 * officer deciding whether to accept a dataset that may have been deliberately poisoned,
 * and who will be asked later to justify that decision.
 *
 * Two choices drive everything else:
 *
 *  - Mono is a PRIMARY face, not a utility one. Digests, ledger sequence numbers and
 *    identifiers are the content here, and a proportional face makes two hashes that differ
 *    look alike.
 *  - Signal colour is reserved for state. `accent` is structural (focus, links, the seq
 *    rail) and is never a disposition; the disposition hues are never decorative.
 */
const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          deep: "hsl(var(--surface-deep) / <alpha-value>)",
          DEFAULT: "hsl(var(--surface) / <alpha-value>)",
          panel: "hsl(var(--surface-panel) / <alpha-value>)",
          raised: "hsl(var(--surface-raised) / <alpha-value>)",
        },
        line: {
          DEFAULT: "hsl(var(--line) / <alpha-value>)",
          strong: "hsl(var(--line-strong) / <alpha-value>)",
        },
        ink: {
          strong: "hsl(var(--ink-strong) / <alpha-value>)",
          DEFAULT: "hsl(var(--ink) / <alpha-value>)",
          muted: "hsl(var(--ink-muted) / <alpha-value>)",
          faint: "hsl(var(--ink-faint) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "hsl(var(--accent) / <alpha-value>)",
          dim: "hsl(var(--accent-dim) / <alpha-value>)",
        },
        quarantine: "hsl(var(--quarantine) / <alpha-value>)",
        review: "hsl(var(--review) / <alpha-value>)",
        accept: "hsl(var(--accept) / <alpha-value>)",
        pending: "hsl(var(--pending) / <alpha-value>)",
        absent: "hsl(var(--absent) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "var(--font-sans)", "ui-sans-serif", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.08em" }],
        eyebrow: ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.16em" }],
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "6px",
        lg: "10px",
        xl: "14px",
      },
      // Depth is a hairline (see `.glass`). The only soft shadow is `.float-shadow`, for
      // layers that float over the page. `glow` is a focus ring, not a halo.
      boxShadow: {
        glow: "0 0 0 3px hsl(var(--accent) / 0.22)",
      },
      // Motion tokens (Emil Kowalski's curves): strong ease-out for entry and press
      // feedback, strong ease-in-out for on-screen movement. Never ease-in on UI.
      transitionTimingFunction: {
        "out-strong": "cubic-bezier(0.23, 1, 0.32, 1)",
        "in-out-strong": "cubic-bezier(0.77, 0, 0.175, 1)",
      },
      transitionDuration: {
        press: "160ms",
      },
      keyframes: {
        // Ambient motion, each tied to something true rather than to decoration.
        sweep: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        "hatch-drift": {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "12.73px 12.73px" },
        },
        "rail-travel": {
          "0%": { top: "0%", opacity: "0" },
          "12%": { opacity: "0.6" },
          "88%": { opacity: "0.6" },
          "100%": { top: "100%", opacity: "0" },
        },
        breathe: {
          "0%, 100%": { boxShadow: "0 0 0 0 hsl(var(--accept) / 0.45)" },
          "50%": { boxShadow: "0 0 0 6px hsl(var(--accept) / 0)" },
        },
        alarm: {
          "0%, 100%": { boxShadow: "0 0 0 0 hsl(var(--quarantine) / 0.55)", opacity: "1" },
          "50%": { boxShadow: "0 0 0 7px hsl(var(--quarantine) / 0)", opacity: "0.7" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        sweep: "sweep 9s linear infinite",
        "hatch-drift": "hatch-drift 24s linear infinite",
        "rail-travel": "rail-travel 7s cubic-bezier(.45,0,.55,1) infinite",
        breathe: "breathe 3.2s ease-in-out infinite",
        alarm: "alarm 1.8s ease-in-out infinite",
        shimmer: "shimmer 2.4s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
