"use client";

import { AnimatePresence, motion, useMotionValue, useTransform } from "framer-motion";
import * as React from "react";
import { usePrefersReducedMotion } from "@/components/ui/primitives";
import { cn } from "@/lib/ui";

/**
 * A bottom sheet on phones, a side panel on desktop.
 *
 * The decide flow is the densest thing in this app and it is unusable in a modal box on a
 * 390px screen. On touch it becomes a sheet you can drag away, built to the rules in
 * Apple's *Designing Fluid Interfaces*:
 *
 *  - **1:1 tracking.** The sheet is glued to the finger for the whole drag, not animated
 *    once at the end.
 *  - **Velocity handoff and momentum projection.** Release velocity decides whether it
 *    closes, and that velocity is handed to the spring, so there is no seam between
 *    dragging and animating. A slow drag halfway down springs back; a fast flick from a
 *    few pixels closes.
 *  - **Rubber-banding.** Dragging upward past the top resists progressively instead of
 *    stopping dead.
 *  - **Interruptible.** `dragElastic` plus a spring means a closing sheet can be grabbed
 *    and thrown back without waiting for anything to finish.
 *  - **Symmetric path.** It leaves the way it arrived (D-E: enter and exit along the same
 *    path), so the spatial relationship holds.
 *
 * Springs are critically damped by default (`bounce: 0`); the only bounce is on a
 * momentum-carrying release, because overshoot on something that merely faded in feels
 * wrong while overshoot on something you flicked feels right.
 */

/** Apple's projection from the sample code — exponential decay, not v²/2a. */
function project(velocity: number, decelerationRate = 0.998) {
  return ((velocity / 1000) * decelerationRate) / (1 - decelerationRate);
}

export function Sheet({
  open,
  onClose,
  title,
  description,
  children,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  const reduced = usePrefersReducedMotion();
  const y = useMotionValue(0);
  // The scrim deepens as the sheet is dragged away: the dimming tracks the gesture rather
  // than snapping at the end, so the background never jumps.
  const scrim = useTransform(y, [0, 400], [1, 0]);
  const sheetRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  React.useEffect(() => {
    if (open) {
      y.set(0);
      // Move focus in, so a keyboard user is not left behind the scrim.
      requestAnimationFrame(() => sheetRef.current?.focus());
    }
  }, [open, y]);

  return (
    <AnimatePresence>
      {open ? (
        <>
          <motion.div
            aria-hidden
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ opacity: reduced ? undefined : scrim }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-surface-deep/70 backdrop-blur-sm"
          />
          <motion.div
            ref={sheetRef}
            role="dialog"
            aria-modal="true"
            aria-label={typeof title === "string" ? title : "Panel"}
            tabIndex={-1}
            style={{ y }}
            initial={reduced ? { opacity: 0 } : { y: "100%" }}
            animate={reduced ? { opacity: 1 } : { y: 0 }}
            exit={reduced ? { opacity: 0 } : { y: "100%" }}
            // Critically damped: it arrives and settles, it does not wobble.
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            drag={reduced ? false : "y"}
            dragConstraints={{ top: 0, bottom: 0 }}
            // Asymmetric elastic: free downward, resistant upward. Dragging past the top
            // resists progressively rather than stopping dead.
            dragElastic={{ top: 0.04, bottom: 0.9 }}
            onDragEnd={(_, info) => {
              const projected = info.offset.y + project(info.velocity.y);
              // Decide on the PROJECTION, not the release point, so a flick throws it.
              if (projected > 180) onClose();
            }}
            className={cn(
              "fixed inset-x-0 bottom-0 z-50 flex max-h-[92vh] flex-col",
              "rounded-t-2xl border-t border-line-strong bg-surface-panel/95 backdrop-blur-xl",
              "shadow-lift",
              // On a wide screen it is a right-hand panel instead, entering and leaving
              // along the same edge.
              "md:inset-y-0 md:left-auto md:right-0 md:max-h-none md:w-[32rem] md:rounded-none md:rounded-l-2xl md:border-l md:border-t-0",
            )}
          >
            {/* The grabber. Present only where dragging is possible, because an affordance
                that does nothing is a lie about the interface. */}
            <div className="flex shrink-0 justify-center pb-1 pt-3 md:hidden">
              <span
                aria-hidden
                className="h-1 w-10 rounded-full bg-line-strong"
              />
            </div>

            <header className="shrink-0 px-5 pb-3 pt-2 md:pt-6">
              <h2 className="text-base font-semibold tracking-tight">{title}</h2>
              {description ? (
                <p className="mt-1 text-sm text-ink-muted">{description}</p>
              ) : null}
            </header>

            {/* The scroll edge fades instead of a hard rule, so floating chrome reads as a
                layer over the content rather than a strip cut out of it. */}
            <div className="scroll-slim min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 pb-4">
              {children}
            </div>

            {footer ? (
              <footer className="shrink-0 border-t border-line bg-surface-panel/80 px-5 py-4 pb-[max(1rem,env(safe-area-inset-bottom))] backdrop-blur-xl">
                {footer}
              </footer>
            ) : null}
          </motion.div>
        </>
      ) : null}
    </AnimatePresence>
  );
}
