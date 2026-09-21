/* cva-web — the server-rendered pages' only script: the theme.
 *
 * The dashboard is the Next.js app at /app/; this serves sign-in, error pages and the
 * evidence route's "not served" page. It reads the same `cva-theme` key the dashboard
 * writes, so a person who chose light in the dashboard signs in to a light page.
 *
 * These pages have no theme toggle of their own; they only follow the dashboard's choice.
 * CSP is `script-src 'self'`; nothing here is fetched from a network.
 */
(function () {
  'use strict';

  var theme = null;
  try { theme = localStorage.getItem('cva-theme'); } catch (e) { /* private window */ }
  if (theme === 'light' || theme === 'dark') {
    document.documentElement.setAttribute('data-theme', theme);
  }
})();
