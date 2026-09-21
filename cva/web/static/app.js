/* cva-web — the server-rendered pages' only script: the theme.
 *
 * The dashboard is the Next.js app at /app/; this serves sign-in, error pages and the
 * evidence route's "not served" page. It reads the same `cva-theme` key the dashboard
 * writes, so a person who chose light in the dashboard signs in to a light page.
 *
 * CSP is `script-src 'self'` and there is no inline handler anywhere, so everything binds
 * through addEventListener. Nothing here is fetched from a network.
 */
(function () {
  'use strict';

  function readTheme() {
    try { return localStorage.getItem('cva-theme'); } catch (e) { return null; }
  }
  function writeTheme(value) {
    try { localStorage.setItem('cva-theme', value); } catch (e) { /* private window */ }
  }
  function applyTheme(value) {
    if (value === 'light' || value === 'dark') {
      document.documentElement.setAttribute('data-theme', value);
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    var btn = document.querySelector('[data-theme-toggle]');
    if (btn) {
      var isLight = document.documentElement.getAttribute('data-theme') === 'light';
      btn.setAttribute('aria-label', isLight ? 'Switch to dark theme' : 'Switch to light theme');
    }
  }
  applyTheme(readTheme());

  document.addEventListener('click', function (e) {
    var toggle = e.target.closest('[data-theme-toggle]');
    if (!toggle) { return; }
    var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    writeTheme(next);
    applyTheme(next);
  });
})();
