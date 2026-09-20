/* cva-web — progressive enhancement only.
 *
 * Every page works with JavaScript disabled: filters are links, forms are forms, findings
 * are <details>-equivalent sections that start open when there is no script to close them.
 * This file makes triage at volume fast; it never makes it possible.
 *
 * CSP is `script-src 'self'` and there is no inline handler anywhere, so everything binds
 * through addEventListener. Nothing here is fetched from a network.
 */
(function () {
  'use strict';

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------------------------------------------------------- theme */
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
      btn.textContent = isLight ? '◓' : '◒';
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

  /* ------------------------------------------------- expandable findings */
  document.addEventListener('click', function (e) {
    var head = e.target.closest('.finding-head');
    if (!head || e.target.closest('a, button:not(.finding-head)')) { return; }
    var body = document.getElementById(head.getAttribute('aria-controls'));
    if (!body) { return; }
    var open = head.getAttribute('aria-expanded') === 'true';
    head.setAttribute('aria-expanded', open ? 'false' : 'true');
    body.hidden = open;
  });

  /* ------------------------------------------- keyboard-only triage (§10.11)
   * j / k move, Enter opens, o opens in place, / focuses the filter, ? shows help.
   * An analyst working a 10,000-finding page should not need a mouse. */
  var cursor = -1;
  function cards() { return Array.prototype.slice.call(document.querySelectorAll('.finding')); }
  function focusCard(index) {
    var list = cards();
    if (!list.length) { return; }
    cursor = Math.max(0, Math.min(index, list.length - 1));
    list.forEach(function (c) { c.classList.remove('is-target'); });
    var card = list[cursor];
    card.classList.add('is-target');
    card.scrollIntoView({ block: 'nearest', behavior: reduced ? 'auto' : 'smooth' });
    var head = card.querySelector('.finding-head');
    if (head) { head.focus(); }
  }

  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.metaKey || e.ctrlKey) {
      return;
    }
    if (e.key === 'j') { e.preventDefault(); focusCard(cursor + 1); }
    else if (e.key === 'k') { e.preventDefault(); focusCard(cursor - 1); }
    else if (e.key === 'o') {
      var list = cards();
      if (cursor >= 0 && list[cursor]) {
        var link = list[cursor].querySelector('[data-open]');
        if (link) { window.location.href = link.getAttribute('href'); }
      }
    } else if (e.key === '/') {
      var search = document.querySelector('[data-filter-search]');
      if (search) { e.preventDefault(); search.focus(); search.select(); }
    } else if (e.key === '?') {
      var help = document.querySelector('[data-shortcuts]');
      if (help) { e.preventDefault(); help.hidden = !help.hidden; }
    } else if (e.key === 'Escape') {
      var open = document.querySelector('[data-shortcuts]');
      if (open && !open.hidden) { open.hidden = true; }
    }
  });

  /* ------------------------------------------ justification live counter
   * The rule is words, not characters: a string of punctuation clears a length floor and
   * records nothing. The counter shows both, and mirrors the server's own check so an
   * analyst is never surprised by a refusal after typing three paragraphs. */
  function countWords(text) {
    var parts = text.trim().split(/\s+/).filter(function (w) { return /[^\W_]/.test(w); });
    return parts.length;
  }
  document.addEventListener('input', function (e) {
    var box = e.target.closest('[data-justification]');
    if (!box) { return; }
    var out = document.getElementById(box.getAttribute('data-counter'));
    if (!out) { return; }
    var minChars = parseInt(box.getAttribute('data-min-chars'), 10) || 0;
    var minWords = parseInt(box.getAttribute('data-min-words'), 10) || 0;
    var chars = box.value.trim().length;
    var words = countWords(box.value);
    var ok = chars >= minChars && words >= minWords;
    out.textContent = chars + '/' + minChars + ' characters · ' + words + '/' + minWords + ' words';
    out.className = 'field-count ' + (ok ? 'ok' : 'short');
    var submit = box.form && box.form.querySelector('[data-needs-justification]');
    if (submit) { submit.disabled = !ok; }
  });

  /* ------------------------------------------------ prov.* interstitial
   * D-E8: lowering a quarantine on a deterministic failure gets a sentence, not a shrug. */
  document.addEventListener('change', function (e) {
    var select = e.target.closest('[data-reason-code]');
    if (!select) { return; }
    var warn = document.getElementById(select.getAttribute('data-warn-for') || '');
    if (warn) { warn.hidden = select.value !== 'false_positive_confirmed'; }
  });

  /* ------------------------------------------------------- copy a digest */
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-copy]');
    if (!btn) { return; }
    var text = btn.getAttribute('data-copy');
    var done = function () {
      var original = btn.textContent;
      btn.textContent = 'copied';
      window.setTimeout(function () { btn.textContent = original; }, 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { /* ignore */ });
    }
  });

  /* ------------------------------------------- double-submit protection
   * The request_id in the form already makes a retry idempotent at the ledger. This just
   * stops the second click looking like it did nothing. */
  document.addEventListener('submit', function (e) {
    var form = e.target;
    if (form.hasAttribute('data-no-guard')) { return; }
    var submit = form.querySelector('button[type="submit"], input[type="submit"]');
    if (submit && !submit.disabled) {
      window.setTimeout(function () {
        submit.disabled = true;
        submit.setAttribute('aria-busy', 'true');
      }, 0);
    }
  });

  /* -------------------------------------------------------------- toasts */
  Array.prototype.forEach.call(document.querySelectorAll('.toast'), function (toast) {
    if (toast.hasAttribute('data-sticky')) { return; }
    window.setTimeout(function () {
      toast.style.transition = 'opacity .3s, transform .3s';
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(6px)';
      window.setTimeout(function () { toast.remove(); }, 320);
    }, 7000);
  });

  /* --------------------------------------------- geometry without inline CSS
   * CSP is `style-src 'self'`, so no template may carry a `style` attribute. Anything whose
   * size depends on data — a confidence interval, a disposition bar, a reliability diagram —
   * is drawn here through the CSSOM instead. Every one of these also prints its numbers as
   * text, so a reader with JavaScript off loses the picture and none of the facts. */
  function clamp(v) { return Math.max(0, Math.min(100, v)); }

  Array.prototype.forEach.call(document.querySelectorAll('[data-ci]'), function (el) {
    var lo = clamp(parseFloat(el.getAttribute('data-lo')));
    var hi = clamp(parseFloat(el.getAttribute('data-hi')));
    var mean = clamp(parseFloat(el.getAttribute('data-mean')));
    var span = el.querySelector('.ci-span');
    var mark = el.querySelector('.ci-mean');
    var base = el.querySelector('.ci-base');
    if (span) { span.style.left = lo + '%'; span.style.width = Math.max(1, hi - lo) + '%'; }
    if (mark) { mark.style.left = mean + '%'; }
    if (base) { base.style.left = clamp(parseFloat(el.getAttribute('data-base'))) + '%'; }
  });

  Array.prototype.forEach.call(document.querySelectorAll('[data-bar]'), function (el) {
    Array.prototype.forEach.call(el.querySelectorAll('[data-share]'), function (part) {
      part.style.width = clamp(parseFloat(part.getAttribute('data-share'))) + '%';
    });
  });

  Array.prototype.forEach.call(document.querySelectorAll('[data-reliability] i'), function (bar) {
    bar.style.height = Math.max(2, clamp(parseFloat(bar.getAttribute('data-h')))) + '%';
  });

  /* ------------------------------------------------------ count-up
   * Only on numbers that are already correct in the HTML, so a reader without JS sees the
   * real value and a reader with it sees the same value arrive. */
  if (!reduced) {
    Array.prototype.forEach.call(document.querySelectorAll('[data-countup]'), function (el) {
      var target = parseFloat(el.textContent);
      if (!isFinite(target) || target <= 0 || target > 100000) { return; }
      var start = null;
      var duration = 520;
      function step(ts) {
        if (start === null) { start = ts; }
        var p = Math.min(1, (ts - start) / duration);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = String(Math.round(target * eased));
        if (p < 1) { window.requestAnimationFrame(step); }
        else { el.textContent = String(target); }
      }
      el.textContent = '0';
      window.requestAnimationFrame(step);
    });
  }
})();
