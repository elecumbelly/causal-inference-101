(() => {
  const desktop = window.matchMedia('(min-width: 1280px)');
  const stateKey = 'ci:left-rail-hidden';

  function readStoredState() {
    try {
      return window.localStorage.getItem(stateKey) === 'true';
    } catch {
      return false;
    }
  }

  function storeState(hidden) {
    try {
      window.localStorage.setItem(stateKey, String(hidden));
    } catch {
      // The control still works when storage is blocked or unavailable.
    }
  }

  function applyDesktopState(button, sidebar, hidden, persist = true) {
    sidebar.id ||= 'ci-primary-navigation';
    button.dataset.ciLeftHidden = String(hidden);
    button.setAttribute('aria-controls', sidebar.id);
    button.setAttribute('aria-expanded', String(!hidden));
    button.setAttribute('aria-label', hidden ? 'Show navigation' : 'Hide navigation');
    button.title = hidden ? 'Show navigation' : 'Hide navigation';

    const label = button.querySelector('.sr-only');
    if (label) label.textContent = hidden ? 'Show navigation' : 'Hide navigation';
    if (persist) storeState(hidden);
  }

  function restoreMobileState(button) {
    delete button.dataset.ciLeftHidden;
    button.removeAttribute('aria-controls');
    button.removeAttribute('aria-expanded');
    button.removeAttribute('aria-label');
    button.removeAttribute('title');

    const label = button.querySelector('.sr-only');
    if (label) label.textContent = 'Open Menu';
  }

  function installNavigationToggle() {
    const button = document.querySelector('.myst-top-nav-menu-button');
    const sidebar = document.querySelector('.myst-primary-sidebar');
    if (!button || !sidebar || button.dataset.ciLayoutControl === 'true') return;

    button.dataset.ciLayoutControl = 'true';
    if (desktop.matches) applyDesktopState(button, sidebar, readStoredState(), false);
  }

  function handleNavigationClick(event) {
    if (!desktop.matches || !(event.target instanceof Element)) return;

    const button = event.target.closest('.myst-top-nav-menu-button');
    const sidebar = document.querySelector('.myst-primary-sidebar');
    if (!button || !sidebar) return;

    // Capture on window before React's document listener. At desktop widths
    // this button controls the persistent rail, not MyST's mobile drawer.
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    applyDesktopState(button, sidebar, button.dataset.ciLeftHidden !== 'true');
  }

  function syncBreakpoint() {
    const button = document.querySelector('.myst-top-nav-menu-button');
    const sidebar = document.querySelector('.myst-primary-sidebar');
    if (!button || !sidebar) return;

    if (button.dataset.ciLayoutControl !== 'true') installNavigationToggle();
    if (desktop.matches) {
      applyDesktopState(button, sidebar, readStoredState(), false);
    } else {
      restoreMobileState(button);
    }
  }

  function start() {
    installNavigationToggle();
    window.addEventListener('click', handleNavigationClick, true);

    // MyST can replace the header during client-side navigation. Reinstall the
    // control on that fresh button instead of binding only to the discarded one.
    const hydrationObserver = new MutationObserver(installNavigationToggle);
    hydrationObserver.observe(document.body, { childList: true, subtree: true });
    desktop.addEventListener('change', syncBreakpoint);
    window.addEventListener('pagehide', () => hydrationObserver.disconnect(), { once: true });
  }

  const startAfterHydration = () => window.setTimeout(start, 250);
  if (document.readyState === 'complete') {
    startAfterHydration();
  } else {
    window.addEventListener('load', startAfterHydration, { once: true });
  }
})();
