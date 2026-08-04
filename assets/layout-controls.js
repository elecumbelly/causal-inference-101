(() => {
  const desktop = window.matchMedia('(min-width: 1280px)');
  const navigationStateKey = 'ci:left-rail-hidden';
  const outlineStateKey = 'ci:right-rail-closed';

  function readStoredState(key, fallback) {
    try {
      const stored = window.localStorage.getItem(key);
      return stored === null ? fallback : stored === 'true';
    } catch {
      return fallback;
    }
  }

  function storeState(key, value) {
    try {
      window.localStorage.setItem(key, String(value));
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
    if (persist) storeState(navigationStateKey, hidden);
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
    if (desktop.matches) applyDesktopState(button, sidebar, readStoredState(navigationStateKey, true), false);
  }

  function installOutlineToggle() {
    const section = document.querySelector('.myst-outline-section');
    const button = section?.querySelector('.myst-outline-collapsible');
    if (!section || !button || section.dataset.ciOutlineControl === 'true') return;

    // Keep the marker on the stable section: MyST replaces the button when its
    // open state changes, which would otherwise re-run the default-state logic.
    section.dataset.ciOutlineControl = 'true';
    const shouldClose = readStoredState(outlineStateKey, true);
    const isClosed = section.dataset.state === 'closed';
    if (shouldClose !== isClosed) window.setTimeout(() => button.click(), 0);
  }

  function scheduleOutlinePersistence(outlineButton) {
    // MyST owns the Contents animation. Record its resulting state after React
    // handles the click so the preference survives navigation and reloads.
    window.setTimeout(() => {
      const section = outlineButton.closest('.myst-outline-section');
      if (section) storeState(outlineStateKey, section.dataset.state === 'closed');
    }, 0);
  }

  function installLayoutControls() {
    installNavigationToggle();
    installOutlineToggle();
  }

  function handleLayoutClick(event) {
    if (!(event.target instanceof Element)) return;

    const outlineButton = event.target.closest('.myst-outline-collapsible');
    if (outlineButton) {
      scheduleOutlinePersistence(outlineButton);
      return;
    }

    if (!desktop.matches) return;

    const button = event.target.closest('.myst-top-nav-menu-button');
    const sidebar = document.querySelector('.myst-primary-sidebar');
    if (button && sidebar) {
      // Capture on window before React's document listener. At desktop widths
      // this button controls the persistent rail, not MyST's mobile drawer.
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      applyDesktopState(button, sidebar, button.dataset.ciLeftHidden !== 'true');
    }
  }

  function syncBreakpoint() {
    const button = document.querySelector('.myst-top-nav-menu-button');
    const sidebar = document.querySelector('.myst-primary-sidebar');
    if (!button || !sidebar) return;

    if (button.dataset.ciLayoutControl !== 'true') installNavigationToggle();
    if (desktop.matches) {
      applyDesktopState(button, sidebar, readStoredState(navigationStateKey, true), false);
      installOutlineToggle();
    } else {
      restoreMobileState(button);
    }
  }

  function start() {
    installLayoutControls();
    window.addEventListener('click', handleLayoutClick, true);

    // MyST can replace the header during client-side navigation. Reinstall the
    // control on that fresh button instead of binding only to the discarded one.
    const hydrationObserver = new MutationObserver(installLayoutControls);
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
