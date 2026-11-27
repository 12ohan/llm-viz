/**
 * Localized WAI-ARIA Accessibility Helpers for NLD-Uni (`src/uni/app/a11yHelpers.ts`)
 *
 * Enforces Locked Decision D4:
 *   - Strictly ZERO imports from `src/ui/controls.ts` so Rollup cannot perturb
 *     Nano's `<= 215.00 KiB` chunk graph.
 *   - Pure Node-testable roving-tabindex, tablist keyboard navigation, global shortcut
 *     deferral, focus trapping, and reduced-motion detection.
 *
 * Zero `Math.random()`, zero `src/twotower/`, `src/diffusion/`, or `src/ui/*` imports.
 */

export interface UniRovingTabLike {
  tabIndex: number;
  setAttribute(k: string, v: string): void;
}

export interface UniFocusableTabLike extends UniRovingTabLike {
  focus(): void;
}

export interface UniKeydownEventLike {
  key: string;
  preventDefault(): void;
}

export interface UniShortcutTargetLike {
  tagName?: string;
  isContentEditable?: boolean;
  getAttribute?: (k: string) => string | null;
}

const UNI_FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Synchronize WAI-ARIA roving `tabindex` (`0` on active tab, `-1` on others) and
 * `aria-selected` (`"true"` vs `"false"`) across a tablist strip.
 */
export function syncRovingTabindex(
  tabs: readonly UniRovingTabLike[],
  activeIndex: number,
): void {
  if (tabs.length === 0) return;
  const clampedActive = Math.max(0, Math.min(tabs.length - 1, Math.trunc(activeIndex)));
  for (let i = 0; i < tabs.length; i++) {
    const tab = tabs[i]!;
    const isActive = i === clampedActive;
    tab.tabIndex = isActive ? 0 : -1;
    tab.setAttribute('tabindex', isActive ? '0' : '-1');
    tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
  }
}

/**
 * Handle WAI-ARIA horizontal tablist keyboard navigation (`ArrowRight`, `ArrowLeft`,
 * `Home`, `End`) with wrap-around, `preventDefault()`, roving `tabindex` + `aria-selected`
 * update, focus transfer, and `onSelect(nextIndex)` callback.
 */
export function handleUniTablistKeydown(
  event: UniKeydownEventLike,
  tabs: readonly UniFocusableTabLike[],
  activeIndex: number,
  onSelect: (nextIndex: number) => void,
): boolean {
  if (tabs.length === 0) return false;
  const cur = Math.max(0, Math.min(tabs.length - 1, Math.trunc(activeIndex)));

  let nextIndex: number;
  switch (event.key) {
    case 'ArrowRight':
      nextIndex = (cur + 1) % tabs.length;
      break;
    case 'ArrowLeft':
      nextIndex = (cur - 1 + tabs.length) % tabs.length;
      break;
    case 'Home':
      nextIndex = 0;
      break;
    case 'End':
      nextIndex = tabs.length - 1;
      break;
    default:
      return false;
  }

  event.preventDefault();
  syncRovingTabindex(tabs, nextIndex);
  const target = tabs[nextIndex]!;
  if (typeof target.focus === 'function') {
    target.focus();
  }
  onSelect(nextIndex);
  return true;
}

/**
 * Returns `true` when global single-key canvas/tour shortcuts must defer to editable
 * form inputs (`INPUT`, `TEXTAREA`, `SELECT`), `contenteditable` regions, or
 * `role="textbox"` / `role="combobox"` widgets.
 */
export function shouldDeferGlobalShortcut(
  target: UniShortcutTargetLike | null,
): boolean {
  if (!target || typeof target !== 'object') return false;
  const tag = typeof target.tagName === 'string' ? target.tagName.toUpperCase() : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    return true;
  }
  if (target.isContentEditable === true) {
    return true;
  }
  if (typeof target.getAttribute === 'function') {
    const contentEditable = target.getAttribute('contenteditable');
    if (contentEditable === 'true' || contentEditable === '') {
      return true;
    }
    const role = target.getAttribute('role');
    if (role === 'textbox' || role === 'combobox') {
      return true;
    }
  }
  return false;
}

function getVisibleFocusableChildren(container: HTMLElement): HTMLElement[] {
  if (typeof container.querySelectorAll !== 'function') return [];
  const raw = Array.from(container.querySelectorAll<HTMLElement>(UNI_FOCUSABLE_SELECTOR));
  return raw.filter((el) => {
    if (el.hidden) return false;
    if ('disabled' in el && (el as HTMLButtonElement).disabled) return false;
    let cur: HTMLElement | null =
      (el.parentElement ?? (el.parentNode as HTMLElement | null)) ?? null;
    while (cur && cur !== container) {
      if (cur.hidden) return false;
      cur = (cur.parentElement ?? (cur.parentNode as HTMLElement | null)) ?? null;
    }
    return true;
  });
}

function restoreElementFocus(el: HTMLElement | null): void {
  if (!el) return;
  const connected = typeof el.isConnected === 'boolean' ? el.isConnected : true;
  if (!connected || el.hidden) return;
  if ('disabled' in el && (el as HTMLButtonElement).disabled) return;
  if (typeof el.focus === 'function') {
    el.focus();
  }
}

/**
 * Create a localized WAI-ARIA modal focus trap cycling `Tab` / `Shift+Tab` inside
 * `container`, calling `onEscape` on `Escape`, and restoring previous focus on `deactivate()`.
 */
export function createUniFocusTrap(
  container: HTMLElement,
  onEscape?: () => void,
): {
  activate(): void;
  deactivate(): void;
  handleKeydown(event: KeyboardEvent): boolean;
} {
  let active = false;
  let previousFocus: HTMLElement | null = null;

  function activate(): void {
    previousFocus =
      typeof document !== 'undefined' ? (document.activeElement as HTMLElement | null) : null;
    active = true;
    const focusables = getVisibleFocusableChildren(container);
    if (focusables.length > 0) {
      const first = focusables[0]!;
      if (typeof first.focus === 'function') first.focus();
    } else if (typeof container.focus === 'function') {
      container.focus();
    }
  }

  function deactivate(): void {
    if (!active) return;
    active = false;
    const toRestore = previousFocus;
    previousFocus = null;
    restoreElementFocus(toRestore);
  }

  function handleKeydown(event: KeyboardEvent): boolean {
    if (!active) return false;

    if (event.key === 'Escape') {
      if (typeof event.preventDefault === 'function') event.preventDefault();
      if (typeof onEscape === 'function') onEscape();
      return true;
    }

    if (event.key !== 'Tab') {
      return false;
    }

    const focusables = getVisibleFocusableChildren(container);
    if (focusables.length === 0) {
      if (typeof event.preventDefault === 'function') event.preventDefault();
      if (typeof container.focus === 'function') container.focus();
      return true;
    }

    const activeEl =
      typeof document !== 'undefined' ? (document.activeElement as HTMLElement | null) : null;
    const activeIdx = activeEl ? focusables.indexOf(activeEl) : -1;
    const first = focusables[0]!;
    const last = focusables[focusables.length - 1]!;

    if (event.shiftKey) {
      if (activeIdx <= 0) {
        if (typeof event.preventDefault === 'function') event.preventDefault();
        if (typeof last.focus === 'function') last.focus();
        return true;
      }
      return false;
    }

    if (activeIdx === -1 || activeIdx >= focusables.length - 1) {
      if (typeof event.preventDefault === 'function') event.preventDefault();
      if (typeof first.focus === 'function') first.focus();
      return true;
    }
    return false;
  }

  return {
    activate,
    deactivate,
    handleKeydown,
  };
}

/**
 * Headless-safe `prefers-reduced-motion: reduce` detector supporting injectable `Window.matchMedia`.
 */
export function isUniReducedMotionPreferred(
  win?: Pick<Window, 'matchMedia'> | null,
): boolean {
  try {
    const targetWin =
      win !== undefined ? win : typeof window !== 'undefined' ? window : null;
    if (!targetWin || typeof targetWin.matchMedia !== 'function') {
      return false;
    }
    return Boolean(targetWin.matchMedia('(prefers-reduced-motion: reduce)')?.matches);
  } catch {
    return false;
  }
}
