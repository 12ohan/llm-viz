/**
 * Shared Pure WAI-ARIA Accessibility & Keyboard Helpers (`src/shared/a11y.ts`)
 *
 * Strictly zero dependencies on any model, scene, render, or UI module.
 * Node-testable roving `tabindex`, horizontal tablist keyboard navigation,
 * global shortcut deferral (interactive activation vs. editable inputs),
 * modal focus trapping, focus restoration, and reduced-motion detection.
 */

export interface RovingTabLike {
  tabIndex: number;
  setAttribute(k: string, v: string): void;
}

export interface FocusableTabLike extends RovingTabLike {
  focus(): void;
}

export interface KeydownEventLike {
  key: string;
  preventDefault(): void;
}

export interface EditableShortcutTargetLike {
  tagName?: string;
  isContentEditable?: boolean;
  getAttribute?: (k: string) => string | null;
}

/**
 * Interactive controls that must retain native Space/Enter activation instead of being
 * intercepted by global canvas/tour shortcuts.
 */
export const INTERACTIVE_ACTIVATION_SELECTOR =
  'button, a[href], [role="button"], [role="tab"], summary, [tabindex]:not([tabindex="-1"]):not(#gl-canvas)';

export const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Returns true when a global `Space`/`Enter` shortcut should defer without calling
 * `preventDefault()` so native keyboard activation operates on focused interactive controls.
 * `Escape` and `?` remain global (return false).
 */
export function shouldDeferActivationKey(
  key: string,
  target: EventTarget | null,
): boolean {
  if (key !== ' ' && key !== 'Enter') return false;
  if (!target || typeof target !== 'object') return false;
  const el = target as Partial<HTMLElement>;
  if (typeof el.closest === 'function') {
    return el.closest(INTERACTIVE_ACTIVATION_SELECTOR) !== null;
  }
  let cur: Partial<HTMLElement> | null = el;
  while (cur) {
    const tag = (cur.tagName ?? '').toUpperCase();
    const role = typeof cur.getAttribute === 'function' ? cur.getAttribute('role') : null;
    const href = typeof cur.getAttribute === 'function' ? cur.getAttribute('href') : null;
    const tabindex = typeof cur.getAttribute === 'function' ? cur.getAttribute('tabindex') : null;
    const id = cur.id ?? '';
    if (
      tag === 'BUTTON' ||
      tag === 'SUMMARY' ||
      (tag === 'A' && href !== null) ||
      role === 'button' ||
      role === 'tab' ||
      (tabindex !== null && tabindex !== '-1' && id !== 'gl-canvas')
    ) {
      return true;
    }
    cur = (cur.parentNode as Partial<HTMLElement> | null) ?? null;
  }
  return false;
}

/**
 * Returns `true` when global single-key shortcuts must defer to editable form inputs
 * (`INPUT`, `TEXTAREA`, `SELECT`), `contenteditable` regions, or `role="textbox"` /
 * `role="combobox"` widgets.
 */
export function isEditableShortcutTarget(
  target: EditableShortcutTargetLike | null,
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

export function getFocusableElements(container: HTMLElement): HTMLElement[] {
  if (typeof container.querySelectorAll !== 'function') return [];
  const nodes = Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
  return nodes.filter((el) => {
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

/**
 * Trap `Tab` / `Shift+Tab` inside an open modal dialog (`aria-modal="true"`),
 * cycling cleanly between the first and last visible focusable descendants.
 */
export function handleModalTabTrap(
  container: HTMLElement,
  ev: Pick<KeyboardEvent, 'key' | 'shiftKey' | 'preventDefault'>,
  activeElement: Element | null = typeof document !== 'undefined' ? document.activeElement : null,
): boolean {
  if (ev.key !== 'Tab') return false;
  const focusables = getFocusableElements(container);
  if (focusables.length === 0) {
    ev.preventDefault();
    if (typeof container.focus === 'function') container.focus();
    return true;
  }
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  const activeIdx = activeElement ? focusables.indexOf(activeElement as HTMLElement) : -1;

  if (ev.shiftKey) {
    if (activeIdx <= 0) {
      ev.preventDefault();
      if (typeof last.focus === 'function') last.focus();
      return true;
    }
  } else {
    if (activeIdx === -1 || activeIdx >= focusables.length - 1) {
      ev.preventDefault();
      if (typeof first.focus === 'function') first.focus();
      return true;
    }
  }
  return false;
}

/** Restore focus to a previously focused element when closing a modal (if still connected). */
export function restoreFocusIfConnected(el: HTMLElement | null): boolean {
  if (!el) return false;
  const connected = typeof el.isConnected === 'boolean' ? el.isConnected : true;
  if (!connected || el.hidden) return false;
  if ('disabled' in el && (el as HTMLButtonElement).disabled) return false;
  if (typeof el.focus === 'function') {
    el.focus();
    return true;
  }
  return false;
}

/**
 * Synchronize WAI-ARIA roving `tabindex` (`0` on active tab, `-1` on others) and
 * `aria-selected` (`"true"` vs `"false"`) across a tablist strip.
 */
export function syncRovingTabindex(
  tabs: readonly RovingTabLike[],
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
 * Full WAI-ARIA Tabs keyboard navigation (`ArrowLeft` / `ArrowRight` / `Home` / `End`)
 * with roving `tabindex` and automatic focus + selection movement across visible enabled tabs.
 */
export function handleTablistKeydown(
  tabs: readonly HTMLButtonElement[],
  ev: Pick<KeyboardEvent, 'key' | 'target' | 'preventDefault' | 'stopPropagation'>,
  onSelect: (tab: HTMLButtonElement, tabIndexInAll: number) => void,
): boolean {
  if (
    ev.key !== 'ArrowLeft' &&
    ev.key !== 'ArrowRight' &&
    ev.key !== 'Home' &&
    ev.key !== 'End'
  ) {
    return false;
  }
  const enabledEntries: { tab: HTMLButtonElement; idx: number }[] = [];
  for (let i = 0; i < tabs.length; i++) {
    const t = tabs[i];
    if (!t.hidden && !t.disabled) {
      enabledEntries.push({ tab: t, idx: i });
    }
  }
  if (enabledEntries.length === 0) return false;

  let curPos = enabledEntries.findIndex(
    (e) =>
      e.tab === ev.target ||
      (typeof e.tab.contains === 'function' && ev.target != null && e.tab.contains(ev.target as Node)),
  );
  if (curPos < 0) {
    curPos = enabledEntries.findIndex((e) => e.tab.getAttribute('aria-selected') === 'true');
  }
  if (curPos < 0) curPos = 0;

  let nextPos = curPos;
  if (ev.key === 'ArrowRight') {
    nextPos = (curPos + 1) % enabledEntries.length;
  } else if (ev.key === 'ArrowLeft') {
    nextPos = (curPos - 1 + enabledEntries.length) % enabledEntries.length;
  } else if (ev.key === 'Home') {
    nextPos = 0;
  } else if (ev.key === 'End') {
    nextPos = enabledEntries.length - 1;
  }

  const target = enabledEntries[nextPos];
  ev.preventDefault();
  ev.stopPropagation();

  for (let i = 0; i < tabs.length; i++) {
    tabs[i].setAttribute('tabindex', i === target.idx ? '0' : '-1');
  }
  if (typeof target.tab.focus === 'function') {
    target.tab.focus();
  }
  onSelect(target.tab, target.idx);
  return true;
}

/**
 * Index-driven horizontal tablist keydown handler (`ArrowRight`, `ArrowLeft`, `Home`, `End`)
 * for lightweight headless/mock tablists.
 */
export function handleIndexedTablistKeydown(
  event: KeydownEventLike,
  tabs: readonly FocusableTabLike[],
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
 * Create a WAI-ARIA modal focus trap cycling `Tab` / `Shift+Tab` inside `container`,
 * calling `onEscape` on `Escape`, and restoring previous focus on `deactivate()`.
 */
export function createFocusTrap(
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
    const focusables = getFocusableElements(container);
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
    restoreFocusIfConnected(toRestore);
  }

  function handleKeydown(event: KeyboardEvent): boolean {
    if (!active) return false;

    if (event.key === 'Escape') {
      if (typeof event.preventDefault === 'function') event.preventDefault();
      if (typeof onEscape === 'function') onEscape();
      return true;
    }

    return handleModalTabTrap(container, event);
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
export function isReducedMotionPreferred(
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
