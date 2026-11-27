/**
 * Compact Accessible Multi-Model Top-Bar Switcher (`src/shared/modelSwitcher.ts`)
 *
 * Renders a fixed-width `Models (N) ▾` menu button + native `<a href>` popover
 * driven solely by `MODEL_REGISTRY` (`src/shared/modelRegistry.ts`). Scales
 * cleanly to 6+ models with zero top-bar overflow, exact viewport-clamped popover
 * alignment, native link semantics, full keyboard navigation, and zero cross-app imports.
 */

import './modelSwitcher.css';
import { restoreFocusIfConnected } from './a11y';
import {
  MODEL_REGISTRY,
  resolveModelHref,
  type ModelEntryId,
  type ModelEntryMeta,
} from './modelRegistry';

export interface ModelSwitcherOptions {
  readonly activeId: ModelEntryId;
  readonly baseUrl?: string;
  readonly theme?: 'dark' | 'light';
  readonly entries?: readonly ModelEntryMeta[];
}

export interface ModelSwitcherHandle {
  readonly root: HTMLElement;
  readonly trigger: HTMLButtonElement;
  readonly menu: HTMLDivElement;
  readonly links: readonly HTMLAnchorElement[];
  isOpen(): boolean;
  open(focusTarget?: 'first-other' | 'first' | 'last' | 'none'): void;
  close(restoreFocus?: boolean): void;
  toggle(): void;
  destroy(): void;
}

let switcherIdSeq = 0;

function makeEl<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function haltEvent(ev: Event): void {
  ev.preventDefault();
  ev.stopPropagation();
}

export function mountModelSwitcher(
  host: HTMLElement,
  options: ModelSwitcherOptions,
): ModelSwitcherHandle {
  const entries = options.entries ?? MODEL_REGISTRY;
  const theme = options.theme ?? 'dark';
  const menuId = `nv-model-switcher-menu-${++switcherIdSeq}`;

  const root = makeEl('nav', `nv-model-switcher nv-model-switcher--${theme}`);
  root.setAttribute('aria-label', 'Nemotron architecture visualizers');
  root.dataset.testid = 'model-switcher';

  const trigger = makeEl('button', 'nv-switcher-btn');
  trigger.type = 'button';
  trigger.dataset.testid = 'model-switcher-trigger';
  trigger.setAttribute('aria-haspopup', 'true');
  trigger.setAttribute('aria-expanded', 'false');
  trigger.setAttribute('aria-controls', menuId);
  trigger.title = `Switch between ${entries.length} Nemotron architecture visualizers`;

  const caret = makeEl('span', 'nv-switcher-caret', '▾');
  caret.setAttribute('aria-hidden', 'true');
  trigger.append(
    makeEl('span', undefined, 'Models'),
    makeEl('span', 'nv-switcher-count', String(entries.length)),
    caret,
  );

  const menu = makeEl('div', 'nv-switcher-menu');
  menu.id = menuId;
  menu.dataset.testid = 'model-switcher-menu';
  menu.hidden = true;
  menu.appendChild(makeEl('div', 'nv-switcher-header', `Nemotron Visualizers (${entries.length})`));

  const links: HTMLAnchorElement[] = [];
  for (const entry of entries) {
    const isCurrent = entry.id === options.activeId;
    const link = makeEl('a', isCurrent ? 'nv-switcher-item is-current' : 'nv-switcher-item');
    link.href = resolveModelHref(options.baseUrl, entry);
    link.dataset.testid = entry.testId;
    link.dataset.modelId = entry.id;
    link.title = `Open ${entry.title} architecture visualizer`;
    if (isCurrent) link.setAttribute('aria-current', 'page');

    const head = makeEl('span', 'nv-switcher-item-head');
    head.appendChild(makeEl('span', 'nv-switcher-item-title', entry.shortLabel));
    if (isCurrent) {
      head.appendChild(makeEl('span', 'nv-switcher-item-badge', 'Current'));
    } else {
      const arrow = makeEl('span', 'nv-switcher-item-arrow', '↗');
      arrow.setAttribute('aria-hidden', 'true');
      head.appendChild(arrow);
    }
    link.append(head, makeEl('span', 'nv-switcher-item-sub', entry.subtitle));
    menu.appendChild(link);
    links.push(link);
  }

  root.append(trigger, menu);
  host.appendChild(root);

  let openState = false;

  function alignMenuToViewport(): void {
    if (typeof window === 'undefined' || typeof trigger.getBoundingClientRect !== 'function' || !menu.style) {
      return;
    }
    const vw = window.innerWidth || 0;
    if (vw <= 0) return;
    const rect = trigger.getBoundingClientRect();
    const pad = 8;
    const menuWidth = Math.min(280, Math.max(180, vw - pad * 2));
    const maxLeft = Math.max(pad, vw - menuWidth - pad);
    const desiredLeft = Math.min(maxLeft, Math.max(pad, rect.right - menuWidth));
    menu.style.left = `${Math.round(desiredLeft - rect.left)}px`;
    menu.style.right = 'auto';
  }

  function focusLink(index: number): void {
    if (links.length === 0) return;
    const target = links[((index % links.length) + links.length) % links.length];
    if (target && typeof target.focus === 'function') target.focus();
  }

  function open(focusTarget: 'first-other' | 'first' | 'last' | 'none' = 'none'): void {
    if (!openState) {
      openState = true;
      alignMenuToViewport();
      menu.hidden = false;
      trigger.setAttribute('aria-expanded', 'true');
    }
    if (focusTarget === 'first-other') {
      const idx = links.findIndex((a) => a.dataset.modelId !== options.activeId);
      focusLink(idx >= 0 ? idx : 0);
    } else if (focusTarget === 'first') {
      focusLink(0);
    } else if (focusTarget === 'last') {
      focusLink(links.length - 1);
    }
  }

  function close(restoreFocus = false): void {
    if (!openState) return;
    openState = false;
    menu.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
    if (restoreFocus) restoreFocusIfConnected(trigger);
  }

  const toggle = (): void => (openState ? close(false) : open('none'));

  const onTriggerClick = (ev: MouseEvent): void => {
    ev.stopPropagation();
    toggle();
  };

  const onTriggerKeydown = (ev: KeyboardEvent): void => {
    if (ev.key === 'ArrowDown') {
      haltEvent(ev);
      open('first-other');
    } else if (ev.key === 'ArrowUp') {
      haltEvent(ev);
      open('last');
    } else if (ev.key === 'Escape' && openState) {
      haltEvent(ev);
      close(true);
    }
  };

  const onMenuKeydown = (ev: KeyboardEvent): void => {
    if (!openState) return;
    const curIdx = links.findIndex((a) => a === ev.target || a === document.activeElement);
    if (ev.key === 'Escape') {
      haltEvent(ev);
      close(true);
    } else if (ev.key === 'ArrowDown') {
      haltEvent(ev);
      focusLink(curIdx >= 0 ? curIdx + 1 : 0);
    } else if (ev.key === 'ArrowUp') {
      haltEvent(ev);
      focusLink(curIdx >= 0 ? curIdx - 1 : links.length - 1);
    } else if (ev.key === 'Home') {
      haltEvent(ev);
      focusLink(0);
    } else if (ev.key === 'End') {
      haltEvent(ev);
      focusLink(links.length - 1);
    } else if (ev.key === 'Tab') {
      close(false);
    }
  };

  const onDocumentPointerDown = (ev: Event): void => {
    if (!openState) return;
    const target = ev.target as Node | null;
    if (target && typeof root.contains === 'function' && root.contains(target)) return;
    close(false);
  };

  const onDocumentKeydown = (ev: KeyboardEvent): void => {
    if (!openState || ev.key !== 'Escape') return;
    haltEvent(ev);
    close(true);
  };

  const onWindowResize = (): void => {
    if (openState) alignMenuToViewport();
  };

  trigger.addEventListener('click', onTriggerClick);
  trigger.addEventListener('keydown', onTriggerKeydown);
  menu.addEventListener('keydown', onMenuKeydown);
  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('pointerdown', onDocumentPointerDown);
    document.addEventListener('keydown', onDocumentKeydown, true);
  }
  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
    window.addEventListener('resize', onWindowResize);
  }

  function destroy(): void {
    trigger.removeEventListener('click', onTriggerClick);
    trigger.removeEventListener('keydown', onTriggerKeydown);
    menu.removeEventListener('keydown', onMenuKeydown);
    if (typeof document !== 'undefined' && typeof document.removeEventListener === 'function') {
      document.removeEventListener('pointerdown', onDocumentPointerDown);
      document.removeEventListener('keydown', onDocumentKeydown, true);
    }
    if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
      window.removeEventListener('resize', onWindowResize);
    }
    if (root.parentNode) root.parentNode.removeChild(root);
  }

  return {
    root,
    trigger,
    menu,
    links,
    isOpen: () => openState,
    open,
    close,
    toggle,
    destroy,
  };
}
