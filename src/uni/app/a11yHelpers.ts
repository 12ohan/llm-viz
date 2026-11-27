/**
 * Localized WAI-ARIA Accessibility Helpers for NLD-Uni (`src/uni/app/a11yHelpers.ts`)
 *
 * Thin compatibility wrapper over `src/shared/a11y.ts` preserving NLD-Uni's exact
 * public helper names while enforcing Locked Decision D4:
 *   - Strictly ZERO imports from `src/ui/controls.ts`
 *   - Strictly ZERO imports from `src/twotower/` or `src/diffusion/`
 */

export {
  createFocusTrap as createUniFocusTrap,
  handleIndexedTablistKeydown as handleUniTablistKeydown,
  isEditableShortcutTarget as shouldDeferGlobalShortcut,
  isReducedMotionPreferred as isUniReducedMotionPreferred,
  syncRovingTabindex,
  type EditableShortcutTargetLike as UniShortcutTargetLike,
  type FocusableTabLike as UniFocusableTabLike,
  type KeydownEventLike as UniKeydownEventLike,
  type RovingTabLike as UniRovingTabLike,
} from '../../shared/a11y';
