---
name: browser-harness
description: "Always use browser-harness for any web interaction: automation, scraping, testing, or site/app work."
---

# browser-harness

Direct browser control via CDP. For task-specific edits, use `agent-workspace/agent_helpers.py`. For setup, install, or connection problems, read https://github.com/browser-use/browser-harness/blob/main/install.md.

## When Not to Use

A basic fetch of public information needs no browser. If a plain HTTP request can read it — a public page, an API, docs — use `curl` or your fetch tool, and leave the browser alone. Use browser-harness when the task needs interaction (click, type, navigate), the user's logged-in session, JS rendering, or a bot-protected page. If a direct fetch fails or returns a shell page, then escalate to the browser.

Domain skills are off by default. Set `BH_DOMAIN_SKILLS=1` to enable them; see the bottom section.

**If `BH_DOMAIN_SKILLS=1` and the task is site-specific, read every file in the matching `$BH_AGENT_WORKSPACE/domain-skills/<site>/` directory before inventing an approach.**

## Usage

```bash
browser-harness <<'PY'
print(page_info())
PY
```

- Invoke as `browser-harness`. Use heredocs for multi-line commands.
- Helpers are pre-imported. `run.py` calls `ensure_daemon()` before `exec`.
- First navigation is `new_tab(url)`, not `goto_url(url)`.
- The normal local flow attaches to the running Chrome/Chromium CDP endpoint. No browser ids or local profile selection.

## JavaScript Execution Note

`js()` requires an executable expression. Use `js("document.body.innerText")` or `js("(() => document.body.innerText)()")`; never pass an uninvoked function such as `js("() => document.body.innerText")`, which serializes as `{}`.

## Page Workflow

- Prefer to find elements with the accessibility tree, not screenshots: `cdp("Accessibility.getFullAXTree")["nodes"]` has every element's role, name, and `backendDOMNodeId` — filter in Python before printing (it is thousands of nodes). Coordinates: `q = cdp("DOM.getBoxModel", backendNodeId=n)["model"]["content"]; x, y = sum(q[0::2])/4, sum(q[1::2])/4` (viewport px, ready for `click_at_xy`; negative/oversized means scroll first).
- Clicking: AX node -> box center -> `click_at_xy(x, y)` -> verify with a targeted `js(...)`/`page_info()` check.
- Fall back to raw HTML via `js(...)` only when the AX tree lacks the element (canvas, exotic widgets); screenshot when layout or imagery matters.
- After navigation, call `wait_for_load()`.
- If the current tab is stale or internal, call `ensure_real_tab()`.
- Use `js(...)` for DOM inspection or extraction when coordinates are the wrong tool.
- Login walls: stop and ask. Exception: use available SSO automatically when Chrome is already signed in; still stop for passwords, MFA, consent, or ambiguous account choice.
- Raw CDP is available with `cdp("Domain.method", ...)`.

## Interaction Skills

If you get stuck on a browser mechanic, read the matching file in `/tmp/browser-harness-interaction-skills/`.

- cookies.md
- cross-origin-iframes.md
- dialogs.md
- downloads.md
- drag-and-drop.md
- dropdowns.md
- iframes.md
- network-requests.md
- print-as-pdf.md
- screenshots.md
- scrolling.md
- shadow-dom.md
- tabs.md
- uploads.md
- viewport.md

## Design Constraints

- Coordinate clicks default. CDP mouse events pass through iframes/shadow/cross-origin at the compositor level.
- Core helpers stay short. Put task-specific helper additions in `$BH_AGENT_WORKSPACE/agent_helpers.py`.

## Gotchas

- Omnibox popups are not real work tabs.
- CDP target order is not Chrome's visible tab-strip order.

## Domain Skills

Only applies when `BH_DOMAIN_SKILLS=1`. Otherwise ignore domain skills.

When enabled, search `$BH_AGENT_WORKSPACE/domain-skills/<host>/` before inventing an approach. `goto_url(...)` returns up to 10 skill filenames for the navigated host.
