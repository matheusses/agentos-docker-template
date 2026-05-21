"""
Authenticated browser fallback workflow.

This endpoint implements the `fallback-agent-browser` workflow contract used by
UPP Integrations. It attaches to the caller's already-authenticated Chromium via
CDP and lets an Agno agent drive that exact page through Playwright tools.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agno.agent import Agent
from fastapi import APIRouter, Form
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field

from app.settings import default_model

router = APIRouter()

_WORKFLOW_ID = "fallback-agent-browser"
_CONTROL_SELECTOR = "input, textarea, select, button, [role='button'], a, [contenteditable='true']"


class FallbackFactoryInput(BaseModel):
    """Structured payload sent by UPP Integrations as `factory_input`."""

    model_config = ConfigDict(extra="allow")

    cdp_url: str = Field(min_length=1)
    target_url: str | None = None
    url: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    storage_state: dict[str, Any] | None = None

    @property
    def resolved_target_url(self) -> str:
        target = self.target_url or self.url
        if not target:
            raise ValueError("factory_input must include `target_url` or `url`")
        return target


class FallbackAgentOutput(BaseModel):
    """Structured result expected from the browser-driving agent."""

    success: bool
    summary: str
    error: str | None = None


@router.post(f"/workflows/{_WORKFLOW_ID}/runs")
async def run_fallback_agent_browser(
    message: str = Form(...),
    stream: str = Form("false"),
    background: str = Form("false"),
    factory_input: str = Form(...),
    version: str = Form(""),
    session_id: str | None = Form(None),
    user_id: str | None = Form(None),
) -> dict[str, Any]:
    """Run the authenticated browser fallback workflow.

    The form fields mirror AgentOS' workflow runner contract so the existing
    integrations client can call this endpoint without special-casing AgentOS.
    `stream`, `background`, and `version` are accepted for compatibility.
    """

    del stream, background, version

    run_id = str(uuid.uuid4())
    try:
        payload = FallbackFactoryInput.model_validate_json(factory_input)
        output = await _run_authenticated_browser_agent(
            message=message,
            payload=payload,
            session_id=session_id,
            user_id=user_id,
        )
    except Exception as exc:
        return {
            "run_id": run_id,
            "status": "failed",
            "content": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "run_id": run_id,
        "status": "completed" if output.success else "failed",
        "content": output.summary,
        "error": output.error,
    }


async def _run_authenticated_browser_agent(
    *,
    message: str,
    payload: FallbackFactoryInput,
    session_id: str | None,
    user_id: str | None,
) -> FallbackAgentOutput:
    target_url = payload.resolved_target_url

    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(payload.cdp_url)
        try:
            page = await _find_authenticated_page(browser.contexts, target_url)
            await page.bring_to_front()
            agent = _build_page_agent(page)
            response = await agent.arun(
                input=_build_agent_input(message=message, payload=payload),
                session_id=session_id,
                user_id=user_id,
                output_schema=FallbackAgentOutput,
            )
        finally:
            # Close only our CDP connection. This does not close the underlying
            # Chromium process owned by UPP Integrations.
            await browser.close()

    content = getattr(response, "content", None)
    if isinstance(content, FallbackAgentOutput):
        return content
    if isinstance(content, dict):
        return FallbackAgentOutput.model_validate(content)
    if isinstance(content, str):
        return FallbackAgentOutput(success=True, summary=content)
    return FallbackAgentOutput(
        success=False,
        summary="Agent returned no structured browser fallback result.",
        error="missing_agent_output",
    )


async def _find_authenticated_page(contexts: list[Any], target_url: str) -> Page:
    pages = [page for context in contexts for page in context.pages]
    for page in pages:
        if _urls_match(page.url, target_url):
            return page

    open_urls = [page.url for page in pages]
    raise RuntimeError(
        f"No existing authenticated tab matched target_url. target_url={target_url!r}; open_urls={open_urls!r}"
    )


def _urls_match(actual: str, expected: str) -> bool:
    return _normalize_url(actual) == _normalize_url(expected)


def _normalize_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def _build_page_agent(page: Page) -> Agent:
    async def inspect_page() -> str:
        """Return current URL, title, and visible form/action controls."""

        controls = await _visible_controls(page)
        payload = {
            "url": page.url,
            "title": await page.title(),
            "controls": controls,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    async def fill_selector(selector: str, value: str) -> str:
        """Fill a field by CSS selector from inspect_page()."""

        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=5000)
        await locator.fill(str(value))
        return f"filled selector {selector!r}"

    async def fill_field(field_hint: str, value: str) -> str:
        """Fill a field by visible label, placeholder, id, name, or selector."""

        await _fill_field(page, field_hint, str(value))
        return f"filled field matching {field_hint!r}"

    async def select_option(field_hint: str, value: str) -> str:
        """Select an option by field hint and option label/value."""

        await _select_option(page, field_hint, str(value))
        return f"selected option {value!r} for field {field_hint!r}"

    async def click_selector(selector: str) -> str:
        """Click an element by CSS selector from inspect_page()."""

        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=5000)
        await locator.click()
        return f"clicked selector {selector!r}"

    async def click_button(button_hint: str) -> str:
        """Click a button or button-like element by visible text/name."""

        await _click_button(page, button_hint)
        return f"clicked button matching {button_hint!r}"

    async def press_key(key: str) -> str:
        """Press a keyboard key such as Enter, Tab, ArrowDown, or Escape."""

        await page.keyboard.press(key)
        return f"pressed {key!r}"

    async def wait_for_text(text: str, timeout_ms: int = 10000) -> str:
        """Wait until the page contains visible text."""

        await page.get_by_text(re.compile(re.escape(text), re.IGNORECASE)).first.wait_for(
            state="visible",
            timeout=timeout_ms,
        )
        return f"text {text!r} is visible"

    return Agent(
        id="fallback-agent-browser-runner",
        name="Fallback Agent Browser Runner",
        model=default_model(),
        tools=[
            inspect_page,
            fill_selector,
            fill_field,
            select_option,
            click_selector,
            click_button,
            press_key,
            wait_for_text,
        ],
        instructions=(
            "You are controlling an existing authenticated browser tab via "
            "Playwright tools. Start with inspect_page. Use the provided inputs "
            "to complete the requested recovery objective on the current page. "
            "Prefer the selectors returned by inspect_page when labels are "
            "ambiguous. Do not navigate away, log out, cancel, submit, finish, "
            "or advance the surrounding workflow unless the objective explicitly "
            "requires it. Return success=true only after the page is left in the "
            "state requested by the objective."
        ),
        tool_call_limit=30,
        markdown=False,
    )


def _build_agent_input(*, message: str, payload: FallbackFactoryInput) -> str:
    return (
        f"{message}\n\n"
        "Structured factory_input:\n"
        f"{json.dumps(payload.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )


async def _visible_controls(page: Page) -> list[dict[str, Any]]:
    return await page.locator(_CONTROL_SELECTOR).evaluate_all(
        """elements => elements
          .filter(el => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.visibility !== 'hidden'
              && style.display !== 'none'
              && rect.width > 0
              && rect.height > 0;
          })
          .slice(0, 120)
          .map((el, index) => {
            const labels = el.labels
              ? Array.from(el.labels).map(label => label.innerText.trim()).filter(Boolean)
              : [];
            const selector = el.id
              ? `#${CSS.escape(el.id)}`
              : el.getAttribute('name')
                ? `${el.tagName.toLowerCase()}[name="${CSS.escape(el.getAttribute('name'))}"]`
                : `${el.tagName.toLowerCase()}:nth-of-type(${index + 1})`;
            return {
              selector,
              tag: el.tagName.toLowerCase(),
              type: el.getAttribute('type'),
              id: el.id || null,
              name: el.getAttribute('name'),
              role: el.getAttribute('role'),
              label: labels.join(' | ') || null,
              placeholder: el.getAttribute('placeholder'),
              ariaLabel: el.getAttribute('aria-label'),
              text: (el.innerText || el.value || '').trim().slice(0, 160),
              disabled: Boolean(el.disabled),
            };
          })"""
    )


async def _fill_field(page: Page, field_hint: str, value: str) -> None:
    candidates = [
        page.get_by_label(re.compile(re.escape(field_hint), re.IGNORECASE)).first,
        page.get_by_placeholder(re.compile(re.escape(field_hint), re.IGNORECASE)).first,
        page.locator(field_hint).first,
        page.locator(_attr_contains_selector("input", "name", field_hint)).first,
        page.locator(_attr_contains_selector("input", "id", field_hint)).first,
        page.locator(_attr_contains_selector("textarea", "name", field_hint)).first,
        page.locator(_attr_contains_selector("textarea", "id", field_hint)).first,
    ]
    for locator in candidates:
        try:
            await locator.wait_for(state="visible", timeout=1500)
            await locator.fill(value)
            return
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue
    raise RuntimeError(f"No visible fillable field matched {field_hint!r}")


async def _select_option(page: Page, field_hint: str, value: str) -> None:
    candidates = [
        page.get_by_label(re.compile(re.escape(field_hint), re.IGNORECASE)).first,
        page.locator(field_hint).first,
        page.locator(_attr_contains_selector("select", "name", field_hint)).first,
        page.locator(_attr_contains_selector("select", "id", field_hint)).first,
    ]
    for locator in candidates:
        try:
            await locator.wait_for(state="visible", timeout=1500)
            await locator.select_option(label=value)
            return
        except Exception:
            try:
                await locator.select_option(value=value)
                return
            except Exception:
                continue
    raise RuntimeError(f"No visible select field matched {field_hint!r}")


async def _click_button(page: Page, button_hint: str) -> None:
    pattern = re.compile(re.escape(button_hint), re.IGNORECASE)
    candidates = [
        page.get_by_role("button", name=pattern).first,
        page.get_by_text(pattern).first,
        page.locator(button_hint).first,
    ]
    for locator in candidates:
        try:
            await locator.wait_for(state="visible", timeout=1500)
            await locator.click()
            return
        except Exception:
            continue
    raise RuntimeError(f"No visible button/action matched {button_hint!r}")


def _attr_contains_selector(tag: str, attr: str, value: str) -> str:
    escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{tag}[{attr}*="{escaped_value}" i]'
