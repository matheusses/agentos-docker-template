"""
Browser Agent
=============

Headless-browser-capable agent. Browser actions are delegated to a separate
Playwright MCP server (see the `playwright-mcp` service in `compose.yaml`) so
this process stays decoupled from the browser runtime — swap providers by
pointing `PLAYWRIGHT_MCP_URL` at any MCP-compatible browser server.

Sessions and memories use the shared Postgres so the agent shows up in the
os.agno Studio eval picker alongside the others.
"""

from os import getenv

from agno.agent import Agent
from agno.tools.mcp import MCPTools

from app.settings import default_model
from db import get_postgres_db

# Playwright MCP server — AgentOS manages MCP lifecycle (connect on startup,
# close on shutdown) for any MCPTools attached to a registered agent.
playwright_mcp = MCPTools(
    url=getenv("PLAYWRIGHT_MCP_URL", "http://playwright-mcp:8931/mcp"),
    transport="streamable-http",
)
# MCPTools forces name="MCPTools" in __init__ — override post-init so the
# Registry shows distinct names for each MCP toolkit.
playwright_mcp.name = "playwright-mcp"


BROWSER_INSTRUCTIONS = """\
Drive a real headless browser to complete tasks the user describes.

Workflow:
1. Plan the navigation steps before opening the browser. Name the URLs you intend to visit.
2. Use `navigate_to` to open pages. Use `get_page_content` to read HTML when you need to find structured data; use `screenshot` only when the user asks for a visual or when text extraction is unreliable.
3. Summarize what you observed, citing the URLs you visited as plain links.
4. Close the browser session with `close_session` when the task is complete.
5. Remember stable user preferences (sites they care about, login emails, recurring tasks) via your memory; do not memorize one-off URLs or volatile session data.
"""


browser_agent = Agent(
    id="browser-agent",
    name="Browser",
    model=default_model(),
    db=get_postgres_db(),
    tools=[playwright_mcp],
    instructions=BROWSER_INSTRUCTIONS,
    enable_agentic_memory=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)
