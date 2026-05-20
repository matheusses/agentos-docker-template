"""
AgentOS Entrypoint
==================
"""

from contextlib import asynccontextmanager
from os import getenv
from pathlib import Path

from agno.agent.agent import get_agents as load_db_agents
from agno.os import AgentOS
from agno.registry import Registry
from agno.utils.log import log_info, log_warning

from agents.browser_agent import browser_agent, playwright_mcp
from agents.code_search import code_search
from agents.web_search import web_search, web_tools
from app.settings import discover_gemini_models
from db import get_postgres_db

postgres_db = get_postgres_db()

# Code-defined agents declared in this repo.
_CODE_AGENTS = [web_search, code_search, browser_agent]
_CODE_AGENT_IDS = {a.id for a in _CODE_AGENTS}

# Studio-saved agent blueprints (rows in `agno_components`). Pull them at
# startup so they show up in `/agents`, the Registry, and the eval picker.
try:
    _db_agents = load_db_agents(db=postgres_db, exclude_component_ids=_CODE_AGENT_IDS)
except Exception as exc:  # pragma: no cover — degrade gracefully if blueprints are malformed
    log_warning(f"Failed to load Studio-saved agents from db: {exc}")
    _db_agents = []

all_agents = [*_CODE_AGENTS, *_db_agents]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
runtime_env = getenv("RUNTIME_ENV", "prd")
scheduler_base_url = getenv("AGENTOS_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Interfaces
# - The CodeSearch agent becomes available on Slack when both env vars are set
# ---------------------------------------------------------------------------
SLACK_BOT_TOKEN = getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = getenv("SLACK_SIGNING_SECRET", "")

interfaces: list = []
if SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET:
    from agno.os.interfaces.slack import Slack

    interfaces.append(
        Slack(
            agent=code_search,
            streaming=True,
            token=SLACK_BOT_TOKEN,
            signing_secret=SLACK_SIGNING_SECRET,
            resolve_user_identity=True,
        )
    )


# ---------------------------------------------------------------------------
# Lifespan — extension hook for app-level startup / teardown.
#
# AgentOS handles the MCP lifecycle (connect on startup, close on shutdown).
# Keep this hook in place so you can plug in your own setup as needed.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):  # type: ignore[no-untyped-def]
    log_info("AgentOS lifespan: startup")
    try:
        yield
    finally:
        log_info("AgentOS lifespan: shutdown")


# ---------------------------------------------------------------------------
# Registry — exposes the Vertex Gemini catalog to the os.agno Studio portal
# via the `/registry` endpoint. Built once at module import.
# ---------------------------------------------------------------------------
registry = Registry(
    name="AgentOS Registry",
    models=discover_gemini_models(),
    tools=[web_tools, playwright_mcp],
    dbs=[postgres_db],
    agents=all_agents,
)

# ---------------------------------------------------------------------------
# Create AgentOS
# ---------------------------------------------------------------------------
agent_os = AgentOS(
    name="AgentOS",
    tracing=True,
    scheduler=True,
    scheduler_base_url=scheduler_base_url,
    authorization=runtime_env == "prd",
    lifespan=lifespan,
    db=postgres_db,
    agents=all_agents,
    interfaces=interfaces,
    registry=registry,
    config=str(Path(__file__).parent / "config.yaml"),
)
app = agent_os.get_app()


# ---------------------------------------------------------------------------
# Workaround: agno 2.6.8 `/models` is broken — the upstream handler skips
# every agent because of an `isinstance(agent, AgentProtocol)` check that
# always matches. Studio's "Run new evaluation" modal reads from `/models`,
# so without this it sees an empty list. We rebind the route to enumerate
# unique (id, provider) pairs across `agent_os.agents`, `agent_os.teams`,
# and the registry's model catalog.
#
# Delete this block when the upstream fix lands and the pinned agno version
# is bumped past it.
# ---------------------------------------------------------------------------
from agno.os.schema import Model as _Model  # noqa: E402


async def _get_models_fixed() -> list[_Model]:
    unique: dict[tuple[str, str], _Model] = {}

    def _add(model: object) -> None:
        mid = getattr(model, "id", None)
        provider = getattr(model, "provider", None)
        if mid and provider:
            unique.setdefault((mid, provider), _Model(id=mid, provider=provider))

    for item in (agent_os.agents or []):
        _add(getattr(item, "model", None))
    for item in (agent_os.teams or []):
        _add(getattr(item, "model", None))
    for item in (getattr(registry, "models", None) or []):
        _add(item)

    return sorted(unique.values(), key=lambda m: m.id)


app.router.routes = [
    r for r in app.router.routes
    if not (getattr(r, "path", None) == "/models" and "GET" in getattr(r, "methods", set()))
]
app.add_api_route(
    "/models",
    _get_models_fixed,
    methods=["GET"],
    response_model=list[_Model],
    response_model_exclude_none=True,
    tags=["Core"],
    operation_id="get_models",
    summary="Get Available Models",
)


if __name__ == "__main__":
    agent_os.serve(app="app.main:app", reload=runtime_env == "dev")
