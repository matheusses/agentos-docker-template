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
from app.fallback_agent_browser import router as fallback_agent_browser_router
from app.settings import discover_gemini_models
from db import get_postgres_db

postgres_db = get_postgres_db()

# Code-defined agents declared in this repo.
_CODE_AGENTS = [web_search, code_search, browser_agent]
_CODE_AGENT_IDS = {a.id for a in _CODE_AGENTS}

# Studio-saved agent blueprints (rows in `agno_components`). Loaded once at
# startup and merged with the code-defined set so the eval router's
# `get_agent_by_id` can resolve them — that lookup only scans `os.agents`,
# DB-stored blueprints alone are not enough.
try:
    _DB_AGENTS = load_db_agents(db=postgres_db, exclude_component_ids=_CODE_AGENT_IDS)
except Exception as exc:  # pragma: no cover — degrade gracefully if blueprints are malformed
    log_warning(f"Failed to load Studio-saved agents from db: {exc}")
    _DB_AGENTS = []

ALL_AGENTS = [*_CODE_AGENTS, *_DB_AGENTS]

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
    # `agents` left empty on purpose. `AgentOS(agents=...)` auto-populates
    # `registry.agents` at init, then `/components` excludes every agent ID it
    # finds in the registry. We override that list back to empty after init
    # (see below) so DB-stored blueprints still show up on Studio's Agents
    # page while remaining runnable for eval.
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
    agents=ALL_AGENTS,
    interfaces=interfaces,
    registry=registry,
    config=str(Path(__file__).parent / "config.yaml"),
)

# Wipe the auto-populated `registry.agents` so `/components` does not exclude
# our agents (Studio's Agents page reads from there). Eval/run routing keeps
# working because it scans `os.agents`, which is unchanged.
registry.agents = []

app = agent_os.get_app()
app.include_router(fallback_agent_browser_router)

# AgentOS also exposes a generic `/workflows/{workflow_id}/runs` route. Put the
# concrete fallback route before that parameterized route so UPP Integrations
# always reaches the CDP-aware implementation below.
app.router.routes = sorted(
    app.router.routes,
    key=lambda route: 0 if getattr(route, "path", None) == "/workflows/fallback-agent-browser/runs" else 1,
)


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

    for item in agent_os.agents or []:
        _add(getattr(item, "model", None))
    for item in agent_os.teams or []:
        _add(getattr(item, "model", None))
    for item in getattr(registry, "models", None) or []:
        _add(item)

    return sorted(unique.values(), key=lambda m: m.id)


app.router.routes = [
    r
    for r in app.router.routes
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


# ---------------------------------------------------------------------------
# Wrap the upstream `/agents` handler so Studio's "Run new evaluation" picker
# can see DB-loaded blueprints. The picker filters by `is_component=False`,
# which agno only sets for agents registered in code — DB blueprints are
# tagged `True` and disappear from the dropdown. We rewrite the flag on the
# way out so every agent shows up in the eval modal.
#
# Remove this once Studio stops filtering by `is_component`.
# ---------------------------------------------------------------------------
from starlette.requests import Request as _Request  # noqa: E402
from starlette.responses import JSONResponse as _JSONResponse  # noqa: E402

_AGENTS_GET_ENDPOINT = next(
    (
        getattr(r, "endpoint")
        for r in app.router.routes
        if getattr(r, "path", None) == "/agents" and "GET" in getattr(r, "methods", set())
    ),
    None,
)


async def _get_agents_flattened(request: _Request) -> _JSONResponse:
    items = await _AGENTS_GET_ENDPOINT(request)  # type: ignore[misc]
    payload: list[dict] = []
    seen: set[str] = set()
    for item in items:
        body = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        agent_id = body.get("id")
        if agent_id is None or agent_id in seen:
            continue
        seen.add(agent_id)
        body["is_component"] = False
        payload.append(body)
    return _JSONResponse(payload)


if _AGENTS_GET_ENDPOINT is not None:
    app.router.routes = [
        r
        for r in app.router.routes
        if not (getattr(r, "path", None) == "/agents" and "GET" in getattr(r, "methods", set()))
    ]
    app.add_api_route(
        "/agents",
        _get_agents_flattened,
        methods=["GET"],
        tags=["Agents"],
        operation_id="get_agents",
        summary="Get Available Agents",
    )


if __name__ == "__main__":
    agent_os.serve(app="app.main:app", reload=runtime_env == "dev")
