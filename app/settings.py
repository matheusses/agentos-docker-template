"""
App Settings
============

Shared runtime objects for the platform.
"""

from os import getenv

from agno.models.google import Gemini
from google import genai

# Skip non-chat Gemini variants in the catalog (TTS, audio, embedding, image-gen).
_EXCLUDED_SUFFIXES = ("-tts", "-native-audio", "-image-preview", "-embedding")


def _gcp_project_and_location() -> tuple[str, str]:
    project = getenv("GCP_PROJECT_ID")
    location = getenv("GCP_LOCATION")
    if not project or not location:
        raise RuntimeError("GCP_PROJECT_ID and GCP_LOCATION must be set to use Vertex AI.")
    return project, location


def default_model() -> Gemini:
    """Fresh model instance per agent — avoids shared-state footguns."""
    project, location = _gcp_project_and_location()
    return Gemini(
        id=getenv("GEMINI_MODEL_ID", "gemini-2.5-pro"),
        vertexai=True,
        project_id=project,
        location=location,
    )


def create_genai_client() -> genai.Client:
    """
    Returns a new GenAI client instance using Vertex AI with Workload Identity.

    Auth is delegated to Application Default Credentials — locally run
    `gcloud auth application-default login`; on GCP, attach a service account
    to the workload.
    """
    project, location = _gcp_project_and_location()
    return genai.Client(vertexai=True, project=project, location=location)


def discover_gemini_models() -> list[Gemini]:
    """
    Discover every chat-capable Gemini model visible to this Vertex project
    and return them as agno `Gemini` instances ready to register in `Registry`.
    """
    project, location = _gcp_project_and_location()
    client = create_genai_client()
    out: list[Gemini] = []
    for m in client.models.list():
        name = getattr(m, "name", "") or ""
        if "gemini" not in name.lower():
            continue
        model_id = name.split("/")[-1]
        if any(model_id.endswith(suffix) or suffix.strip("-") in model_id for suffix in _EXCLUDED_SUFFIXES):
            continue
        out.append(Gemini(id=model_id, vertexai=True, project_id=project, location=location))
    return sorted(out, key=lambda g: g.id)
