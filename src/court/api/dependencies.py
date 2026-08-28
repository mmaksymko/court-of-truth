from collections.abc import Mapping
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from court.config import Settings
from court.forensics.registry import LoadedDetector
from court.tribunal.llm import LLMClient


def get_settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def get_registry(request: Request) -> Mapping[str, LoadedDetector]:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(503, "detector registry is not ready")
    return cast("Mapping[str, LoadedDetector]", registry)


def get_llm(request: Request) -> LLMClient:
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(503, "OpenAI API key is not configured")
    return cast("LLMClient", llm)


def rate_subject(request: Request) -> str:
    client = request.client
    if client:
        return client.host
    return "unknown"


SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[Mapping[str, LoadedDetector], Depends(get_registry)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
