from collections.abc import Mapping
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from court.config import Settings
from court.forensics.registry import LoadedDetector


def get_settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def get_registry(request: Request) -> Mapping[str, LoadedDetector]:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(503, "detector registry is not ready")
    return cast("Mapping[str, LoadedDetector]", registry)


def rate_subject(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"


SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[Mapping[str, LoadedDetector], Depends(get_registry)]
