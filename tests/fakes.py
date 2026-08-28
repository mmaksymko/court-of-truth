import asyncio

from pydantic import HttpUrl

from court.config import Settings
from court.forensics.registry import LoadedDetector
from court.forensics.schemas import DetectorMeta, Scope
from court.tribunal.schemas import (
    Argument,
    Claim,
    Role,
    Source,
    Verdict,
    VerdictProbabilities,
)
from court.tribunal.telemetry import CallTelemetry, SearchRecord, SearchSource, Usage


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


def make_detector(
    detector_id: str,
    scope: Scope,
    labels: tuple[str, str],
    positive: str,
    probability: float,
    threshold: float = 0.5,
) -> LoadedDetector:
    meta = DetectorMeta(
        id=detector_id,
        scope=scope,
        backend="sklearn",
        labels=labels,
        positive_label=positive,
        threshold=threshold,
        version="test",
        metrics={},
    )
    return LoadedDetector(
        meta=meta,
        predict=lambda _text: probability,
        evidence=lambda _text: ["мітка"],
    )


class FakeLLM:
    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.active_arguments = 0
        self.max_active_arguments = 0
        self.closed = False
        self._telemetry: list[CallTelemetry] | None = None

    def enable_telemetry(self) -> None:
        self._telemetry = []

    def drain_telemetry(self) -> list[CallTelemetry]:
        drained = self._telemetry or []
        if self._telemetry is not None:
            self._telemetry = []
        return drained

    def _record(self, kind: str) -> None:
        if self._telemetry is None:
            return
        self._telemetry.append(
            CallTelemetry(
                kind=kind,
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                searches=(
                    SearchRecord(
                        query=f"запит {kind}",
                        sources=(
                            SearchSource(url="https://example.org/evidence", title="Приклад"),
                        ),
                    ),
                ),
            )
        )

    async def argue(self, role: Role, _user: str) -> Argument:
        self.active_arguments += 1
        self.max_active_arguments = max(self.max_active_arguments, self.active_arguments)
        await asyncio.sleep(self.delay)
        self.active_arguments -= 1
        self._record(role)
        return Argument(
            role=role,
            thesis=f"теза {role}",
            claims=[Claim(id=f"{role[0]}c1", text="сигнал")],
            cited_detectors=["jeansa"],
            sources=[
                Source(
                    claim_id=f"{role[0]}c1",
                    url=HttpUrl("https://example.org/evidence"),
                    title="Приклад",
                    excerpt="Перевірений фрагмент джерела.",
                    supports="підтверджує тезу",
                )
            ],
        )

    async def judge(self, _user: str) -> Verdict:
        self._record("judge")
        return Verdict(
            label="questionable",
            probabilities=VerdictProbabilities(reliable=0.25, questionable=0.6, unreliable=0.15),
            rationale="через сукупність сигналів",
            key_signals=["jeansa"],
            source_assessment="джерела помірно надійні",
            cited_detectors=["jeansa"],
        )

    async def aclose(self) -> None:
        self.closed = True
