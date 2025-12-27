import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.runnables import Runnable

from .dataset import BenchmarkRecord
from .providers import ProviderConfig, ProviderSetupError, build_chat_model
from .supervisor import Supervisor, SupervisorConfig, SupervisorVerdict


@dataclass
class BenchmarkResult:
    record_id: str
    provider: str
    model: str
    status: str
    latency_seconds: Optional[float]
    output: Optional[str]
    supervisor: Optional[SupervisorVerdict]
    detail: Optional[str] = None


class BenchmarkRunner:
    def __init__(
        self,
        providers: List[ProviderConfig],
        supervisor_config: SupervisorConfig,
        max_concurrency: int = 3,
        temperature: float = 0.0,
        dry_run: bool = False,
    ):
        self.providers = providers
        self.supervisor = Supervisor(supervisor_config)
        self.max_concurrency = max_concurrency
        self.temperature = temperature
        self.dry_run = dry_run

        self._chat_models: Dict[str, Runnable] = {}
        self._provider_status: Dict[str, str] = {}
        self._prepare_providers()

    def _prepare_providers(self) -> None:
        for config in self.providers:
            if self.dry_run:
                self._provider_status[config.label] = "dry_run"
                continue
            try:
                cfg = ProviderConfig(
                    provider=config.provider,
                    model=config.model,
                    name=config.name,
                    kwargs={**{"temperature": self.temperature}, **config.kwargs},
                )
                self._chat_models[config.label] = build_chat_model(cfg)
                self._provider_status[config.label] = "ready"
            except ProviderSetupError as exc:
                self._provider_status[config.label] = f"skipped ({exc.reason}) - {exc.detail}"

    async def _run_one(
        self, record: BenchmarkRecord, provider_label: str, chat_model: Runnable
    ) -> BenchmarkResult:
        start = time.perf_counter()
        try:
            output_msg = await chat_model.ainvoke(record.prompt)
            latency = time.perf_counter() - start
            text = getattr(output_msg, "content", str(output_msg))
            supervisor = await self.supervisor.evaluate(record.prompt, record.expected_answer, text)
            return BenchmarkResult(
                record_id=record.row_id,
                provider=provider_label,
                model=provider_label,
                status="ok",
                latency_seconds=latency,
                output=text,
                supervisor=supervisor,
            )
        except Exception as exc:  # pragma: no cover - network/runtime defensive path
            latency = time.perf_counter() - start
            return BenchmarkResult(
                record_id=record.row_id,
                provider=provider_label,
                model=provider_label,
                status="error",
                latency_seconds=latency,
                output=None,
                supervisor=None,
                detail=str(exc),
            )

    async def run(self, dataset: List[BenchmarkRecord]) -> List[BenchmarkResult]:
        if self.dry_run:
            return [
                BenchmarkResult(
                    record_id=record.row_id,
                    provider=label,
                    model=label,
                    status="skipped",
                    latency_seconds=None,
                    output=None,
                    supervisor=None,
                    detail=reason,
                )
                for record in dataset
                for label, reason in self._provider_status.items()
            ]

        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = []
        for record in dataset:
            for label, chat_model in self._chat_models.items():
                if not chat_model:
                    continue

                async def task(rec: BenchmarkRecord = record, lbl: str = label, model=chat_model):
                    async with semaphore:
                        return await self._run_one(rec, lbl, model)

                tasks.append(asyncio.create_task(task()))

        results = await asyncio.gather(*tasks)

        skipped_results: List[BenchmarkResult] = []
        for label, status in self._provider_status.items():
            if status.startswith("skipped"):
                for record in dataset:
                    skipped_results.append(
                        BenchmarkResult(
                            record_id=record.row_id,
                            provider=label,
                            model=label,
                            status="skipped",
                            latency_seconds=None,
                            output=None,
                            supervisor=None,
                            detail=status,
                        )
                    )

        return results + skipped_results

    def provider_statuses(self) -> Dict[str, str]:
        if self.dry_run:
            return {label: "dry_run" for label in self._provider_status}
        return self._provider_status
