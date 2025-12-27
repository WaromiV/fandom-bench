import os
from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .providers import ProviderSetupError, build_chat_model, ProviderConfig


class SupervisorVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    decision: str
    notes: str


@dataclass
class SupervisorConfig:
    model: str = "gpt-5"
    enabled: bool = True
    temperature: float = 0.0


class Supervisor:
    def __init__(self, config: SupervisorConfig):
        self.config = config
        self._chain = None
        self._setup_error: Optional[str] = None
        if not config.enabled:
            self._setup_error = "disabled"
            return
        try:
            provider_cfg = ProviderConfig(provider="openai", model=config.model, kwargs={"temperature": config.temperature})
            llm = build_chat_model(provider_cfg)
        except ProviderSetupError as exc:
            self._setup_error = f"{exc.reason}: {exc.detail}"
            return

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a strict grader. Score the candidate answer compared to the expected answer. "
                        "Return a score between 0 and 1, set decision to one of correct/partial/incorrect, "
                        "and include a short note. Be concise."
                    ),
                ),
                (
                    "user",
                    (
                        "Prompt: {prompt}\n"
                        "Expected answer: {expected}\n"
                        "Candidate answer: {candidate}\n"
                        "Provide your evaluation."
                    ),
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(SupervisorVerdict)

    @property
    def is_ready(self) -> bool:
        return self._chain is not None

    @property
    def error(self) -> Optional[str]:
        return self._setup_error

    async def evaluate(self, prompt: str, expected: Optional[str], candidate: str) -> Optional[SupervisorVerdict]:
        if not self._chain:
            return None
        expected_text = expected or "(no expected answer provided)"
        return await self._chain.ainvoke({"prompt": prompt, "expected": expected_text, "candidate": candidate})
