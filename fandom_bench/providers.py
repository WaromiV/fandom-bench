import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable


@dataclass
class ProviderConfig:
    provider: str
    model: str
    name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or f"{self.provider}-{self.model}"


@dataclass
class ProviderDetails:
    env_var: str
    import_path: str
    pip_package: str


SUPPORTED_PROVIDERS: Dict[str, ProviderDetails] = {
    "openai": ProviderDetails(
        env_var="OPENAI_API_KEY",
        import_path="langchain_openai.ChatOpenAI",
        pip_package="langchain-openai",
    ),
    "anthropic": ProviderDetails(
        env_var="ANTHROPIC_API_KEY",
        import_path="langchain_anthropic.ChatAnthropic",
        pip_package="langchain-anthropic",
    ),
    "cohere": ProviderDetails(
        env_var="COHERE_API_KEY",
        import_path="langchain_cohere.ChatCohere",
        pip_package="langchain-cohere",
    ),
    "mistral": ProviderDetails(
        env_var="MISTRAL_API_KEY",
        import_path="langchain_mistralai.ChatMistralAI",
        pip_package="langchain-mistralai",
    ),
    "groq": ProviderDetails(
        env_var="GROQ_API_KEY",
        import_path="langchain_groq.ChatGroq",
        pip_package="langchain-groq",
    ),
}


class ProviderSetupError(Exception):
    def __init__(self, provider: str, reason: str, detail: Optional[str] = None):
        super().__init__(detail or reason)
        self.provider = provider
        self.reason = reason
        self.detail = detail or reason


def _split_import_path(import_path: str) -> tuple[str, str]:
    module, _, attr = import_path.rpartition(".")
    if not module or not attr:
        raise ValueError(f"Invalid import path: {import_path}")
    return module, attr


def build_chat_model(config: ProviderConfig) -> Runnable[Dict[str, Any], BaseChatModel]:
    """Instantiate a chat model for the given provider."""
    provider = config.provider.lower()
    details = SUPPORTED_PROVIDERS.get(provider)
    if not details:
        raise ProviderSetupError(provider, "unsupported_provider", f"Provider '{provider}' is not registered")

    if not os.getenv(details.env_var):
        raise ProviderSetupError(provider, "missing_api_key", f"Set {details.env_var} to enable this provider")

    module_path, attr = _split_import_path(details.import_path)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
    except ImportError as exc:
        raise ProviderSetupError(
            provider,
            "missing_dependency",
            f"Install '{details.pip_package}' to use provider '{provider}' ({exc})",
        ) from exc
    except AttributeError as exc:
        raise ProviderSetupError(provider, "invalid_import", str(exc)) from exc

    try:
        return cls(model=config.model, **config.kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ProviderSetupError(provider, "init_failed", str(exc)) from exc
