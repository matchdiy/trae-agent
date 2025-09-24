# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DeepSeek provider configuration."""

import os

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class DeepSeekProvider(ProviderConfig):
    """DeepSeek provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create DeepSeek client with DeepSeek base URL."""
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "deepseek"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "deepseek"

    def get_extra_headers(self) -> dict[str, str]:
        """Get DeepSeek-specific headers."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Most modern models on DeepSeek support tool calling
        # We'll be conservative and check for known capable models

        # tool_capable_patterns = [
        #     "gpt-4",
        #     "gpt-3.5-turbo",
        #     "claude-3",
        #     "claude-2",
        #     "gemini",
        #     "mistral",
        #     "llama-3",
        #     "command-r",
        # ]
        # return any(pattern in model_name.lower() for pattern in tool_capable_patterns)
        return True


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_config: ModelConfig):
        if (
            model_config.model_provider.base_url is None
            or model_config.model_provider.base_url == ""
        ):
            model_config.model_provider.base_url = "https://dpapi.cn/v1"
        super().__init__(model_config, DeepSeekProvider())
