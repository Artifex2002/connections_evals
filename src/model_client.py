from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ModelResponse:
    raw_output: str
    latency_ms: float

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    provider_request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleModel:
    """
    Generic client for any OpenAI-compatible inference server.

    Works with vLLM and can therefore be reused for Nemotron, QwQ,
    DeepSeek-R1, Qwen3, Gemma, and similar models.
    """

    provider_name = "vllm"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        system_prompt_prefix: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt_prefix = system_prompt_prefix

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def recover_answer(
        self,
        *,
        system_prompt: str,
        current_state_prompt: str,
        original_generation: str,
        recovery_type: str,
        available_words: list[str] | None = None,
        max_tokens: int = 192,
    ) -> ModelResponse:
        """
        Cheap constrained follow-up used only when the original generation
        did not yield a parseable final guess.

        The original reasoning is NOT replaced or modified.

        recovery_type:
            - "truncation_recovery":
                commit to the guess most supported by reasoning already
                produced before truncation.

            - "format_repair":
                serialize the decision already expressed in the completed
                response.
        """

        if recovery_type == "truncation_recovery":
            instruction = """
Your previous response was cut off before producing its required final answer.

Do NOT solve the puzzle again.
Do NOT reconsider the puzzle.
Do NOT explore new alternatives.
Do NOT explain your answer.

Based only on the reasoning already produced in YOUR ORIGINAL RESPONSE,
commit to the four-word guess most supported by that reasoning.

Return only the required structured answer.
""".strip()

        elif recovery_type == "format_repair":
            instruction = """
Your previous response finished, but its final answer could not be parsed.

Do NOT solve the puzzle again.
Do NOT reconsider or change your decision.
Do NOT perform additional analysis.
Do NOT explain your answer.

Extract the four-word guess and proposed category already expressed in
YOUR ORIGINAL RESPONSE.

Return only the required structured answer.
""".strip()

        else:
            raise ValueError(
                f"Unknown recovery_type: {recovery_type}"
            )

        if available_words:
            guess_item_schema = {
                "type": "string",
                "enum": list(available_words),
            }
        else:
            guess_item_schema = {
                "type": "string",
            }

        recovery_schema = {
            "type": "object",
            "properties": {
                "guess": {
                    "type": "array",
                    "items": guess_item_schema,
                    "minItems": 4,
                    "maxItems": 4,
                },
                "category": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "required": [
                "guess",
                "category",
            ],
            "additionalProperties": False,
        }

        recovery_prompt = f"""
{instruction}

CURRENT AUTHORITATIVE GAME STATE
--------------------------------
{current_state_prompt}

YOUR ORIGINAL RESPONSE
----------------------
{original_generation}
""".strip()

        started = time.perf_counter()

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are performing answer recovery only. "
                        "Return the requested structured answer."
                    ),
                },
                {
                    "role": "user",
                    "content": recovery_prompt,
                },
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "connections_recovered_answer",
                    "schema": recovery_schema,
                },
            },
        )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        message = completion.choices[0].message
        raw_output = message.content or ""

        usage = completion.usage

        return ModelResponse(
            raw_output=raw_output,
            latency_ms=latency_ms,
            prompt_tokens=(
                usage.prompt_tokens
                if usage is not None
                else None
            ),
            completion_tokens=(
                usage.completion_tokens
                if usage is not None
                else None
            ),
            total_tokens=(
                usage.total_tokens
                if usage is not None
                else None
            ),
            provider_request_id=completion.id,
            metadata={
                "finish_reason": (
                    completion.choices[0].finish_reason
                ),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "recovery_type": recovery_type,
                "structured_output": True,
                "schema_name": (
                    "connections_recovered_answer"
                ),
            },
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelResponse:

        if self.system_prompt_prefix:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt_prefix,
                },
                {
                    "role": "user",
                    "content": (
                        f"{system_prompt}\n\n"
                        f"{user_prompt}"
                    ),
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

        started = time.perf_counter()

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        message = completion.choices[0].message

        # We deliberately launch vLLM without a reasoning parser for this
        # experiment. Therefore message.content should contain the complete
        # generated sequence, including <think> tags when the model emits them.
        raw_output = message.content or ""

        usage = completion.usage

        return ModelResponse(
            raw_output=raw_output,
            latency_ms=latency_ms,
            prompt_tokens=(
                usage.prompt_tokens
                if usage is not None
                else None
            ),
            completion_tokens=(
                usage.completion_tokens
                if usage is not None
                else None
            ),
            total_tokens=(
                usage.total_tokens
                if usage is not None
                else None
            ),
            provider_request_id=completion.id,
            metadata={
                "finish_reason": completion.choices[0].finish_reason,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "raw_output_reconstructed": False,
                "system_prompt_prefix": self.system_prompt_prefix,
            },
        )

    def health_check(self) -> list[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]
