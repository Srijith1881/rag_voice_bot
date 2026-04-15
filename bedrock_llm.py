"""AWS Bedrock LLM integration for LiveKit agents."""

import asyncio
import json
import os
import uuid
from dotenv import load_dotenv
import boto3

from livekit.agents.llm import LLM, ChatContext, LLMStream, ChatChunk, ChoiceDelta
from livekit.agents.llm.llm import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

load_dotenv(".env")


class BedrockLLMStream(LLMStream):
    """LiveKit LLMStream implementation for AWS Bedrock Claude."""

    def __init__(
        self,
        llm: "BedrockLLM",
        *,
        chat_ctx: ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)

    async def _run(self) -> None:
        """Call Bedrock synchronously in executor and push ChatChunks to event channel."""

        # Use Anthropic format: returns (messages, format_data) where format_data has system_messages
        messages, format_data = self._chat_ctx.to_provider_format("anthropic")

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "messages": messages,
        }

        if format_data.system_messages:
            body["system"] = "\n\n".join(format_data.system_messages)

        # Run blocking boto3 call in threadpool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._llm._client.invoke_model(
                modelId=self._llm._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            ),
        )

        response_body = json.loads(response["body"].read())
        text = response_body["content"][0]["text"]

        self._event_ch.send_nowait(
            ChatChunk(
                id=str(uuid.uuid4()),
                delta=ChoiceDelta(role="assistant", content=text),
            )
        )


class BedrockLLM(LLM):
    """LiveKit-compatible AWS Bedrock Claude LLM."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str = "us-east-1",
    ):
        super().__init__()
        self._model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        self._region = region

        try:
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Bedrock client: {e}\n"
                "Ensure AWS credentials are configured. Run: aws sso login --profile <your-profile>"
            )

    @property
    def model(self) -> str:
        return self._model_id

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> LLMStream:
        return BedrockLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    def prewarm(self) -> None:
        """Pre-warm connection (sync method, matches LLM base class)."""
        pass

    async def aclose(self) -> None:
        """Cleanup."""
        pass
