import pytest
from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.completion import (
    CompletionRequest,
    CompletionSummary,
    StreamChunk,
    StreamSummary,
)
from aleph_alpha_client.prompt import (
    ControlTokenOverlap,
    Image,
    Prompt,
    Text,
    TextControl,
    Tokens,
)

from .conftest import llama_prompt


async def test_can_complete_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )

    response = await async_client.complete(request, model=model_name)
    assert len(response.completions) == 1
    assert response.model_version is not None


async def test_can_use_streaming_support_with_async_client(
    async_client: AsyncClient, model_name: str
):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )

    stream_items = [
        stream_item
        async for stream_item in async_client.complete_with_streaming(
            request, model=model_name
        )
    ]

    assert len(stream_items) >= 3
    assert isinstance(stream_items[-3], StreamChunk)
    assert isinstance(stream_items[-2], StreamSummary)
    assert isinstance(stream_items[-1], CompletionSummary)


def test_complete_maximum_tokens_none(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_text("Hello, World!"),
        maximum_tokens=7,
        stop_sequences=[","],
    )

    response = sync_client.complete(request, model=model_name)
    assert len(response.completions) == 1
    assert response.completions[0].completion is not None
    assert len(response.completions[0].completion) < 100


def test_complete(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt(
            [
                Text(
                    "Hello, World!",
                    controls=[
                        TextControl(start=1, length=5, factor=0.5),
                        TextControl(
                            start=1,
                            length=5,
                            factor=0.5,
                            token_overlap=ControlTokenOverlap.Complete,
                        ),
                    ],
                )
            ]
        ),
        maximum_tokens=7,
        tokens=False,
        log_probs=0,
        logit_bias={1: 2.0},
    )

    response = sync_client.complete(request, model=model_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


def test_complete_with_token_ids(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_tokens([49222, 2998]),  # Hello world
        maximum_tokens=32,
    )

    response = sync_client.complete(request, model=model_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


def test_complete_with_optimized_prompt(
    sync_client: Client, model_name: str, prompt_image: Image
):
    prompt_tokens = Tokens.from_token_ids([1, 2])
    request = CompletionRequest(
        prompt=Prompt(
            [
                Text.from_text(" Hello "),
                Text.from_text(" world! "),
                prompt_image,
                prompt_tokens,
                Text.from_text(" The "),
                Text.from_text(" end "),
            ]
        ),
        maximum_tokens=5,
    )

    response = sync_client.complete(request, model=model_name)

    assert response.optimized_prompt is not None
    assert response.optimized_prompt.items[0] == Text.from_text("Hello  world! ")
    assert response.optimized_prompt.items[2] == prompt_tokens
    assert response.optimized_prompt.items[3] == Text.from_text(" The  end")
    assert isinstance(response.optimized_prompt.items[1], Image)


def test_complete_with_echo(sync_client: Client, model_name: str, prompt_image: Image):
    request = CompletionRequest(
        prompt=Prompt.from_text("Hello world"),
        maximum_tokens=0,
        tokens=True,
        echo=True,
        log_probs=0,
    )

    response = sync_client.complete(request, model=model_name)
    completion_result = response.completions[0]
    assert completion_result.completion == " Hello world"
    assert completion_result.completion_tokens is not None
    assert len(completion_result.completion_tokens) > 0
    assert completion_result.log_probs is not None
    assert len(completion_result.log_probs) > 0


def test_num_tokens_prompt_total_with_best_of(sync_client: Client, model_name: str):
    tokens = [49222, 2998]  # Hello world
    best_of = 2
    request = CompletionRequest(
        prompt=Prompt.from_tokens(tokens),
        best_of=best_of,
        maximum_tokens=1,
    )

    response = sync_client.complete(request, model=model_name)
    assert response.num_tokens_prompt_total == len(tokens) * best_of


def test_num_tokens_generated_with_best_of(sync_client: Client, model_name: str):
    hello_world = [49222, 2998]  # Hello world
    best_of = 2
    request = CompletionRequest(
        prompt=Prompt.from_tokens(hello_world),
        best_of=best_of,
        maximum_tokens=1,
        tokens=True,
    )

    response = sync_client.complete(request, model=model_name)
    completion_result = response.completions[0]
    assert completion_result.completion_tokens is not None
    number_tokens_completion = len(completion_result.completion_tokens)

    assert response.num_tokens_generated == best_of * number_tokens_completion
