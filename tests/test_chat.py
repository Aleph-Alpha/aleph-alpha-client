import pytest

from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.chat import (
    ChatRequest,
    ChatStreamChunk,
    FinishReason,
    Message,
    Role,
    StreamOptions,
    Usage,
    stream_chat_item_from_json,
)


@pytest.mark.system_test
async def test_can_not_chat_with_all_models(async_client: AsyncClient, model_name: str):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=model_name,
        maximum_tokens=7,
    )

    with pytest.raises(ValueError):
        await async_client.chat(request, model=model_name)


def test_can_chat_with_chat_model(sync_client: Client, chat_model_name: str):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
    )

    response = sync_client.chat(request, model=chat_model_name)
    assert response.message.role == Role.Assistant
    assert response.message.content is not None
    assert isinstance(response.finish_reason, FinishReason)


async def test_can_chat_with_async_client(
    async_client: AsyncClient, chat_model_name: str
):
    system_msg = Message(role=Role.System, content="You are a helpful assistant.")
    user_msg = Message(role=Role.User, content="Hello, how are you?")
    request = ChatRequest(
        messages=[system_msg, user_msg],
        model=chat_model_name,
    )

    response = await async_client.chat(request, model=chat_model_name)
    assert response.message.role == Role.Assistant
    assert response.message.content is not None


async def test_can_chat_with_streaming_support(
    async_client: AsyncClient, chat_model_name: str
):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
    )

    stream = async_client.chat_with_streaming(request, model=chat_model_name)
    stream_items = [stream_item async for stream_item in stream]

    first = stream_items[0]
    assert isinstance(first, ChatStreamChunk) and first.role is not None
    assert all(
        isinstance(item, ChatStreamChunk) and item.content is not None
        for item in stream_items[1:-1]
    )
    assert isinstance(stream_items[-1], FinishReason)


async def test_usage_response_is_parsed():
    # Given an API response with usage data and no choice
    data = {
        "choices": [],
        "created": 1730133402,
        "model": "llama-3.1-70b-instruct",
        "system_fingerprint": ".unknown.",
        "object": "chat.completion.chunk",
        "usage": {"prompt_tokens": 31, "completion_tokens": 88, "total_tokens": 119},
    }

    # When parsing it
    result = stream_chat_item_from_json(data)

    # Then a usage instance is returned
    assert isinstance(result, Usage)
    assert result.prompt_tokens == 31


async def test_finish_reason_response_is_parsed():
    # Given an API response with finish reason and no choice
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "delta": {},
                "logprobs": None,
            }
        ],
        "created": 1730133402,
        "model": "llama-3.1-70b-instruct",
        "system_fingerprint": ".unknown.",
        "object": "chat.completion.chunk",
    }

    # When parsing it
    result = stream_chat_item_from_json(data)

    # Then a finish reason instance is returned
    assert isinstance(result, FinishReason)
    assert result == FinishReason.Stop


def test_chunk_response_is_parsed():
    # Given an API response without usage data
    data = {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {"content": " way, those clothes you're wearing"},
                "logprobs": None,
            }
        ],
        "created": 1730133401,
        "model": "llama-3.1-70b-instruct",
        "system_fingerprint": None,
        "object": "chat.completion.chunk",
        "usage": None,
    }

    # When parsing it
    result = stream_chat_item_from_json(data)

    # Then a ChatStreamChunk instance is returned
    assert isinstance(result, ChatStreamChunk)
    assert result.content == " way, those clothes you're wearing"


async def test_stream_options(async_client: AsyncClient, chat_model_name: str):
    # Given a request with include usage options set
    stream_options = StreamOptions(include_usage=True)
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
        stream_options=stream_options,
    )

    # When receiving the chunks
    stream = async_client.chat_with_streaming(request, model=chat_model_name)
    stream_items = [stream_item async for stream_item in stream]

    # Then
    assert all(isinstance(item, ChatStreamChunk) for item in stream_items[:-2])
    assert isinstance(stream_items[-2], FinishReason)
    assert isinstance(stream_items[-1], Usage)


def test_steering_chat(sync_client: Client, chat_model_name: str):
    base_request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
    )

    steered_request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
        steering_concepts=["_worker/slang"],
    )

    base_response = sync_client.chat(base_request, model=chat_model_name)
    steered_response = sync_client.chat(steered_request, model=chat_model_name)

    base_completion_result = base_response.message.content
    steered_completion_result = steered_response.message.content

    assert base_completion_result
    assert steered_completion_result
    assert base_completion_result != steered_completion_result
