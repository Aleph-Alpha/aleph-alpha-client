import json
from pathlib import Path
from typing import List, Union

import pytest
from PIL import Image

from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.chat import (
    ChatRequest,
    ChatStreamChunk,
    FinishReason,
    Message,
    Role,
    StreamOptions,
    TextMessage,
    Usage,
    stream_chat_item_from_json,
)
from aleph_alpha_client.structured_output import JSONSchema

from .conftest import GenericClient
from .test_steering import create_sample_steering_concept_creation_request


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

    steering_concept_id = sync_client.create_steering_concept(
        create_sample_steering_concept_creation_request()
    ).id

    steered_request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
        steering_concepts=[steering_concept_id],
    )

    base_response = sync_client.chat(base_request, model=chat_model_name)
    steered_response = sync_client.chat(steered_request, model=chat_model_name)

    base_completion_result = base_response.message.content
    steered_completion_result = steered_response.message.content

    assert base_completion_result
    assert steered_completion_result
    assert base_completion_result != steered_completion_result


def test_response_format_json_schema(sync_client: Client, dummy_model_name: str):
    # This example is taken from json-schema.org:
    example_json_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/product.schema.json",
        "title": "Product",
        "description": "A product from Acme's catalog",
        "type": "object",
        "properties": {
            "productId": {
                "description": "The unique identifier for a product",
                "type": "integer"
            }
        }
    }

    request = ChatRequest(
        messages=[Message(role=Role.User, content="Give me JSON!")],
        model=dummy_model_name,
        response_format=JSONSchema(example_json_schema),
    )

    response = sync_client.chat(request, model=dummy_model_name)

    # Dummy worker simply returns the JSON schema that the user has submitted
    assert json.loads(response.message.content) == example_json_schema


@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
async def test_can_chat_with_images(generic_client: GenericClient, dummy_model_name: str):
    image_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    image = Image.open(image_path)

    request = ChatRequest(
        messages=[Message(
            role=Role.User,
            content=[
                "Describe the following image.",
                image,
            ],
        )],
        model=dummy_model_name,
        maximum_tokens=200,
    )
    response = await generic_client.chat(request, model=dummy_model_name)

    # If the dummy worker receives images, it returns their dimensions in pixels
    # as token ids. Currently, the scheduler will crop and resize the images to
    # 384x384 pixels.
    #
    # el = 384
    assert response.message.content == "el el"


TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="


def test_multimodal_message_serialization() -> None:
    image_path = Path(__file__).parent / "tiny.png"
    message = Message(
        role=Role.User,
        content=[
            "Describe the following image.",
            Image.open(image_path),
        ]
    )
    assert message.to_json() == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the following image."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}}
        ]
    }


def test_multimodal_message_serialization_unknown_type() -> None:
    image_path = Path(__file__).parent / "tiny.png"
    message = Message(
        role=Role.User,
        content=[
            "Describe the following image.",
            Path(image_path),  # type: ignore
        ]
    )
    with pytest.raises(ValueError) as e:
        message.to_json()
    assert str(e.value) == "The item in the prompt is not valid. Try either a string or an Image."

def test_request_serialization_no_default_values() -> None:
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model="dummy-model",
    )
    assert request.to_json() == {
        "model": "dummy-model",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    }


def test_multi_turn_chat_serialization(sync_client: Client, dummy_model_name: str):
    """
    Test that TextMessage can be serialized when included in multi-turn chat history.

    We previously encountered an error in a multi-turn chat conversation.
    The returned TextMessage could not be made part of the chat history for the 
    next request as the method for serialization was missing. 
    This test should catch such conversion issues.
    """

    # First turn
    first_request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello")],
        model=dummy_model_name,
    )
    first_response = sync_client.chat(first_request, model=dummy_model_name)
    
    # Second turn - includes the TextMessage from first response in history
    messages_with_history: List[Union[Message, TextMessage]] = [
        Message(role=Role.User, content="Hello"),
        first_response.message,  # This TextMessage must be serializable
        Message(role=Role.User, content="Follow up question"),
    ]
    
    second_request = ChatRequest(
        messages=messages_with_history,
        model=dummy_model_name,
    )
    
    # This would fail if TextMessage.to_json() doesn't exist
    serialized = second_request.to_json()
    assert len(serialized["messages"]) == 3
    assert serialized["messages"][1]["role"] == "assistant"
