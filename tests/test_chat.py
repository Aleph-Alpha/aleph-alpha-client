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
    ToolCall,
    FunctionCall,
)
from aleph_alpha_client.structured_output import JSONSchema
from pydantic import BaseModel

from .conftest import GenericClient
from .test_steering import create_sample_steering_concept_creation_request


@pytest.mark.vcr
async def test_can_not_chat_with_all_models(async_client: AsyncClient, model_name: str):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=model_name,
        maximum_tokens=7,
    )

    with pytest.raises(ValueError):
        await async_client.chat(request, model=model_name)


@pytest.mark.vcr
def test_can_chat_with_chat_model(sync_client: Client, chat_model_name: str):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
    )

    response = sync_client.chat(request, model=chat_model_name)
    assert response.message.role == Role.Assistant
    assert response.message.content is not None
    assert isinstance(response.finish_reason, FinishReason)


@pytest.mark.vcr
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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


@pytest.mark.vcr
async def test_can_chat_with_tools(
    async_client: AsyncClient, tool_calling_model_name: str
):
    system_msg = Message(role=Role.System, content="You are a helpful assistant.")
    user_msg = Message(
        role=Role.User, content="What is the weather like in Paris today?"
    )
    request = ChatRequest(
        messages=[system_msg, user_msg],
        model=tool_calling_model_name,
        tools=TOOLS,
    )

    response = await async_client.chat(request, model=tool_calling_model_name)
    assert response.message.role == Role.Assistant
    assert response.message.content is not None
    assert response.message.tool_calls is not None
    calls = response.message.tool_calls
    assert len(calls) == 1
    assert calls[0].type == "function"
    assert calls[0].function.name == "get_weather"


@pytest.mark.vcr
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


@pytest.mark.vcr
async def test_can_chat_with_tools_streamed(
    async_client: AsyncClient, tool_calling_model_name: str
):
    system_msg = Message(role=Role.System, content="You are a helpful assistant.")
    user_msg = Message(
        role=Role.User, content="What is the weather like in Paris today?"
    )
    request = ChatRequest(
        messages=[system_msg, user_msg],
        model=tool_calling_model_name,
        tools=TOOLS,
    )

    stream = async_client.chat_with_streaming(request, model=tool_calling_model_name)
    stream_items = [stream_item async for stream_item in stream]

    v = stream_items[-2]
    tool_call_id = v.id if isinstance(v, ToolCall) else ""

    assert stream_items[-3:] == [
        ChatStreamChunk(
            content="</think>\n\n",
            role=None,
            tool_calls=None,
        ),
        ToolCall(
            id=tool_call_id,
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "Paris, France"}',
            ),
        ),
        FinishReason.ToolCalls,
    ]


def test_usage_response_is_parsed():
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


def test_finish_reason_response_is_parsed():
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


@pytest.mark.vcr
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

    print(stream_items)
    # Then
    assert all(isinstance(item, ChatStreamChunk) for item in stream_items[:-2])
    assert isinstance(stream_items[-2], FinishReason)
    assert isinstance(stream_items[-1], Usage)


@pytest.mark.vcr
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


def test_response_format_json_schema(
    sync_client: Client, structured_output_model_name: str
):
    example_json_schema = {
        "type": "object",
        "title": "Aquarium",
        "properties": {
            "nemo": {
                "type": "string",
                "title": "Nemo",
                "description": "Name of the fish",
            },
            "species": {
                "type": "string",
                "title": "Species",
                "description": "The species of the fish (e.g., Clownfish, Goldfish)",
            },
            "color": {
                "type": "string",
                "title": "Color",
                "description": "Primary color of the fish",
            },
            "size_cm": {
                "type": "number",
                "title": "Size in centimeters",
                "description": "Length of the fish in centimeters",
                "minimum": 0.1,
                "maximum": 100.0,
            },
        },
        "required": ["nemo", "species", "color", "size_cm"],
    }

    request = ChatRequest(
        messages=[
            Message(role=Role.System, content="You are a helpful assistant."),
            Message(
                role=Role.User,
                content=f"Give me JSON {example_json_schema}! Tell me about nemo",
            ),
        ],
        model=structured_output_model_name,
        response_format=JSONSchema(
            schema=example_json_schema,
            name="aquarium",
            description="Describe nemo",
            strict=True,
        ),
    )

    response = sync_client.chat(request, model=structured_output_model_name)
    json_response = json.loads(response.message.content)

    # Validate all required fields are present
    required_fields = ["nemo", "species", "color", "size_cm"]
    for field in required_fields:
        assert field in json_response.keys(), (
            f"Required field '{field}' is missing from response"
        )
    # Validate field types
    assert isinstance(json_response["nemo"], str), "Field 'nemo' should be a string"
    assert isinstance(json_response["species"], str), (
        "Field 'species' should be a string"
    )
    assert isinstance(json_response["color"], str), "Field 'color' should be a string"
    assert isinstance(json_response["size_cm"], (int, float)), (
        "Field 'size_cm' should be a number"
    )
    # Validate size constraints
    assert 0.1 <= json_response["size_cm"] <= 100.0, (
        "Field 'size_cm' should be between 0.1 and 100.0"
    )


def test_response_format_json_schema_pydantic(
    sync_client: Client, structured_output_model_name: str
):
    class Aquarium(BaseModel):
        nemo: str
        species: str
        color: str
        size_cm: float

    request = ChatRequest(
        messages=[
            Message(role=Role.System, content="You are a helpful assistant."),
            Message(
                role=Role.User,
                content=f"Give me JSON of type {Aquarium}! Tell me about nemo",
            ),
        ],
        model=structured_output_model_name,
        response_format=Aquarium,
    )

    response = sync_client.chat(request, model=structured_output_model_name)
    # Tests that it is valid json and loads
    json.loads(response.message.content)

    # Validate against desired fields
    Aquarium.model_validate_json(response.message.content)


@pytest.mark.vcr
@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
async def test_can_chat_with_images(
    generic_client: GenericClient, dummy_model_name: str
):
    image_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    image = Image.open(image_path)

    request = ChatRequest(
        messages=[
            Message(
                role=Role.User,
                content=[
                    "Describe the following image.",
                    image,
                ],
            )
        ],
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
        ],
    )
    assert message.to_json() == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the following image."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"},
            },
        ],
    }


def test_multimodal_message_serialization_unknown_type() -> None:
    image_path = Path(__file__).parent / "tiny.png"
    message = Message(
        role=Role.User,
        content=[
            "Describe the following image.",
            Path(image_path),  # type: ignore
        ],
    )
    with pytest.raises(ValueError) as e:
        message.to_json()
    assert (
        str(e.value)
        == "The item in the prompt is not valid. Try either a string or an Image."
    )


def test_request_serialization_no_default_values() -> None:
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model="dummy-model",
    )
    assert request.to_json() == {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }


def test_multi_turn_chat_serialization(sync_client: Client, chat_model_name: str):
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
        model=chat_model_name,
    )
    first_response = sync_client.chat(first_request, model=chat_model_name)
    # Second turn - includes the TextMessage from first response in history
    messages_with_history: List[Union[Message, TextMessage]] = [
        Message(role=Role.User, content="Hello"),
        first_response.message,  # This TextMessage must be serializable
        Message(role=Role.User, content="Follow up question"),
    ]

    second_request = ChatRequest(
        messages=messages_with_history,
        model=chat_model_name,
    )

    # This would fail if TextMessage.to_json() doesn't exist
    serialized = second_request.to_json()
    assert len(serialized["messages"]) == 3
    assert serialized["messages"][1]["role"] == "assistant"
