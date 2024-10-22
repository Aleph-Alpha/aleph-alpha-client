import pytest

from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.chat import ChatRequest, Message, Role
from tests.common import async_client, sync_client, model_name, chat_model_name


@pytest.mark.system_test
async def test_can_not_chat_with_all_models(async_client: AsyncClient, model_name: str):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=model_name,
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


async def test_can_chat_with_async_client(async_client: AsyncClient, chat_model_name: str):
    system_msg = Message(role=Role.System, content="You are a helpful assistant.")
    user_msg = Message(role=Role.User, content="Hello, how are you?")
    request = ChatRequest(
        messages=[system_msg, user_msg],
        model=chat_model_name,
    )

    response = await async_client.chat(request, model=chat_model_name)
    assert response.message.role == Role.Assistant
    assert response.message.content is not None


async def test_can_chat_with_streaming_support(async_client: AsyncClient, chat_model_name: str):
    request = ChatRequest(
        messages=[Message(role=Role.User, content="Hello, how are you?")],
        model=chat_model_name,
    )

    stream_items = [
        stream_item async for stream_item in async_client.chat_with_streaming(request, model=chat_model_name)
    ]

    assert stream_items[0].role is not None
    assert all(item.content is not None for item in stream_items[1:])
