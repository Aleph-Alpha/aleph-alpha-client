import pytest
from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import (
    ControlTokenOverlap,
    Image,
    Prompt,
    Text,
    TextControl,
    Tokens,
)

from tests.common import (
    sync_client,
    async_client,
    model_name,
    prompt_image,
)


# AsyncClient


@pytest.mark.system_test
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


# Client


@pytest.mark.system_test
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


@pytest.mark.system_test
def test_complete_with_token_ids(sync_client: Client, model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_tokens([49222, 2998]),  # Hello world
        maximum_tokens=32,
    )

    response = sync_client.complete(request, model=model_name)

    assert len(response.completions) == 1
    assert response.model_version is not None


@pytest.mark.system_test
def test_complete_with_optimized_prompt(
    sync_client: Client, model_name: str, prompt_image: Image
):
    prompt_text = " Hello World! "
    prompt_tokens = Tokens.from_token_ids([1, 2])
    request = CompletionRequest(
        prompt=Prompt([Text.from_text(prompt_text), prompt_image, prompt_tokens]),
        maximum_tokens=5,
    )

    response = sync_client.complete(request, model=model_name)

    assert response.optimized_prompt
    assert response.optimized_prompt.items[0] == Text.from_text(prompt_text.strip())
    assert response.optimized_prompt.items[2] == prompt_tokens
    assert isinstance(response.optimized_prompt.items[1], Image)


@pytest.mark.system_test
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
