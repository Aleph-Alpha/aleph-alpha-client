import os
import pytest
from aleph_alpha_client.aleph_alpha_client import AsyncClient
from aleph_alpha_client.completion import CompletionRequest
from aleph_alpha_client.prompt import Prompt
from aleph_alpha_client.detokenization import DetokenizationRequest
from aleph_alpha_client.tokenization import TokenizationRequest
from aleph_alpha_client.embedding import (
    EmbeddingRequest,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
from aleph_alpha_client.summarization import SummarizationRequest
from aleph_alpha_client.evaluation import EvaluationRequest
from aleph_alpha_client.qa import QaRequest
from aleph_alpha_client.explanation import ExplanationRequest
from aleph_alpha_client.document import Document
from .common import (
    async_client,
    model_name,
    checkpoint_name,
    qa_checkpoint_name,
    summarization_checkpoint_name,
)


@pytest.mark.system_test
async def test_can_use_async_client_without_context_manager(model_name: str):
    request = CompletionRequest(
        prompt=Prompt.from_text(""),
        maximum_tokens=7,
    )
    token = os.environ["TEST_TOKEN"]
    client = AsyncClient(token, host=os.environ["TEST_API_URL"])
    try:
        _ = await client.complete(request, model=model_name)
    finally:
        await client.close()
