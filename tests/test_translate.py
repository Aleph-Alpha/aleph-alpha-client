import asyncio
import pytest
from aleph_alpha_client import AsyncClient, Client
from aleph_alpha_client.translation import TranslationRequest, TranslationResponse
from tests.conftest import GenericClient


LONG_ENGLISH_TEXT = 'Question: What are some benefits of surface micro-machining?\nAnswer the question on the basis of the text. If there is no answer within the text, respond "no answer in text".\n\nSurface micromachining\n\nSurface micromachining builds microstructures by deposition and etching structural layers over a substrate.[1] This is different from Bulk micromachining, in which a silicon substrate wafer is selectively etched to produce structures.\n\nLayers\n\nGenerally, polysilicon is used as one of the substrate layers while silicon dioxide is used as a sacrificial layer. The sacrificial layer is removed or etched out to create any necessary void in the thickness direction. Added layers tend to vary in size from 2-5 micrometres. The main advantage of this machining process is the ability to build electronic and mechanical components (functions) on the same substrate. Surface micro-machined components are smaller compared to their bulk micro-machined counterparts.\n\nAs the structures are built on top of the substrate and not inside it, the substrate\'s properties are not as important as in bulk micro-machining. Expensive silicon wafers can be replaced by cheaper substrates, such as glass or plastic. The size of the substrates may be larger than a silicon wafer, and surface micro-machining is used to produce thin-film transistors on large area glass substrates for flat panel displays. This technology can also be used for the manufacture of thin film solar cells, which can be deposited on glass, polyethylene terepthalate substrates or other non-rigid materials.\n\nFabrication process\n\nMicro-machining starts with a silicon wafer or other substrate upon which new layers are grown. These layers are selectively etched by photo-lithography; either a wet etch involving an acid, or a dry etch involving an ionized gas (or plasma). Dry etching can combine chemical etching with physical etching or ion bombardment. Surface micro-machining involves as many layers as are needed with a different mask (producing a different pattern) on each layer. Modern integrated circuit fabrication uses this technique and can use as many as 100 layers. Micro-machining is a younger technology and usually uses no more than 5 or 6 layers. Surface micro-machining uses developed technology (although sometimes not enough for demanding applications) which is easily repeatable for volume production.\n\n### Response:'

@pytest.mark.parametrize(
    "generic_client", ["sync_client", "async_client"], indirect=True
)
async def test_can_translate_single_request(
    generic_client: GenericClient, translation_model_name: str
):
    request = TranslationRequest(
        model=translation_model_name,
        source=LONG_ENGLISH_TEXT,
        target_language="de",
    )

    response = await generic_client.translate(request)
    assert isinstance(response, TranslationResponse)
    assert response.translation is not None
    assert response.score >= 0.0
    assert response.score <= 1.0
    assert response.num_tokens_prompt_total > 0
    assert response.num_tokens_generated == 0  # Translation models don't expose tokens

    if response.segments is not None:
        for segment in response.segments:
            assert segment.source is not None
            assert segment.translation is not None
            assert segment.score >= 0.0
            assert segment.score <= 1.0


async def test_can_translate_multiple_requests_concurrently_with_async_client(
    async_client: AsyncClient, translation_model_name: str
):
    # Create multiple translation requests for different languages
    languages = ["de", "fr", "es", "it", "nl"]
    requests = [
        TranslationRequest(
            model=translation_model_name,
            source=LONG_ENGLISH_TEXT,
            target_language=lang,
        )
        for lang in languages
    ]

    responses = await asyncio.gather(
        *[async_client.translate(request) for request in requests]
    )

    assert len(responses) == len(languages)
    for response in responses:
        assert isinstance(response, TranslationResponse)
        assert response.translation is not None
        assert response.translation != LONG_ENGLISH_TEXT
        assert response.score >= 0.0
        assert response.score <= 1.0
        assert response.num_tokens_prompt_total > 0
        assert response.num_tokens_generated == 0

        if response.segments is not None:
            for segment in response.segments:
                assert segment.source is not None
                assert segment.translation is not None
                assert segment.score >= 0.0
                assert segment.score <= 1.0

