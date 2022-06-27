from aleph_alpha_client.aleph_alpha_client import AlephAlphaClient
from aleph_alpha_client.completion import CompletionRequest, CompletionResponse


class AlephAlphaModel:

    def __init__(self, client: AlephAlphaClient, model_name: str, hosting: str = "cloud") -> None:
        self.client = client
        self.model_name = model_name
        self.hosting = hosting

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response_json = self.client.complete(model = self.model_name, hosting=self.hosting, **request._asdict())
        return CompletionResponse.from_json(response_json)