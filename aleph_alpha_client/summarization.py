from dataclasses import asdict, dataclass
from typing import Any, Mapping

from aleph_alpha_client.document import Document


@dataclass(frozen=True)
class SummarizationRequest:
    """DEPRECATED: `SummarizationRequest` is deprecated and will be removed in the future. New
    methods of processing Summarization tasks will be provided before this is removed.

    Summarizes a document.

    Parameters:
        document (Document, required):
            A single document. This can be one of the following formats:

            - Docx: A base64 encoded Docx file
            - Text: A string of text
            - Prompt: A multimodal prompt, as is used in our other tasks like Completion

            Documents of types Docx and Text are usually preferred, and will have optimizations (such as chunking) applied to work better with the respective task that is being run.

            Prompt documents are assumed to be used for advanced use cases, and will be left as-is.

        disable_optimizations  (bool, default False)
            We continually research optimal ways to work with our models. By default, we apply these optimizations to both your query, documents, and answers for you.
            Our goal is to improve your results while using our API.
            But you can always pass `disable_optimizations: true` and we will leave your document and summary untouched.

    Examples:
        >>> docx_file = "./tests/sample.docx"
        >>> document = Document.from_docx_file(docx_file)
        >>> request = SummarizationRequest(document)
    """

    document: Document
    disable_optimizations: bool = False

    def to_json(self) -> Mapping[str, Any]:
        return {**self._asdict(), "document": self.document._to_serializable_document()}

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SummarizationResponse:
    """DEPRECATED: `SummarizationResponse` is deprecated and will be removed in the future. New
    methods of processing Summarization tasks will be provided before this is removed.
    """

    summary: str

    @classmethod
    def from_json(cls, json: Mapping[str, Any]) -> "SummarizationResponse":
        return cls(summary=json["summary"])
