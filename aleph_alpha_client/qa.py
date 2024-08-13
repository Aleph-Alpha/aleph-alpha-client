from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from aleph_alpha_client.document import Document


@dataclass(frozen=True)
class QaRequest:
    """DEPRECATED: `QaRequest` is deprecated and will be removed in the future. New
    methods of processing Q&A tasks will be provided before this is removed.

    Answers a question about a prompt.

    Parameters:
        query (str, required):
            The question to be answered about the documents by the model.

        documents (List[Document], required):
            A list of documents. This can be either docx documents or text/image prompts.

        max_answers (int, default None):
            The maximum number of answers.

        Examples:
            >>> request = QaRequest(
                    query = "What is a computer program?",
                    documents = [document]
                )
    """

    query: str
    documents: Sequence[Document]
    max_answers: Optional[int] = None

    def to_json(self) -> Mapping[str, Any]:
        payload = {
            **self._asdict(),
            "documents": [
                document._to_serializable_document() for document in self.documents
            ],
        }
        if self.max_answers is None:
            del payload["max_answers"]
        return payload

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaAnswer:
    """DEPRECATED: `QaAnswer` is deprecated and will be removed in the future. New
    methods of processing Q&A tasks will be provided before this is removed.
    """

    answer: str
    score: float
    evidence: str


@dataclass(frozen=True)
class QaResponse:
    """DEPRECATED: `QaResponse` is deprecated and will be removed in the future. New
    methods of processing Q&A tasks will be provided before this is removed.
    """

    answers: Sequence[QaAnswer]

    @staticmethod
    def from_json(json: Mapping[str, Any]) -> "QaResponse":
        return QaResponse(
            answers=[
                QaAnswer(item["answer"], item["score"], item["evidence"])
                for item in json["answers"]
            ],
        )
