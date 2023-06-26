from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence

from aleph_alpha_client.document import Document


class QaRequest(NamedTuple):
    """
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

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        payload["documents"] = [
            document._to_serializable_document() for document in self.documents
        ]
        if self.max_answers is None:
            del payload["max_answers"]
        return payload


class QaAnswer(NamedTuple):
    answer: str
    score: float
    evidence: str


class QaResponse(NamedTuple):
    answers: Sequence[QaAnswer]

    @staticmethod
    def from_json(json: Mapping[str, Any]) -> "QaResponse":
        return QaResponse(
            answers=[
                QaAnswer(item["answer"], item["score"], item["evidence"])
                for item in json["answers"]
            ],
        )
