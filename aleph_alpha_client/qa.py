from typing import Any, Dict, Mapping, NamedTuple, Sequence

from aleph_alpha_client.document import Document


class QaRequest(NamedTuple):
    """
    Answers a question about a prompt.

    Parameters:
        query (str, required):
            The question to be answered about the documents by the model.

        documents (List[Document], required):
            A list of documents. This can be either docx documents or text/image prompts.

        maximum_tokens (int, default 64):
            The maximum number of tokens to be generated. Completion will terminate after the maximum number of tokens is reached.

            Increase this value to generate longer texts. A text is split into tokens. Usually there are more tokens than words.

            The maximum supported number of tokens depends on the model (for luminous-base, it may not exceed 2048 tokens).
            The prompt's tokens plus the maximum_tokens request must not exceed this number.

        max_chunk_size (int, default 175):
            Long documents will be split into chunks if they exceed max_chunk_size.
            The splitting will be done along the following boundaries until all chunks are shorter than max_chunk_size or all splitting criteria have been exhausted.
            The splitting boundaries are, in the given order:
            1. Split first by double newline
            (assumed to mark the boundary between 2 paragraphs).
            2. Split paragraphs that are still too long by their median sentence as long as we can still find multiple sentences in the paragraph.
            3. Split each remaining chunk of a paragraph or sentence further along white spaces until each chunk is smaller than max_chunk_size or until no whitespace can be found anymore.

        disable_optimizations  (bool, default False)
            We continually research optimal ways to work with our models. By default, we apply these optimizations to both your query, documents, and answers for you.
            Our goal is to improve your results while using our API.
            But you can always pass `disable_optimizations: true` and we will leave your query, documents, and answers untouched.

        max_answers (int, default 0):
            The maximum number of answers.

        min_score (float, default 0.0):
            The lower limit of minimum score for every answer.

        Examples:
            >>> request = QaRequest(
                    query = "What is a computer program?",
                    documents = [document]
                )
    """

    query: str
    documents: Sequence[Document]
    maximum_tokens: int = 64
    max_chunk_size: int = 175
    disable_optimizations: bool = False
    max_answers: int = 0
    min_score: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        payload = self._asdict()
        payload["documents"] = [
            document._to_serializable_document() for document in self.documents
        ]
        return payload


class QaAnswer(NamedTuple):
    answer: str
    score: float
    evidence: str


class QaResponse(NamedTuple):
    model_version: str
    answers: Sequence[QaAnswer]

    @staticmethod
    def from_json(json: Mapping[str, Any]) -> "QaResponse":
        return QaResponse(
            model_version=json["model_version"],
            answers=[
                QaAnswer(item["answer"], item["score"], item["evidence"])
                for item in json["answers"]
            ],
        )
