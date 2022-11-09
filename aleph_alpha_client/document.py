import base64
from typing import Any, Dict, List, Optional, Sequence, Union

from aleph_alpha_client.image import ImagePrompt
from aleph_alpha_client.prompt import _to_prompt_item


class Document:
    """
    A document that can be either a docx document or text/image prompts.
    """

    def __init__(
        self,
        docx: Optional[str] = None,
        prompt: Optional[Sequence[Union[str, ImagePrompt]]] = None,
        text: Optional[str] = None,
    ):
        # We use a base_64 representation for docx documents, because we want to embed the file
        # into a prompt send in JSON.
        self.docx = docx
        self.prompt = prompt
        self.text = text

    @classmethod
    def from_docx_bytes(cls, bytes: bytes):
        """
        Pass a docx file in bytes and prepare it to be used as a document
        """
        docx_base64 = base64.b64encode(bytes).decode()
        return cls(docx=docx_base64)

    @classmethod
    def from_docx_file(cls, path: str):
        """
        Load a docx file from disk and prepare it to be used as a document

        Examples
            >>> docx_file = "./tests/sample.docx"
            >>> document = Document.from_docx_file(docx_file)
        """
        with open(path, "rb") as f:
            docx_bytes = f.read()
        return cls.from_docx_bytes(docx_bytes)

    @classmethod
    def from_prompt(cls, prompt: Sequence[Union[str, ImagePrompt]]):
        """
        Pass a prompt that can contain multiple strings and Image prompts and prepare it to be used as a document
        """
        return cls(prompt=prompt)

    @classmethod
    def from_text(cls, text: str):
        """
        Pass a single text and prepare it to be used as a document

        Example:
            >>> prompt = "This is an example."
            >>> document = Document.from_text(prompt)
        """
        return cls(text=text)

    def _to_serializable_document(self) -> Dict[str, Any]:
        """
        A dict if serialized to JSON is suitable as a document element
        """
        if self.docx is not None:
            # Serialize docx to Document JSON format
            return {
                "docx": self.docx,
            }
        elif self.prompt is not None:
            # Serialize prompt to Document JSON format
            prompt_data = [_to_prompt_item(prompt_item) for prompt_item in self.prompt]
            return {"prompt": prompt_data}
        elif self.text is not None:
            return {
                "text": self.text,
            }
        else:
            raise NotImplementedError("unsupported document type")
