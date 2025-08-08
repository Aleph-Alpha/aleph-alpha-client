from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Union


@dataclass(frozen=True)
class JSONSchema:
    """
    JSON schema that structured output must adhere to.

    Parameters:
        schema:
            JSON schema that structured output must adhere to.
        name:
            Name of the schema.
        description:
            Description of the schema.
        strict:
            Whether the schema should be strictly enforced.

    Examples:
        >>> schema = JSONSchema(
        >>>     schema={
        >>>         'type': 'object',
        >>>         'title': 'Aquarium',
        >>>         'properties': {
        >>>             'nemo': {
        >>>                 'type': 'string',
        >>>                 'title': 'Nemo'
        >>>             }
        >>>         },
        >>>         'required': ['nemo']
        >>>     },
        >>>     name="aquarium",
        >>>     description="Describe nemo",
        >>>     strict=True
        >>> )
    """

    schema: Mapping[str, Any]
    name: str
    description: Optional[str] = None
    strict: Optional[bool] = False

    def to_json(self) -> Mapping[str, Any]:
        return {"type": "json_schema", "json_schema": asdict(self)}


ResponseFormat = Union[JSONSchema]
