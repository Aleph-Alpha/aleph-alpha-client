from dataclasses import asdict, dataclass
from typing import Any, Mapping, Union

@dataclass(frozen=True)
class JSONSchema:
    """
    JSON schema that structured output must adhere to.

    Parameters:
        json_schema:
            JSON schema that structured output must adhere to.

    Examples:
        >>> schema = [
        >>>     JSONSchema(
        >>>         json_schema={'properties': {'bar': {'type': 'integer'}, 'type': 'object'}}
        >>>     )
    """

    json_schema: Mapping[str, Any]

    def to_json(self) -> Mapping[str, Any]:
        return {"type": "json_schema", **asdict(self)}

ResponseFormat = Union[JSONSchema]
