from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class SteeringPairedExample:
    negative: str
    positive: str

    def to_json(self) -> Mapping[str, Any]:
        return self._asdict()

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SteeringConceptCreationRequest:
    """
    Creates a new steering concept that can be used in completion requests.

    Parameters:
        examples
            A list of SteeringPairedExample objects.

    Examples:
        >>> examples = [
        >>>     SteeringPairedExample(
        >>>         negative="I appreciate your valuable feedback on this matter.",
        >>>         positive="Thanks for the real talk, fam.",
        >>>     ),
        >>>     SteeringPairedExample(
        >>>         negative="The financial projections indicate significant growth potential.",
        >>>         positive="Yo, these numbers are looking mad stacked!",
        >>>     ),
        >>> ]
        >>> SteeringConceptCreationRequest(examples=examples)
    """

    examples: list[SteeringPairedExample]

    def to_json(self) -> Mapping[str, Any]:
        return {
            **self._asdict(),
            "examples": [e.to_json() for e in self.examples],
        }

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SteeringConceptCreationResponse:
    id: str

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "SteeringConceptCreationResponse":
        return SteeringConceptCreationResponse(
            id=json["id"],
        )
