from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from aleph_alpha_client.prompt import Prompt


@dataclass(frozen=True)
class CompletionRequest:
    """
    Describes a completion request

    Parameters:
        prompt:
            The text or image prompt to be completed.
            Unconditional completion can be started with an empty string (default).
            The prompt may contain a zero shot or few shot task.

        maximum_tokens (int, optional, default None):
            The maximum number of tokens to be generated.
            Completion will terminate after the maximum number of tokens is reached. Increase this value to generate longer texts.
            A text is split into tokens. Usually there are more tokens than words.
            The maximum supported number of tokens depends on the model (for luminous-base, it may not exceed 2048 tokens).
            The prompt's tokens plus the maximum_tokens request must not exceed this number. If set to None, the model will stop
            generating tokens either if it outputs a sequence specified in `stop_sequences` or if it reaches its technical limit.
            For most models, this means that the sum of input and output tokens is equal to its context window.

        temperature (float, optional, default 0.0)
            A higher sampling temperature encourages the model to produce less probable outputs ("be more creative"). Values are expected in a range from 0.0 to 1.0. Try high values (e.g. 0.9) for a more "creative" response and the default 0.0 for a well defined and repeatable answer.

            It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

        top_k (int, optional, default 0)
            Introduces random sampling from generated tokens by randomly selecting the next token from the k most likely options. A value larger than 1 encourages the model to be more creative. Set to 0 if repeatable output is to be produced.
            It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

        top_p (float, optional, default 0.0)
            Introduces random sampling for generated tokens by randomly selecting the next token from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p. Set to 0.0 if repeatable output is to be produced.
            It is recommended to use either temperature, top_k or top_p and not all at the same time. If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed first. Then top_k is applied. Top_p follows last.

        presence_penalty (float, optional, default 0.0)
            The presence penalty reduces the likelihood of generating tokens that are already present in the
            generated text (`repetition_penalties_include_completion=true`) respectively the prompt (`repetition_penalties_include_prompt=true`).
            Presence penalty is independent of the number of occurences. Increase the value to produce text that is not repeating the input.

        frequency_penalty (float, optional, default 0.0)
            The frequency penalty reduces the likelihood of generating tokens that are already present in the
            generated text (`repetition_penalties_include_completion=true`) respectively the prompt (`repetition_penalties_include_prompt=true`).
            Frequency penalty is dependent on the number of occurences of a token.

        repetition_penalties_include_prompt (bool, optional, default False)
            Flag deciding whether presence penalty or frequency penalty are updated from the prompt

        use_multiplicative_presence_penalty (bool, optional, default True)
            Flag deciding whether presence penalty is applied multiplicatively (True) or additively (False). This changes the formula stated for presence and frequency penalty.

        penalty_bias (string, optional)
            If set, all tokens in this text will be used in addition to the already penalized tokens for repetition penalties.
            These consist of the already generated completion tokens if ``repetition_penalties_include_completion`` is set to ``true``
            and the prompt tokens, if ``repetition_penalties_include_prompt`` is set to ``true``,

            *Potential use case for a chatbot-based completion:*

            Instead of using ``repetition_penalties_include_prompt``, construct a new string with only the chatbot's responses included. You would leave out any tokens you use for stop sequences (i.e. ``\\nChatbot:``), and all user messages.

            With this bias, if you turn up the repetition penalties, you can avoid having your chatbot repeat itself, but not penalize the chatbot from mirroring language provided by the user.

        penalty_exceptions (List(str), optional)
            List of strings that may be generated without penalty, regardless of other penalty settings.

            This is particularly useful for any completion that uses a structured few-shot prompt. For example, if you have a prompt such as:

            ::

                I want to travel to a location, where I can enjoy both beaches and mountains.

                - Lake Garda, Italy. This large Italian lake in the southern alps features gravel beaches and mountainside hiking trails.
                - Mallorca, Spain. This island is famous for its sandy beaches, turquoise water and hilly landscape.
                - Lake Tahoe, California. This famous lake in the Sierra Nevada mountains offers an amazing variety of outdoor activities.
                -

            You could set ``penalty_exceptions`` to ``["\\n-"]`` to not penalize the generation of a new list item, but still increase other penalty settings to encourage the generation of new list items without repeating itself.

            By default, we will also include any ``stop_sequences`` you have set, since completion performance can be degraded if expected stop sequences are penalized. You can disable this behavior by settings ``penalty_exceptions_include_stop_sequences`` to ``false``.

        penalty_exceptions_include_stop_sequences (bool, optional, default true)
            By default, we include any ``stop_sequences`` in ``penalty_exceptions``, to not penalize the presence of stop sequences that are present in few-shot prompts to provide structure to your completions.

            You can set this to ``false`` if you do not want this behavior.

            See the description of ``penalty_exceptions`` above for more information on what ``penalty_exceptions`` are used for.

        best_of (int, optional, default None)
            Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
            When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.

        n (int, optional, default 1)
            How many completions to generate for each prompt.

        logit_bias (dict mapping token ids to score, optional, default None)
            The logit bias allows to influence the likelihood of generating tokens. A dictionary mapping token ids (int) to a bias (float) can be provided. Such bias is added to the logits as generated by the model.

        log_probs (int, optional, default None)
            Number of top log probabilities to be returned for each generated token. Log probabilities may be used in downstream tasks or to assess the model's certainty when producing tokens.

            If set to 0, you will always get the log probability of the sampled token. 1 or more will return the argmax token(s) plus the sampled one, if not already included.

        stop_sequences (List(str), optional, default None)
            List of strings which will stop generation if they're generated. Stop sequences may be helpful in structured texts.

            Example: In a question answering scenario a text may consist of lines starting with either "Question: " or "Answer: " (alternating). After producing an answer, the model will be likely to generate "Question: ". "Question: " may therfore be used as stop sequence in order not to have the model generate more questions but rather restrict text generation to the answers.

        tokens (bool, optional, default False)
            return tokens of completion

        disable_optimizations  (bool, optional, default False)
            We continually research optimal ways to work with our models. By default, we apply these optimizations to both your prompt and  completion for you.

            Our goal is to improve your results while using our API. But you can always pass disable_optimizations: true and we will leave your prompt and completion untouched.

        minimum_tokens (int, default 0)
            Generate at least this number of tokens before an end-of-text token is generated.

        echo (bool, default False)
            Echo the prompt in the completion. This may be especially helpful when log_probs is set to return logprobs for the prompt.

        use_multiplicative_frequency_penalty (bool, default False)
            Flag deciding whether frequency penalty is applied multiplicatively (True) or additively (False).

        sequence_penalty (float, default 0.0)
            Increasing the sequence penalty reduces the likelihood of reproducing token sequences that already appear in the prompt
            (if repetition_penalties_include_prompt is True) and prior completion (if repetition_penalties_include_completion is True).

        sequence_penalty_min_length (int, default 2)
            Minimal number of tokens to be considered as sequence. Must be greater or eqaul 2.

        use_multiplicative_sequence_penalty (bool, default False)
            Flag deciding whether sequence penalty is applied multiplicatively (True) or additively (False).

        completion_bias_inclusion (List[str], default [])
            Bias the completion to only generate options within this list;
            all other tokens are disregarded at sampling

            Note that strings in the inclusion list must not be prefixes
            of strings in the exclusion list and vice versa

        completion_bias_inclusion_first_token_only (bool, default False)
            Only consider the first token for the completion_bias_inclusion

        completion_bias_exclusion (List[str], default [])
            Bias the completion to NOT generate options within this list;
            all other tokens are unaffected in sampling

            Note that strings in the inclusion list must not be prefixes
            of strings in the exclusion list and vice versa

        completion_bias_exclusion_first_token_only (bool, default False)
            Only consider the first token for the completion_bias_exclusion

        contextual_control_threshold (float, default None)
            If set to None, attention control parameters only apply to those tokens that have
            explicitly been set in the request.
            If set to a non-None value, we apply the control parameters to similar tokens as well.
            Controls that have been applied to one token will then be applied to all other tokens
            that have at least the similarity score defined by this parameter.
            The similarity score is the cosine similarity of token embeddings.

        control_log_additive (bool, default True)
            True: apply control by adding the log(control_factor) to attention scores.
            False: apply control by (attention_scores - - attention_scores.min(-1)) * control_factor

        repetition_penalties_include_completion (bool, optional, default True)
            Flag deciding whether presence penalty or frequency penalty are updated from the completion

        raw_completion (bool, default False)
            Setting this parameter to true forces the raw completion of the model to be returned.
            For some models, we may optimize the completion that was generated by the model and
            return the optimized completion in the completion field of the CompletionResponse.
            The raw completion, if returned, will contain the un-optimized completion.

    Examples:
        >>> prompt = Prompt.from_text("Provide a short description of AI:")
        >>> request = CompletionRequest(prompt=prompt, maximum_tokens=20)
    """

    prompt: Prompt
    maximum_tokens: Optional[int] = None
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalties_include_prompt: bool = False
    use_multiplicative_presence_penalty: bool = False
    penalty_bias: Optional[str] = None
    penalty_exceptions: Optional[List[str]] = None
    penalty_exceptions_include_stop_sequences: Optional[bool] = None
    best_of: Optional[int] = None
    n: int = 1
    logit_bias: Optional[Dict[int, float]] = None
    log_probs: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    tokens: bool = False
    disable_optimizations: bool = False
    minimum_tokens: int = 0
    echo: bool = False
    use_multiplicative_frequency_penalty: bool = False
    sequence_penalty: float = 0.0
    sequence_penalty_min_length: int = 2
    use_multiplicative_sequence_penalty: bool = False
    completion_bias_inclusion: Optional[Sequence[str]] = None
    completion_bias_inclusion_first_token_only: bool = False
    completion_bias_exclusion: Optional[Sequence[str]] = None
    completion_bias_exclusion_first_token_only: bool = False
    contextual_control_threshold: Optional[float] = None
    control_log_additive: Optional[bool] = True
    repetition_penalties_include_completion: bool = True
    raw_completion: bool = False

    def to_json(self) -> Mapping[str, Any]:
        payload = {k: v for k, v in self._asdict().items() if v is not None}
        payload["prompt"] = self.prompt.to_json()
        return payload

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletionResult:
    log_probs: Optional[Sequence[Mapping[str, Optional[float]]]] = None
    completion: Optional[str] = None
    completion_tokens: Optional[Sequence[str]] = None
    finish_reason: Optional[str] = None
    raw_completion: Optional[str] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "CompletionResult":
        return CompletionResult(
            log_probs=json.get("log_probs"),
            completion=json.get("completion"),
            completion_tokens=json.get("completion_tokens"),
            finish_reason=json.get("finish_reason"),
            raw_completion=json.get("raw_completion"),
        )

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletionResponse:
    """
    Describes a completion response

    Parameters:
        model_version:
            Model name and version (if any) of the used model for inference.
        completions:
            List of completions; may contain only one entry if no more are requested (see parameter n).
        num_tokens_prompt_total:
            Number of tokens combined across all completion tasks.
            In particular, if you set best_of or n to a number larger than 1 then we report the
            combined prompt token count for all best_of or n tasks.
        num_tokens_generated:
            Number of tokens combined across all completion tasks.
            If multiple completions are returned or best_of is set to a value greater than 1 then
            this value contains the combined generated token count.
        optimized_prompt:
            Describes prompt after optimizations. This field is only returned if the flag
            `disable_optimizations` flag is not set and the prompt has actually changed.
    """

    model_version: str
    completions: Sequence[CompletionResult]
    num_tokens_prompt_total: int
    num_tokens_generated: int
    optimized_prompt: Optional[Prompt] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "CompletionResponse":
        optimized_prompt_json = json.get("optimized_prompt")
        return CompletionResponse(
            model_version=json["model_version"],
            completions=[
                CompletionResult.from_json(item) for item in json["completions"]
            ],
            num_tokens_prompt_total=json["num_tokens_prompt_total"],
            num_tokens_generated=json["num_tokens_generated"],
            optimized_prompt=(
                Prompt.from_json(optimized_prompt_json)
                if optimized_prompt_json
                else None
            ),
        )

    def to_json(self) -> Mapping[str, Any]:
        return {
            **self._asdict(),
            "completions": [completion._asdict() for completion in self.completions],
        }

    def _asdict(self) -> Mapping[str, Any]:
        return asdict(self)


CompletionResponseStreamItem = Union[
    "StreamChunk", "StreamSummary", "CompletionSummary"
]


def stream_item_from_json(json: Dict[str, Any]) -> CompletionResponseStreamItem:
    if json["type"] == "stream_chunk":
        return StreamChunk.from_json(json)
    elif json["type"] == "stream_summary":
        return StreamSummary.from_json(json)
    elif json["type"] == "completion_summary":
        return CompletionSummary.from_json(json)
    else:
        raise ValueError(f"Unknown stream item type: {json['type']}")


@dataclass(frozen=True)
class StreamChunk:
    """
    Describes a chunk of a completion stream

    Parameters:
        index:
            The index of the stream that this chunk belongs to.
            This is relevant if multiple completion streams are requested (see parameter n).
        log_probs:
            The log probabilities of the generated tokens.
        completion:
            The generated tokens formatted as single a string.
        raw_completion:
            The generated tokens including special tokens formatted as single a string.
        completion_tokens:
            The generated tokens as a list of strings.
    """

    index: int
    log_probs: Optional[Sequence[Mapping[str, Optional[float]]]]
    completion: str
    raw_completion: Optional[str]
    completion_tokens: Optional[Sequence[str]]

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "StreamChunk":
        return StreamChunk(
            index=json["index"],
            log_probs=json.get("log_probs"),
            completion=json["completion"],
            raw_completion=json.get("raw_completion"),
            completion_tokens=json.get("completion_tokens"),
        )

    def to_json(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StreamSummary:
    """
    Denotes the end of a completion stream

    Parameters:
        index:
            The index of the stream that is being terminated.
            This is relevant if multiple completion streams are requested (see parameter n).
        model_version:
            Model name and version (if any) of the used model for inference.
        finish_reason:
            The reason why the model stopped generating new tokens.
    """

    index: int
    model_version: str
    finish_reason: str

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "StreamSummary":
        return StreamSummary(
            index=json["index"],
            model_version=json["model_version"],
            finish_reason=json["finish_reason"],
        )

    def to_json(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletionSummary:
    """
    Denotes the end of all completion streams

    Parameters:
        optimized_prompt:
            Describes prompt after optimizations. This field is only returned if the flag
            `disable_optimizations` flag is not set and the prompt has actually changed.
        num_tokens_prompt_total:
            Number of tokens combined across all completion tasks.
            In particular, if you set best_of or n to a number larger than 1 then we report the
            combined prompt token count for all best_of or n tasks.
        num_tokens_generated:
            Number of tokens combined across all completion tasks.
            If multiple completions are returned or best_of is set to a value greater than 1 then
            this value contains the combined generated token count.
    """

    optimized_prompt: Optional[Prompt]
    num_tokens_prompt_total: int
    num_tokens_generated: int

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "CompletionSummary":
        optimized_prompt_json = json.get("optimized_prompt")
        return CompletionSummary(
            optimized_prompt=(
                Prompt.from_json(optimized_prompt_json)
                if optimized_prompt_json
                else None
            ),
            num_tokens_prompt_total=json["num_tokens_prompt_total"],
            num_tokens_generated=json["num_tokens_generated"],
        )
