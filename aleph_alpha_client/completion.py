from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

from aleph_alpha_client.prompt import Prompt


class CompletionRequest(NamedTuple):
    """
    Describes a completion request

    Parameters:
        prompt:
            The text or image prompt to be completed.
            Unconditional completion can be started with an empty string (default).
            The prompt may contain a zero shot or few shot task.

        maximum_tokens (int, optional, default 64):
            The maximum number of tokens to be generated.
            Completion will terminate after the maximum number of tokens is reached. Increase this value to generate longer texts.
            A text is split into tokens. Usually there are more tokens than words.
            The maximum supported number of tokens depends on the model (for luminous-base, it may not exceed 2048 tokens).
            The prompt's tokens plus the maximum_tokens request must not exceed this number.

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
            The presence penalty reduces the likelihood of generating tokens that are already present in the text.
            Presence penalty is independent of the number of occurences. Increase the value to produce text that is not repeating the input.

        frequency_penalty (float, optional, default 0.0)
            The frequency penalty reduces the likelihood of generating tokens that are already present in the text.
            Frequency penalty is dependent on the number of occurences of a token.

        repetition_penalties_include_prompt (bool, optional, default False)
            Flag deciding whether presence penalty or frequency penalty are applied to the prompt and completion (True) or only the completion (False)

        use_multiplicative_presence_penalty (bool, optional, default True)
            Flag deciding whether presence penalty is applied multiplicatively (True) or additively (False). This changes the formula stated for presence and frequency penalty.

        penalty_bias (string, optional)
            If set, all tokens in this text will be used in addition to the already penalized tokens for repetition penalties. These consist of the already generated completion tokens and the prompt tokens, if ``repetition_penalties_include_prompt`` is set to ``true``\,

            *Potential use case for a chatbot-based completion:*

            Instead of using ``repetition_penalties_include_prompt``\, construct a new string with only the chatbot's reponses included. You would leave out any tokens you use for stop sequences (i.e. ``\\nChatbot:``\), and all user messages.

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

            By default, we will also include any ``stop_sequences`` you have set, since completion performance can be degraded if expected stop sequences are penalized. You can disable this behavior by settings ``penalty_exceptions_include_stop_sequences`` to ``false``\.

        penalty_exceptions_include_stop_sequences (bool, optional, default true)
            By default, we include any ``stop_sequences`` in ``penalty_exceptions``\, to not penalize the presence of stop sequences that are present in few-shot prompts to provide structure to your completions.

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
            (if repetition_penalties_include_prompt is True) and prior completion.

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

    Examples:
        >>> prompt = Prompt.from_text("Provide a short description of AI:")
        >>> request = CompletionRequest(prompt=prompt, maximum_tokens=20)
    """

    prompt: Prompt
    maximum_tokens: int = 64
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

    def to_json(self) -> Dict[str, Any]:
        payload = {k: v for k, v in self._asdict().items() if v is not None}
        payload["prompt"] = self.prompt.to_json()
        return payload


class CompletionResult(NamedTuple):
    log_probs: Optional[Sequence[Mapping[str, Optional[float]]]] = None
    completion: Optional[str] = None
    completion_tokens: Optional[Sequence[str]] = None
    finish_reason: Optional[str] = None


class CompletionResponse(NamedTuple):
    model_version: str
    completions: Sequence[CompletionResult]
    optimized_prompt: Optional[Sequence[str]] = None

    @staticmethod
    def from_json(json: Dict[str, Any]) -> "CompletionResponse":
        return CompletionResponse(
            model_version=json["model_version"],
            completions=[CompletionResult(**item) for item in json["completions"]],
            optimized_prompt=json.get("optimized_prompt"),
        )

    def to_json(self) -> Mapping[str, Any]:
        return {
            **self._asdict(),
            "completions": [completion._asdict() for completion in self.completions],
        }
