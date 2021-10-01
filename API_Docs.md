# Aleph Alpha API

> ## `GET /version`
> Get the current server version
>> ### Response
>> Responds with the current server version `MAJOR.MINOR.PATCH`.
>>> Example:
>>> ```
>>> 1.12.36
>>> ```

> ## `POST /users/me/token`
> Authenticate with the server.   
> The returned token must be used in an `Authorization: Bearer <token>` header for further requests.
>> ### Request
>> ```ts
>> interface Request {
>>   email: string;
>>   password: string;
>> }
>> ```
>
>> ### Response
>> ```ts
>> interface Response {
>>   token: string | null;
>>   role: "admin" | "worker" | "client" | "monitor" ;
>> }
>> ```

> ## `GET /models_available`
> Get all currently available models.
>> ### Request
>>> Headers:  
>>> `Authorization: Bearer <Token>`  
>>> `Accept: application/json`
>
>> ### Response
>> ```ts
>> type Response = Array<{
>>   name: string;
>>   description: string;
>>   hostings: string[];
>> }>
>> ```

> ## `POST /complete`
> Complete a prompt using a specific model.
> To obtain a valid `model`, use `GET /models_available`.
>> ### Request
>>> Headers:  
>>> `Authorization: Bearer <Token>`  
>>> `Accept: application/json`  
>>> `Content-Type: application/json`
>> ```ts
>> interface Request {
>>   // Name of model to use.  
>>   // A model name refers to a model architecture (number of parameters among others).  
>>   // Always the latest version of model is used. The model output contains information as to the model version.
>>   model: string;
>>   hosting: string;
>>   // The text to be completed.  
>>   // Unconditional completion can be started with an empty string (default).  
>>   // The prompt may contain a zero shot or few shot task.
>>   prompt: string;
>>   // The maximum number of tokens to be generated. Completion will terminate after the maximum number of tokens is 
>>   // reached. Increase this value to generate longer texts. A text is split into tokens.  Usually there are more 
>>   // tokens than words. The summed number of tokens of prompt and maximum_tokens depends on the model 
>>   // (for EleutherAI/gpt-neo-2.7B, it may not exceed 2048 tokens).
>>   maximum_tokens: number;
>>   // A higher sampling temperature encourages the model to produce less probable outputs ("be more creative").  
>>   // Values are expected in a range from 0.0 to 1.0. Try high values (e.g. 0.9) for a more "creative" response and 
>>   // the default 0.0 for a well defined and repeatable answer.
>>   // It is recommended to use either temperature, top_k or top_p and not all at the same time.  
>>   // If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed 
>>   // first. Then top_k is applied. Top_p follows last.
>>   temperature: number | null;
>>   // Introduces random sampling for generated tokens by randomly selecting the next token from the k most likely 
>>   // options. A value larger than 1 encourages the model to be more creative. Set to 0 if repeatable output is to be 
>>   // produced.
>>   // It is recommended to use either temperature, top_k or top_p and not all at the same time.  
>>   // If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed 
>>   // first. Then top_k is applied. Top_p follows last.
>>   top_k: number | null;
>>   // Introduces random sampling for generated tokens by randomly selecting the next token from the smallest possible 
>>   // set of tokens whose cumulative probability exceeds the probability top_p. Set to 0.0 if repeatable output is to 
>>   // be produced.
>>   // It is recommended to use either temperature, top_k or top_p and not all at the same time.  
>>   // If a combination of temperature, top_k or top_p is used rescaling of logits with temperature will be performed
>>   // first. Then top_k is applied. Top_p follows last.
>>   top_p: number | null;
>>   // The presence penalty reduces the likelihood of generating tokens that are already present in the text.  
>>   // Presence penalty is independent of the number of occurences. Increase the value to produce text that is not 
>>   // repeating the input.
>>   // An operation of like the following is applied: 
>>   //     logits[t] -> logits[t] - 1 * penalty
>>   // where logits[t] is the logits for any given token. Note that the formula is independent of the number of times 
>>   // that a token appears in context_tokens.
>>   presence_penalty: number | null;
>>   // The frequency penalty reduces the likelihood of generating tokens that are already present in the text. 
>>   // Presence penalty is dependent on the number of occurences of a token.
>>   // An operation of like the following is applied: 
>>   //     logits[t] -> logits[t] - count[t] * penalty
>>   // where logits[t] is the logits for any given token and count[t] is the number of times that token appears in context_tokens
>>   frequency_penalty: number | null;
>>   // Flag deciding whether presence penalty or frequency penalty are applied to the prompt and completion (True) 
>>   // or only the completion (False)
>>   repetition_penalties_include_prompt: boolean | null;
>>   // Flag deciding whether presence penalty is applied multiplicatively (True) or additively (False). 
>>   // This changes the formula stated for presence and frequency penalty.
>>   use_multiplicative_presence_penalty: boolean | null;
>>   // best_of number of completions are created on server side. The completion with the highest log probability per 
>>   // token is returned. If the parameter n is larger than 1 more than 1 (n) completions will be returned. 
>>   // best_of must be strictly greater than n.
>>   best_of: number | null;
>>   // Number of completions to be returned. If only the argmax sampling is used 
>>   // (temperature, top_k, top_p are all default) the same completions will be produced. This parameter should only be
>>   // increased if a random sampling is chosen.
>>   n: number | null;
>>   // The logit bias allows to influence the likelihood of generating tokens. A dictionary mapping token ids (int) to 
>>   // a bias (float) can be provided. Such bias is added to the logits as generated by the model.
>>   logit_bias: { [key: number]: number } | null;
>>   // Number of top log probabilities to be returned for each generated token. Log probabilities may be used in 
>>   // downstream tasks or to assess the model's certainty when producing tokens.
>>   // No log probs are returned if set to None. Log probs of generated tokens are returned if set to 0. Log probs of 
>>   // generated tokens and top n logprobs are returned if set to n.
>>   log_probs: number | null;
>>   // List of strings which will stop generation if they're generated. Stop sequences may be helpful in structured texts.
>>   // Example: In a question answering scenario a text may consist of lines starting with either "Question: " or "Answer: " (alternating). After producing an answer, the model will be likely to generate "Question: ". "Question: " may therfore be used as stop sequence in order not to have the model generate more questions but rather restrict text generation to the answers.
>>   stop_sequences: string[] | null;
>>   // Flag indicating whether individual tokens of the completion are to be returned (True) or whether solely the 
>>   // generated text (i.e. the completion) is sufficient (False).
>>   tokens: boolean | null;
>> }
>> ```
>
>> ### Response
>> ```ts
>> interface Response {
>>   // unique identifier of a task for traceability
>>   id: string;
>>   // model name and version (if any) of the used model for inference
>>   model_version: string;
>>   // list of completions; may contain only one entry if no more are requested (see parameter n)
>>   completions: {
>>     // list with a dictionary for each generated token. The dictionary maps the keys' tokens to the respective log 
>>     // probabilities. This field is only returned if requested with the parameter "log_probs".
>>     log_probs?: { [key: string]: number | null }[];
>>     // generated completion on the basis of the prompt
>>     completion: string | null;
>>     // completion split into tokens. This field is only returned if requested with the parameter "tokens".
>>     completion_tokens?: string[];
>>     // reason for termination of generation. This may be a stop sequence or maximum number of tokens reached.
>>     finish_reason: string | null;
>>     // an optional message by the system. This may contain warnings or hints.
>>     message?: string;
>>   }[];
>> }
>> ```

> ## `POST /embed`
> Embeds a text using a specific model.  
> Resulting vectors that can be used for downstream tasks (e.g. semantic similarity) and models (e.g. classifiers).  
> To obtain a valid `model`, use `GET /models_available`.
>> ### Request
>>> Headers:  
>>> `Authorization: Bearer <Token>`  
>>> `Accept: application/json`  
>>> `Content-Type: application/json`
>> ```ts
>> interface Request {
>>   // Name of model to use. A model name refers to a model architecture (number of parameters among others). 
>>   // Always the latest version of model is used. The model output contains information as to the model version.
>>   model: string;
>>   // The text to be embedded.
>>   prompt: string;
>>   // A list of layer indices from which to return embeddings.
>>   //     - Index 0 corresponds to the word embeddings used as input to the first transformer layer
>>   //     - Index 1 corresponds to the hidden state as output by the first transformer layer, index 2 to the output of the second layer etc.
>>   //     - Index -1 corresponds to the last transformer layer (not the language modelling head), index -2 to the second last layer etc.
>>   layers: number[];
>>   // Flag indicating whether the tokenized prompt is to be returned (True) or not (False)
>>   tokens: boolean | null;
>>   // Pooling operation to use. No pooling is used (an embedding per input token is returned) if None.
>>   // Pooling operations include:
>>   //     - mean: aggregate token embeddings across the sequence dimension using an average
>>   //     - max: aggregate token embeddings across the sequence dimension using a maximum
>>   //     - last_token: just use the last token
>>   //     - abs_max: aggregate token embeddings across the sequence dimension using a maximum of absolute values
>>   pooling: ("mean" | "max" | "last_token" | "abs_max")[] | null;
>> }
>> ```
>
>> ### Response
>> ```ts
>> interface Response {
>>   // unique identifier of a task for traceability
>>   id: string;
>>   // model name and version (if any) of the used model for inference
>>   model_version: string;
>>   // an optional message by the system. This may contain warnings or hints.
>>   message: string | null;
>>   // embeddings:
>>   //     - no pooling: a dict with layer names as keys and a list of embeddings of size hidden-dim with one entry for each token as values
>>   //     - pooling: a dict with layer names as keys and and pooling output as values. A pooling output is a dict with pooling operation as key and a pooled embedding (list of floats) as values
>>   embeddings:
>>     { [key: string]: number[][] } |
>>     { [key: string]: { [key: "mean" | "max" | "last_token" | "abs_max"]: number[] }; } |
>>     null;
>>   // a list of tokens
>>   tokens: string[] | null;
>> }
>> ```

> ## `POST /evaluate`
> Evaluates the model's likelihood to produce a completion given a prompt.
>> ### Request
>>> Headers:  
>>> `Authorization: Bearer <Token>`  
>>> `Accept: application/json`  
>>> `Content-Type: application/json`
>> ```ts
>> interface Request {
>>   // Name of model to use. A model name refers to a model architecture (number of parameters among others). 
>>   // Always the latest version of model is used. The model output contains information as to the model version.
>>   model: string;
>>   // The ground truth completion expected to be produced given the prompt.
>>   prompt: string;
>>   // The text to be completed. Unconditional completion can be used with an empty string (default). 
>>   // The prompt may contain a zero shot or few shot task.
>>   completion_expected: string;
>> }
>> ```
>
>> ### Response
>> ```ts
>> interface Response {
>>   // unique identifier of a task for traceability
>>   id: string;
>>   // model name and version (if any) of the used model for inference
>>   model_version: string;
>>   // an optional message by the system. This may contain warnings or hints.
>>   message: string | null;
>>   // dictionary with result metrics of the evaluation
>>   result: {
>>     // log probability of producing the expected completion given the prompt. This metric refers to all tokens and 
>>     // is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.
>>     log_probability: number | null;
>>     // log perplexity associated with the expected completion given the prompt. This metric refers to all tokens and
>>     // is therefore dependent on the used tokenizer. It cannot be directly compared among models with different tokenizers.
>>     log_perplexity: number | null;
>>     // log perplexity associated with the expected completion given the prompt normalized for the number of tokens. 
>>     // This metric computes an average per token and is therefore dependent on the used tokenizer. 
>>     // It cannot be directly compared among models with different tokenizers.
>>     log_perplexity_per_token: number | null;
>>     // log perplexity associated with the expected completion given the prompt normalized for the number of characters. 
>>     // This metric is independent of any tokenizer. It can be directly compared among models with different tokenizers.
>>     log_perplexity_per_character: number | null;
>>     // Flag indicating whether a greedy completion would have produced the expected completion.
>>     correct_greedy: boolean | null;
>>     // Number of tokens in the expected completion.
>>     token_count: number | null;
>>     // Number of characters in the expected completion.
>>     character_count: number | null;
>>     // argmax completion given the input consisting of prompt and expected completion. This may be used as an indicator 
>>     // of what the model would have produced. As only one single forward is performed an incoherent text could be 
>>     // produced especially for long expected completions.
>>     completion: string | null;
>>   };
>> }
>> ```

> ## `POST /tokenize`
> Tokenize a prompt for a specific model.  
> To obtain a valid `model`, use `GET /models_available`.
>> ### Request
>>> Headers:  
>>> `Authorization: Bearer <Token>`  
>>> `Accept: application/json`  
>>> `Content-Type: application/json`
>> ```ts
>> interface Request {
>>   model: string;
>>   prompt: string;
>>   tokens: boolean;
>>   token_ids: boolean;
>> }
>> ```
>
>> ### Response
>> ```ts
>> interface Response {
>>   tokens?: string[];
>>   token_ids?: number[];
>> }
>> ```

> ## `POST /detokenize`
> Detokenize a list of tokens into a string.  
> To obtain a valid `model`, use `GET /models_available`.
>> ### Request
>>> Headers:  
>>> `Authorization: Bearer <Token>`  
>>> `Accept: application/json`  
>>> `Content-Type: application/json`
>> ```ts
>> interface Request {
>>   model: string,
>>   token_ids: number[],
>> }
>> ```
>
>> ### Response
>> ```ts
>> interface Response {
>>   result: string
>> }
>> ```
