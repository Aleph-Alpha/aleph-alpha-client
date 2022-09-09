# Using Experimental Checkpoints with the Python Client

We occaisionally run experimental checkpoints on our API. For users with access (Aleph Alpha employees), you are able to use these checkpoints from the Python client by specifying a `checkpoint_name` rather than a `model_name` when initializing `AlephAlphaModel`.

From this point, you are able to use the client like you normally do, and are able to send any supported requests for that checkpoint.

```python
from aleph_alpha_client import ImagePrompt, AlephAlphaModel, AlephAlphaClient, CompletionRequest, Prompt
import os

model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=os.getenv("AA_TOKEN")),
    checkpoint_name = "some-experimental-checkpoint"
)

request = CompletionRequest(prompt=Prompt.from_text("The api works"), maximum_tokens=20)
result = model.complete(request)

print(result.completions[0].completion)
```
