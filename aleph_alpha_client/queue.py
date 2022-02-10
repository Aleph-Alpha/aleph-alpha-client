
from .image import ImagePrompt
from .aleph_alpha_client import AlephAlphaClient
import asyncio
from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Union

class TaskQueue:
    """
    A async model queue that can be filles with aleph alpha tasks and sent to the API/ models
    """
    def __init__(self, client, no_threads=4, debug=False) -> None:
        self.tasks=[]
        self.results=[]
        self.loop = asyncio.get_event_loop()
        self.threadpool = ThreadPoolExecutor(no_threads)

        if not isinstance(client, AlephAlphaClient):
            raise ValueError("client must be a valid AlephAlphaClient object")

        self.client = client
        pass

    def add_complete_task(
        self,
        model: str,
        prompt: Union[str, List[Union[str, ImagePrompt]]] = "",
        hosting: str = "cloud",
        maximum_tokens: Optional[int] = 64,
        temperature: Optional[float] = 0.0,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        repetition_penalties_include_prompt: Optional[bool] = False,
        use_multiplicative_presence_penalty: Optional[bool] = False,
        best_of: Optional[int] = None,
        n: Optional[int] = 1,
        logit_bias: Optional[Dict[int, float]] = None,
        log_probs: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tokens: Optional[bool] = False,
    ):
        task = locals()
        task.pop("self")
        task["type"]="complete"
        self.tasks.append(task)


    def add_evaluate_task(
        self,
        model,
        completion_expected,
        hosting: str = "cloud",
        prompt: Union[str, List[Union[str, ImagePrompt]]] = "",
    ):
        task = locals()
        task.pop("self")
        task["type"]="evaluate"
        self.tasks.append(task)

    def add_embed_task(
        self,
        model,
        prompt: Union[str, List[Union[str, ImagePrompt]]],
        pooling: List[str],
        layers: List[int],
        hosting: str = "cloud",
        tokens: Optional[bool] = False,
    ):
        task = locals()
        task.pop("self")
        task["type"]="enbed"
        self.tasks.append(task)

    def send_task(self, task):
        model = task.pop("model")
        prompt = task.pop("prompt")
        type = task.pop("type")
        if type == "complete":
            luminous_out = self.client.complete(model, prompt=prompt, **task)
        elif type == "embed":
            luminous_out = self.client.embed(model, prompt=prompt, **task)
        elif type == "evaluate":
            luminous_out = self.client.evaluate(model, prompt=prompt, **task)
        result = {"type":type, "prompt":prompt, "model":model}
        result["result"] = luminous_out
        return result

    def get_results(self):
        return self.results

    def execute(self, overwrite=False):
        self.loop.run_until_complete(self.run())

    async def run(self):
        futures = [
            self.loop.run_in_executor(
                self.threadpool,
                self.send_task, 
                task
            )
            for task in self.tasks
        ]
        for response in await tqdm.gather(*futures):
            self.results.append(response)
