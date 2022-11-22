import asyncio
import time
from aleph_alpha_client import (
    AsyncClient,
    CompletionRequest,
    CompletionResponse,
    Prompt,
)


token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjozNiwicm9sZSI6IkFkbWluIiwiY3JlYXRlZCI6MTYzNDcyOTQ0N30._JyORCppn_8bHH1cCMLdoeTsI0c-ZymwF-AJseYhmhI"


async def main():
    async with AsyncClient(token=token) as client:
        request = CompletionRequest(
            prompt=Prompt.from_text(f"Request"), maximum_tokens=64
        )
        response = await client.complete(request, "luminous-base")
        print(response.completions[0].completion)


async def main_manual():
    client = AsyncClient(token=token)

    request = CompletionRequest(
        prompt=Prompt.from_text(f"Request"), maximum_tokens=64, temperature=1
    )
    response: CompletionResponse = await client.complete(request, "luminous-base")
    print(response.completions[0].completion)

    # Make sure to close at the end of your script/app to close the underlying connection pool
    await client.close()


async def main_gather():
    async with AsyncClient(token=token) as client:
        # You have several prompts you need to generate completions for
        requests = (
            CompletionRequest(prompt=Prompt.from_text(prompt), maximum_tokens=64)
            for prompt in ("Fewshot prompt 1", "Fewshot prompt 2")
        )
        # await the requests together
        responses = await asyncio.gather(
            *(client.complete(req, model="luminous-base") for req in requests)
        )
        for response in responses:
            print(response.completions[0].completion)


async def main_semaphore():
    start = time.perf_counter()

    async def gather_with_concurrency(n, *tasks):
        semaphore = asyncio.Semaphore(n)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(sem_task(task) for task in tasks))

    async with AsyncClient(token=token) as client:
        # Lots of requests to execute
        requests = (
            CompletionRequest(prompt=Prompt.from_text(f"Prompt {i}"), maximum_tokens=64)
            for i in range(1000)
        )
        # await the requests together, 60 running at a time
        conc_req = 40  # Depends a lot on which model and the size of task
        responses = await gather_with_concurrency(
            conc_req, *(client.complete(req, model="luminous-base") for req in requests)
        )
        # for response in responses:
        #     print(response.completions[0].completion)

    end = time.perf_counter()
    print(f"Time: {(end-start)} seconds")


# asyncio.run(main())
# asyncio.run(main_manual())
asyncio.run(main_gather())
asyncio.run(main_semaphore())
