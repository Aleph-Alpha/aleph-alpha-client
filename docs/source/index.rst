.. aleph-alpha-client documentation master file, created by
   sphinx-quickstart on Wed Nov 30 13:14:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to aleph-alpha-client's documentation!
==============================================

Python client for the `Aleph Alpha`_ API.

Usage
-----

Synchronous client.

.. code:: python

      from aleph_alpha_client import Client, CompletionRequest, Prompt
      import os

      client = Client(token=os.getenv("AA_TOKEN"))
      prompt = Prompt.from_text("Provide a short description of AI:")
      request = CompletionRequest(prompt=prompt, maximum_tokens=20)
      result = client.complete(request, model="luminous-extended")

      print(result.completions[0].completion)

Synchronous client with prompt containing an image.

.. code:: python

      from aleph_alpha_client import Client, CompletionRequest, PromptTemplate, Image
      import os

      client = Client(token=os.getenv("AA_TOKEN"))
      image = Image.from_file("path-to-an-image")
      prompt_template = PromptTemplate("{{image}}This picture shows ")
      prompt = prompt_template.to_prompt(image=prompt_template.placeholder(image))
      request = CompletionRequest(prompt=prompt, maximum_tokens=20)
      result = client.complete(request, model="luminous-extended")

      print(result.completions[0].completion)


Asynchronous client.

.. code:: python

   import os
   from aleph_alpha_client import AsyncClient, CompletionRequest, Prompt

   # Can enter context manager within an async function
   async with AsyncClient(token=os.environ["AA_TOKEN"]) as client:
      request = CompletionRequest(
         prompt=Prompt.from_text("Request"),
         maximum_tokens=64,
      )
      response = await client.complete(request, model="luminous-base")

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. _Aleph Alpha: https://aleph-alpha.com