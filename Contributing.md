# Contributions

## Testing

Tests use pytests with (optional) coverage plugin. Install the locally cloned repo in editable mode with:

```sh
uv sync
```

**Tests make api calls that reduce your quota!**

### Run tests

Tests can be run using pytest. Make sure to create a `.env` file with the following content:

```env
# test settings
TEST_API_URL=https://inference-api.your-domain.com
TEST_TOKEN=your_token
```

Instead of a token username and password can be used.

```env
# test settings
TEST_API_URL=https://inference-api.your-domain.com
TEST_USERNAME=your_username
TEST_PASSWORD=your_password
```

- A coverage report can be created using the optional arguments --cov-report and --cov (see pytest documentation)
- A subset of tests can be selected by pointing to the module within tests

```sh
# run all tests, output coverage report of aleph_alpha_client module in terminal
uv run pytest --cov-report term --cov=aleph_alpha_client tests
uv run pytest tests -v # start verbose
```

If an html coverage report has been created a simple http server can be run to serve static files.

```sh
uv run python -m http.server --directory htmlcov 8000
```

## Releasing a new version

1. Update version.py to the to-be-released version, say 1.2.3
2. Commit and push the changes to version.py to master
3. Tag the commit with v1.2.3 to match the version in version.py
4. Push the tag
5. Create a Release in github from the new tag. This will trigger the "Upload Python Package" workflow.

## Working on our documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) for our documentation and publish it on [Read the Docs](https://aleph-alpha-client.readthedocs.io/en/latest/).
To work on the documentation, you need to install the project with the `docs` dependency group.

```sh
uv sync --group docs
```

The documentation can be generated with:

```sh
cd docs
uv run make html
```

Make sure that the documentation can be generated without Sphinx warnings or errors.
