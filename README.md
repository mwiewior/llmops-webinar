# llmops-webinar

## Dependencies

### Software
* Python >=3.12
* [Poetry](https://python-poetry.org/docs/)
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](httpshttps://docs.docker.com/compose/install/)
* [Ollama](https://ollama.com/)

### Hardware
* >= 16GB VRAM
* CUDA or MPS (Apple Silicon chips) compatible GPU ()

## Observability with Langfuse
```bash
cd ext/langfuse
docker compose up -d
docker compose logs -f
```

## Environment variables
```bash
AZURE_OPENAI_API_KEY=xxx
LANGFUSE_HOST=http://localhost:3000;
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
```


## Finetuning with torchtune
Ensure you use torchtune from the virtual environment managed by poetry.
```bash
poetry shell
which tune
```
should return something like:
```text
/Users/xxx/Library/Caches/pypoetry/virtualenvs/llmops-webinar-h7IZpbbf-py3.12/bin/tune
```