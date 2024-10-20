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
