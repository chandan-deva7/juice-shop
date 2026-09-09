# Juice Shop Product Assistant

This is a standalone Python assistant exposed through FastAPI. It reads the OpenAI key from the root `.env.openai` file, indexes the running Juice Shop product catalog as OpenAI embeddings in Chroma, and grounds responses in retrieved products.

## Run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai_assistant\requirements.txt
uvicorn ai_assistant.main:app --reload --port 8000
```

The API is available at `http://localhost:8000/docs`.

## Run with Docker

Start Chroma and the assistant together while Juice Shop is running on port 3000:

```powershell
docker compose up --build -d
```

Chroma is persisted in the `chroma-data` Docker volume and is exposed at `http://localhost:8001` for inspection. The assistant talks to Chroma over the internal Docker network.

The Juice Shop frontend is available at `http://localhost:3000` and its chat requests go directly to the assistant at `http://localhost:8000/rest/chat`. The assistant automatically indexes the catalog after startup.

The OpenAI account must have available API credits for both chat completions and embeddings. Run ingestion again after credit is available; existing Chroma data remains in the volume.

`PRODUCTS_URL` can point to another Juice Shop instance, `OPENAI_MODEL` can select a different chat model, and `OPENAI_EMBEDDING_MODEL` can select a different embedding model.

Example request:

```powershell
Invoke-RestMethod http://localhost:8000/chat -Method Post -ContentType 'application/json' -Body '{"message":"Which apple products do you have?"}'
```