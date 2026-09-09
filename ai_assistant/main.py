from __future__ import annotations

import json
import os
import time
import asyncio
from dataclasses import dataclass, field
# from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import chromadb
import httpx
# from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
# from openai import AsyncOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ENV_FILE = Path(__file__).resolve().parent.parent / ".env.openai"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PRODUCTS_URL = os.getenv("PRODUCTS_URL", "http://localhost:3000/api/Products")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "juice-shop-local")
CATALOG_CACHE_SECONDS = 300
RAG_RESULTS = 6


class Product(BaseModel):
    id: int
    name: str
    description: str
    price: float
    deluxePrice: float | None = None
    image: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class FrontendChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)


class ChatResponse(BaseModel):
    answer: str
    products: list[Product]


@dataclass
class RagStore:
    client: chromadb.ClientAPI | None = None
    embedding_model: Any = None

    def model(self) -> Any:
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.embedding_model

    def collection(self):
        if self.client is None:
            self.client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT
            )
        return self.client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )

    async def ingest(self, products: list[Product]) -> int:
        if not products:
            return 0

        documents = [
            f"{product.name}: {product.description}"
            for product in products
        ]

        embeddings = self.model().encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        self.collection().upsert(
            ids=[str(product.id) for product in products],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    key: value
                    for key, value in product.model_dump().items()
                    if value is not None
                }
                for product in products
            ]
        )

        return len(products)

    async def search(
        self,
        query: str,
        limit: int = RAG_RESULTS
    ) -> list[Product]:
        embedding = self.model().encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        result = self.collection().query(
            query_embeddings=embedding,
            n_results=limit
        )

        metadata = result.get("metadatas", [[]])[0]

        return [
            Product.model_validate(item)
            for item in metadata
            if item
        ]


@dataclass
class Catalog:
    products: list[Product] = field(default_factory=list)
    loaded_at: float = 0

    async def all(self) -> list[Product]:
        if self.products and time.monotonic() - self.loaded_at < CATALOG_CACHE_SECONDS:
            return self.products

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(PRODUCTS_URL)
                response.raise_for_status()
                payload = response.json()
            self.products = [Product.model_validate(item) for item in payload["data"]]
            self.loaded_at = time.monotonic()
            return self.products
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=503, detail="The product catalog is unavailable") from error

    async def search(self, query: str, limit: int = 8) -> list[Product]:
        terms = {term.lower() for term in query.split() if term.strip()}
        products = await self.all()

        def score(product: Product) -> tuple[int, str]:
            searchable = f"{product.name} {product.description}".lower()
            exact_name = query.lower().strip() in product.name.lower()
            return (len(terms & set(searchable.split())) + (10 if exact_name else 0), product.name)

        matches = [
            product for product in products
            if any(term in f"{product.name} {product.description}".lower() for term in terms)
        ]
        return sorted(matches, key=score, reverse=True)[:limit]

# def load_api_key() -> str:
#     values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
#     return os.getenv("OPEN_AI_KEY") or values.get("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY") or ""

def create_app(
    catalog: Catalog | None = None,
    rag_store: RagStore | None = None
) -> FastAPI:
    catalog = catalog or Catalog()
    rag_store = rag_store or RagStore()
    app = FastAPI(title="Juice Shop Product Assistant", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:4200"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # async def answer(request: ChatRequest) -> ChatResponse:
    #     if not load_api_key():
    #         raise HTTPException(status_code=503, detail="OPEN_AI_KEY is not configured")
    async def answer(request: ChatRequest) -> ChatResponse:

        try:
            matches = await rag_store.search(request.message)
        except Exception as error:
            raise HTTPException(status_code=503, detail="The product knowledge base is unavailable") from error

        product_context = json.dumps([product.model_dump() for product in matches])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise product assistant for OWASP Juice Shop. "
                    "Answer product questions using only the retrieved product context below. "
                    "Never invent products, prices, availability, or features. "
                    "If the retrieved context is empty, say that no matching products were found. "
                    f"Retrieved product context: {product_context}"
                ),
            },
            *[message.model_dump() for message in request.history],
            {"role": "user", "content": request.message},
        ]

        # try:
        #     completion = await client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.2)
        #     answer_text = completion.choices[0].message.content or "I could not find an answer for that product question."
        # except Exception as error:
        #     raise HTTPException(status_code=502, detail="The AI provider is unavailable") from error
        try:
            async with httpx.AsyncClient(timeout=120) as ollama_client:
                response = await ollama_client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.2
                        }
                    },
                )

                response.raise_for_status()
                data = response.json()

            answer_text = data["message"]["content"]

        except Exception as error:
            print(
                f"OLLAMA ERROR: {type(error).__name__}: {error}",
                flush=True
            )
            raise HTTPException(
                status_code=502,
                detail="The AI provider is unavailable"
            ) from error
        return ChatResponse(answer=answer_text, products=matches)

    async def ingest_on_startup() -> None:
        for attempt in range(12):
            try:
                count = await rag_store.ingest(await catalog.all())
                print(f"Indexed {count} products in Chroma", flush=True)
                return
            except Exception as error:
                print(f"Catalog indexing attempt {attempt + 1} failed: {error}", flush=True)
                await asyncio.sleep(5)

    @app.on_event("startup")
    async def startup() -> None:
        if os.getenv("AUTO_INGEST", "true").lower() == "true":
            asyncio.create_task(ingest_on_startup())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/products", response_model=list[Product])
    async def products(query: str = "") -> list[Product]:
        return await catalog.search(query) if query else await catalog.all()

    @app.post("/ingest")
    async def ingest() -> dict[str, int | str]:
        try:
            count = await rag_store.ingest(await catalog.all())
        except HTTPException:
            raise
        except Exception as error:
                print(f"INGEST ERROR: {type(error).__name__}: {error}", flush=True)
                raise HTTPException(
                    status_code=502,
                    detail=f"{type(error).__name__}: {error}"
                ) from error
        return {"status": "ok", "products_indexed": count}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            matches = await rag_store.search(request.message)
        except Exception as error:
            print(f"RAG SEARCH ERROR: {type(error).__name__}: {error}", flush=True)
            raise HTTPException(
                status_code=503,
                detail="The product knowledge base is unavailable"
            ) from error

        product_context = json.dumps(
            [product.model_dump() for product in matches]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise product assistant for OWASP Juice Shop. "
                    "Answer product questions using only the retrieved product "
                    "context below. Never invent products, prices, availability, "
                    "or features. If the retrieved context is empty, say that "
                    "no matching products were found. "
                    f"Retrieved product context: {product_context}"
                ),
            },
            *[message.model_dump() for message in request.history],
            {"role": "user", "content": request.message},
        ]

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                    },
                )

                response.raise_for_status()
                data = response.json()

            answer = data["message"]["content"]

        except Exception as error:
            print(
                f"OLLAMA ERROR: {type(error).__name__}: {error}",
                flush=True
            )
            raise HTTPException(
                status_code=502,
                detail="The local AI provider is unavailable"
            ) from error

        return ChatResponse(
            answer=answer,
            products=matches
        )

    @app.post("/rest/chat")
    async def frontend_chat(request: FrontendChatRequest) -> StreamingResponse:
        user_messages = [message for message in request.messages if message.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="A user message is required")
        latest = user_messages[-1]
        history = request.messages[:-1]

        async def stream():
            try:
                result = await answer(ChatRequest(message=latest.content, history=history))
                yield f"data: {json.dumps({'choices': [{'delta': {'content': result.answer}}]})}\n\n"
                yield "data: [DONE]\n\n"
            except HTTPException as error:
                yield f"data: {json.dumps({'error': error.detail})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()