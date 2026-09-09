from unittest.mock import AsyncMock
from time import monotonic
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from main import Catalog, Product, create_app


class FakeCatalog(Catalog):
    def __init__(self) -> None:
        self.products = [
            Product(id=1, name="Apple Juice", description="Fresh apple drink", price=1.99),
            Product(id=2, name="Banana Juice", description="Ripe banana drink", price=2.49),
        ]
        self.loaded_at = monotonic()


class FakeRagStore:
    indexed = 0

    async def ingest(self, products: list[Product]) -> int:
        self.indexed = len(products)
        return self.indexed

    async def search(self, query: str, limit: int = 6) -> list[Product]:
        return await FakeCatalog().search(query, limit)


def test_product_search_returns_matching_products() -> None:
    client = TestClient(create_app(catalog=FakeCatalog()))

    response = client.get("/products", params={"query": "apple"})

    assert response.status_code == 200
    assert [product["name"] for product in response.json()] == ["Apple Juice"]


def test_chat_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "ENV_FILE", main.ENV_FILE.with_name("missing.env"))
    monkeypatch.delenv("OPEN_AI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(catalog=FakeCatalog(), client=AsyncMock(), rag_store=FakeRagStore()))

    response = client.post("/chat", json={"message": "Which apple products do you have?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "OPEN_AI_KEY is not configured"


def test_ingest_indexes_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_AI_KEY", "test-key")
    rag_store = FakeRagStore()
    client = TestClient(create_app(catalog=FakeCatalog(), client=AsyncMock(), rag_store=rag_store))

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "products_indexed": 2}
    assert rag_store.indexed == 2


def test_chat_answers_from_matching_product_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_AI_KEY", "test-key")
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Apple Juice costs $1.99."))]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=completion))))
    test_client = TestClient(create_app(catalog=FakeCatalog(), client=client, rag_store=FakeRagStore()))

    response = test_client.post("/chat", json={"message": "How much is apple juice?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Apple Juice costs $1.99."
    assert response.json()["products"][0]["name"] == "Apple Juice"
    sent_messages = client.chat.completions.create.await_args.kwargs["messages"]
    assert "Apple Juice" in sent_messages[0]["content"]