import httpx
import respx

from astromesh_rag.embedders import HFEmbedder, OllamaEmbedder
from astromesh_rag.llms import OpenAICompatLLM
from astromesh_rag.rerankers import CohereReranker
from astromesh_rag.types import Chunk, Hit


@respx.mock
async def test_ollama_batches_in_one_call():
    route = respx.post("http://o/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0], [2.0]]})
    )
    assert await OllamaEmbedder(endpoint="http://o").embed(["a", "b"]) == [[1.0], [2.0]]
    assert route.call_count == 1


@respx.mock
async def test_hf_batch_and_nested_normalization():
    respx.post("http://h/x").mock(return_value=httpx.Response(200, json=[[[1.0, 2.0]]]))
    assert await HFEmbedder("m", endpoint="http://h/x", api_key="k").embed(["a"]) == [[1.0, 2.0]]


@respx.mock
async def test_openai_compat_returns_message_and_usage():
    respx.post("http://gw/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "hola"}}], "usage": {"total_tokens": 7}}
        )
    )
    llm = OpenAICompatLLM("http://gw/v1", "k", "kimi")
    assert await llm.complete("hi") == "hola"
    assert llm.usage == [{"total_tokens": 7}]


@respx.mock
async def test_openai_compat_omits_temperature_when_none():
    route = respx.post("http://gw/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
    )
    await OpenAICompatLLM("http://gw/v1", "k", "m", temperature=None).complete("x")
    assert "temperature" not in route.calls[0].request.content.decode()


@respx.mock
async def test_cohere_rerank_keeps_details():
    respx.post("https://api.cohere.ai/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9}]})
    )
    hits = [
        Hit(Chunk(f"d#{i}", "d", t), 0.1, "vector", {"vector": 0.1}) for i, t in enumerate("ab")
    ]
    out = await CohereReranker(api_key="k").rerank("q", hits, 1)
    assert out[0].chunk.text == "b"
    assert out[0].score == 0.9
    assert out[0].details == {"vector": 0.1, "rerank": 0.9}
