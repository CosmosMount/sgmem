import io
import json
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from sgmem import raptor_memory as raptor


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return text.split()

    def __call__(self, text, **kwargs):
        assert kwargs["return_offsets_mapping"] and kwargs["truncation"] is False
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
        assert messages[0]["content"] == raptor.SUMMARY_SYSTEM
        assert messages[1] == {"role": "user", "content": "source text"}
        return [1, 2, 3, 4]


class Encoder:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] and kwargs["convert_to_numpy"]
        assert all(text.startswith(("passage: ", "query: User_A ")) for text in texts)
        return [[1, 0] for _ in texts]


def messages(*texts):
    return [{"domain": "Finance", "id": f"M{i}", "content": text, "speaker": "User_A",
             "group": "Audit", "reply_to": None, "timestamp": "2026-01-02"} for i, text in enumerate(texts)]


def get_text(nodes):
    return "".join(" ".join(node.text.splitlines()) + "\n\n" for node in nodes)


def test_build_feeds_message_leaves_and_recursive_hooks(monkeypatch):
    official = ModuleType("raptor")
    official.BaseEmbeddingModel = type("Embedding", (), {})
    official.BaseSummarizationModel = type("Summarizer", (), {})

    def node(text, index, children, embeddings):
        return SimpleNamespace(text=text, index=index, children=children, embeddings=embeddings)

    official.Node = node
    official.ClusterTreeConfig = SimpleNamespace
    sizes = []
    observed_contexts = []

    class Builder:
        def __init__(self, config):
            self.config = config
            assert config.cluster_embedding_model == "e5"
            assert config.summarization_length == 256 and config.num_layers == 5
            assert isinstance(config.embedding_models["e5"], official.BaseEmbeddingModel)
            assert isinstance(config.summarization_model, official.BaseSummarizationModel)

        def construct_tree(self, leaves, all_nodes, layers, *, use_multithreading, max_workers):
            assert use_multithreading is False and max_workers == 1
            assert len(leaves) == 9 and all(not leaf.children for leaf in leaves.values())
            current = list(leaves.values())
            for layer in (1, 2):
                current = self.build_layer(current, all_nodes)
                layers[layer] = current

        def build_layer(self, current, all_nodes):
            groups = self.config.clustering_algorithm.perform_clustering(current, "e5", 2)
            parents = []
            for group in groups:
                text = self.config.summarization_model.summarize(get_text(group), self.config.summarization_length)
                vector = self.config.embedding_models["e5"].create_embedding(text)
                index = len(all_nodes)
                parent = all_nodes[index] = node(text, index, {child.index for child in group}, {"e5": vector})
                parents.append(parent)
            return parents

    official.ClusterTreeBuilder = Builder

    def cluster(vectors, *, dim, threshold):
        assert dim == 2 and threshold == 0.1
        sizes.append(len(vectors))
        labels = 4 if len(vectors) > 4 else 2
        return [np.array([0, 1] if i == 0 else [i % labels]) for i in range(len(vectors))]

    def summarize(text, max_tokens):
        assert max_tokens == 256
        assert "User_A" not in text and "Audit" not in text
        observed_contexts.append(text)
        return f"summary {len(observed_contexts)}"

    monkeypatch.setitem(sys.modules, "raptor", official)
    monkeypatch.setitem(sys.modules, "raptor.cluster_utils", SimpleNamespace(perform_clustering=cluster))
    monkeypatch.setitem(sys.modules, "raptor.utils", SimpleNamespace(get_text=get_text))
    source = messages("first part travels further", *[f"body{i}" for i in range(7)])
    tree = raptor.build_raptor(source, Encoder(), Tokenizer(), summarize,
                               config=raptor.RaptorConfig(leaf_tokens=2, reduction_dimension=2))
    assert sizes == [9, 4]
    assert len(observed_contexts) == 6 and "summary" in observed_contexts[-1]
    assert tree.nodes[0] == raptor.RaptorNode("first part ", 0, (), "M0")
    assert tree.nodes[1] == raptor.RaptorNode("travels further", 0, (), "M0")
    assert 0 in tree.nodes[9].children and 0 in tree.nodes[10].children
    assert [item.layer for item in tree.nodes[-2:]] == [2, 2]
    retriever = raptor.RaptorRetriever(source, tree, Encoder(), Tokenizer())
    assert retriever.retrieve("which?", 1, asking_user_id="User_A")[0]["message"] is source[0]


def test_unsplittable_and_small_clusters_fit_token_budget():
    calls = []

    def stagnant(vectors, **_):
        calls.append(len(vectors))
        return [np.array([0])] * len(vectors)

    nodes = [SimpleNamespace(text=f"part {index}", embeddings={"e5": [1, 0]}) for index in range(4)]
    tokens = raptor._Tokens(Tokenizer())
    assert raptor._partition(nodes, stagnant, get_text, tokens, 4, 2, 0.1) == [nodes[:2], nodes[2:]]
    assert raptor._partition(nodes[:2], stagnant, get_text, tokens, 4, 2, 0.1) == [nodes[:2]]
    assert calls == [4]


def test_cache_round_trip_collapsed_leaf_and_native_retrieval(tmp_path):
    source = messages("alpha", "beta", "gamma")
    domain, digest = raptor._source_identity(source)
    nodes = [raptor.RaptorNode(row["content"], 0, (), row["id"]) for row in source]
    nodes += [raptor.RaptorNode("alpha gamma summary", 1, (0, 2), None)]
    vectors = np.array([[0.6, 0.8], [0.8, 0.6], [0, 1], [1, 0]], dtype=np.float32)
    tree = raptor.RaptorTree(domain, digest, raptor.RaptorConfig(), {"embedding_revision": "fixed"}, nodes, vectors)
    path = tmp_path / "tree.npz"
    tree.save(path)
    loaded = raptor.RaptorTree.load(path)
    assert tree.config == loaded.config and nodes == loaded.nodes and tree.provenance == loaded.provenance
    assert np.array_equal(vectors, loaded.vectors)
    retriever = raptor.RaptorRetriever(source, loaded, Encoder(), Tokenizer())
    call = {"asking_user_id": "User_A"}
    collapsed = retriever.retrieve("what?", 3, **call)
    assert [row["message"]["id"] for row in collapsed] == ["M0", "M2", "M1"]
    assert collapsed[0]["message"] is source[0] and collapsed[0]["trace"]["node_id"] == 3
    native = retriever.retrieve_nodes("what?", 3, max_context_tokens=3, **call)
    assert native == [{"node_id": 3, "layer": 1, "text": "alpha gamma summary", "children": [0, 2],
                       "source_ids": ["M0", "M2"], "score": 1.0}]
    assert [hit["node_id"] for hit in retriever.retrieve_nodes("what?", 3, max_context_tokens=None, **call)] == [3, 1, 0]
    leaf = raptor.RaptorRetriever(source, loaded, Encoder(), Tokenizer(), leaf_only=True)
    assert [row["message"]["id"] for row in leaf.retrieve("what?", 3, **call)] == ["M1", "M0", "M2"]
    assert all(hit["layer"] == 0 for hit in leaf.retrieve_nodes("what?", 3, **call))

    with pytest.raises(ValueError, match="one domain"):
        raptor.RaptorRetriever([source[0], {**source[1], "domain": "Technology"}], loaded, Encoder(), Tokenizer())
    with pytest.raises(ValueError, match="does not match"):
        raptor.RaptorRetriever(source[::-1], loaded, Encoder(), Tokenizer())
    loaded.nodes[-1] = raptor.RaptorNode("cycle", 1, (3,), None)
    with pytest.raises(ValueError, match="Invalid RAPTOR children"):
        raptor.RaptorRetriever(source, loaded, Encoder(), Tokenizer())


@pytest.mark.parametrize("endpoint", ("http://example.com/v1", "http://localhost/v1", "https://127.0.0.1/v1",
                                       "http://user@127.0.0.1/v1", "http://127.0.0.1/v1?other=1"))
def test_summary_requires_direct_loopback_endpoint(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        raptor.LocalQwenSummarizer(endpoint, "local", Tokenizer())


def test_summary_rejects_overflow_redirect_and_bad_token_accounting(monkeypatch):
    cap = raptor.RaptorConfig().summary_tokens
    state = SimpleNamespace(status=200, finish="stop", reported_prompt=329, completion=110, calls=0)

    class LongTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            super().apply_chat_template(messages, **kwargs)
            return [0] * 329

    class Connection:
        def __init__(self, host, port, **_):
            assert host == "127.0.0.1" and port == 18951

        def request(self, verb, path, body, headers):
            state.calls += 1
            assert verb == "POST" and path == "/v1/chat/completions"
            assert headers["Content-Type"] == "application/json"
            payload = json.loads(body)
            assert payload["max_tokens"] == 256 and payload["model"] == "local"
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}

        def getresponse(self):
            response = io.StringIO(json.dumps({
                "choices": [{"finish_reason": state.finish, "message": {"content": "summary"}}],
                "usage": {"prompt_tokens": state.reported_prompt, "completion_tokens": state.completion},
            }))
            response.status = state.status
            return response

        def close(self):
            pass

    monkeypatch.setattr(raptor, "HTTPConnection", Connection)
    summarize = raptor.LocalQwenSummarizer("http://127.0.0.1:18951/v1", "local", LongTokenizer(),
                                         context_tokens=329 + cap)
    with pytest.raises(ValueError, match="context budget"):
        summarize("source text", cap + 1)
    assert state.calls == 0 and summarize("source text", cap) == "summary"
    assert summarize.last_usage["completion_tokens"] == 110
    state.status = 302
    with pytest.raises(RuntimeError, match="302"):
        summarize("source text", cap)
    state.status, state.finish, state.completion = 200, "length", cap
    with pytest.raises(ValueError, match="finish=length, prompt=329/329, completion=256/256"):
        summarize("source text", cap)
    assert summarize.last_response["usage"]["completion_tokens"] == 256
    state.finish, state.reported_prompt = "stop", 328
    with pytest.raises(ValueError, match="token accounting"):
        summarize("source text", cap)
    assert state.calls == 4 and summarize.last_usage is None
