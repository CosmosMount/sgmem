import re
from types import SimpleNamespace

import pytest

from sgmem.memory_context import select_units
from sgmem.memory_retrieval import public_sources
from sgmem.qa import build_answer_prompt, evaluate_question, format_message


QUESTION = {"domain": "Finance", "question": "What is my budget?", "asking_user_id": "User_1",
            "answer": "SECRET_GOLD"}
SOURCE = {"domain": "Finance", "id": "M1", "speaker": "User_2", "group": "G1", "role": "Auditor",
          "content": "Budget &amp; 10", "timestamp": "2026-01-01", "reply_to": "M0"}


class FramedTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "SECRET_GOLD" not in messages[1]["content"]
        return (f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def __call__(self, text, **kwargs):
        assert kwargs == {"return_tensors": "pt"}
        # Treat labels, punctuation and newlines as tokens. Include a default special token.
        count = 1 + len(re.findall(r"\w+|[^\w\s]|\n", text))
        return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, count)))


def unit(unit_id, text, source_ids, score=0.25):
    return dict(unit_id=unit_id, text=text, source_ids=source_ids, score=score)


def test_original_qa_prompt_bytes_and_preformatted_documents():
    calls = []

    class Client:
        last_usage = None

        def complete(self, system, user, max_tokens):
            calls.append((system, user.encode("utf-8"), max_tokens))
            return "Final: 10" if system == "agent" else "Final: Correct"

    legacy = (b"Asking user: User_1\n\nQuestion:\nWhat is my budget?\n\nRetrieved passages:\n"
              b"[1] [user=User_2 / speaker_role=Auditor / channel=G1 / timestamp=2026-01-01 / reply_to=M0 / msg_node=M1]\n"
              b"Budget & 10\n\nAnswer the question using the retrieved passages.")
    result = evaluate_question(QUESTION, [{"message": SOURCE}], Client(), Client(), "agent", "judge")
    assert calls == [("agent", legacy, 512),
                     ("judge", b"Question:\nWhat is my budget?\n\nGold Answer:\nSECRET_GOLD\n\nAgent Answer:\n10\n", 256)]
    assert result["retrieved_docs"] == [format_message(SOURCE)]
    calls.clear()
    documents = ["[Native memory / source_user=User_2]\nBudget is 10."]
    native = evaluate_question(QUESTION, [], Client(), Client(), "agent", "judge", documents=documents)
    assert native["retrieved_docs"] == documents
    assert calls[0][1] == build_answer_prompt(QUESTION, documents).encode()
    assert b"[1] [Native memory" in calls[0][1] and b"msg_node" not in calls[0][1]


def test_ranked_prefix_budget_includes_real_prompt_framing():
    sources = public_sources([SOURCE])
    tokenizer = FramedTokenizer()
    candidates = [unit(0, "word " * 1800, ["M1"], 0.1),
                  unit(1, "word " * 1700, ["M1"], 0.9), unit(2, "Short.", ["M1"], 1.0)]
    assert len((candidates[0]["text"] + candidates[1]["text"]).split()) == 3500
    docs, trace = select_units(QUESTION, candidates, tokenizer, "Read evidence.", sources=sources)
    assert len(docs) == 1 and docs[0].endswith(candidates[0]["text"])
    assert trace["kept"] == [{"unit_id": 0, "source_ids": ["M1"], "score": 0.1}]
    assert [row["unit_id"] for row in trace["dropped"]] == [1, 2]
    assert 1800 < trace["evidence_tokens"] <= trace["budget_tokens"] == 3500
    assert trace["first_rejected_prompt_tokens"] - trace["empty_prompt_tokens"] > 3500
    expected_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Read evidence."},
         {"role": "user", "content": build_answer_prompt(QUESTION, docs)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert trace["prompt_tokens"] == tokenizer(expected_prompt, return_tensors="pt").input_ids.shape[1]


def test_raw_control_and_native_attribution_follow_authoritative_sources():
    sources = public_sources([
        SOURCE, {**SOURCE, "id": "M2", "speaker": "User_3"}, {**SOURCE, "id": "M3", "group": "G2"},
    ])
    entries = [unit("M1", SOURCE["content"], ["M1"])]
    docs, _ = select_units(QUESTION, entries, FramedTokenizer(), "agent", sources=sources, raw=True)
    assert docs == [format_message(SOURCE)]
    entries = [unit("multi-speaker", "Two speakers discuss the budget.", ["M2", "M1"]),
               unit("multi-group", "One user discusses two groups.", ["M3", "M1"])]
    docs, trace = select_units(QUESTION, entries, FramedTokenizer(), "agent", sources=sources)
    assert docs[0].startswith("[Native memory / source_channel=G1]\n")
    assert docs[1].startswith("[Native memory / source_user=User_2]\n")
    assert trace["kept"][0]["source_ids"] == ["M2", "M1"]
    assert trace["kept"][1]["source_ids"] == ["M3", "M1"]


@pytest.mark.parametrize(("bad", "error"), [
    (unit("later", "Text", ["missing"]), "Unknown or wrong-domain source"),
    (unit("later", "Text", ["M1", "M1"]), "Invalid source lineage"),
    (unit("later", "Text", ["M1"], float("inf")), "Invalid text or score"),
    (unit("later", " ", ["M1"]), "Invalid text or score"),
    (unit(True, "Text", ["M1"]), "Invalid memory unit ID"),
    (unit("first", "Text", ["M1"]), "Duplicate memory unit ID"),
])
def test_invalid_dropped_suffix_cannot_escape_validation(bad, error):
    with pytest.raises(ValueError, match=error):
        select_units(QUESTION, [unit("first", "Too long for a zero budget.", ["M1"]), bad],
                     FramedTokenizer(), "agent", 0, sources=public_sources([SOURCE]))


def test_cross_domain_lineage_and_empty_prefix():
    sources = public_sources([SOURCE])
    entries = [unit("first", "Too long for a zero budget.", ["M1"])]
    docs, trace = select_units(QUESTION, entries, FramedTokenizer(), "agent", 0, sources=sources)
    assert docs == [] and trace["kept"] == [] and trace["evidence_tokens"] == 0
    assert trace["dropped"][0]["unit_id"] == "first"
    with pytest.raises(ValueError, match="Question domain"):
        select_units({**QUESTION, "domain": "Technology"}, entries, FramedTokenizer(), "agent", sources=sources)
    wrong_source = {("Finance", "M1"): {**SOURCE, "domain": "Technology"}}
    with pytest.raises(ValueError, match="wrong-domain source"):
        select_units(QUESTION, entries, FramedTokenizer(), "agent", sources=wrong_source)
