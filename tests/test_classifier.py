"""ClassifierBackend (needs torch + transformers; skipped otherwise)."""
import pytest

from shishya.backends import BackendError
from shishya.backends.base import render
from shishya.schema import compile_schema

SCHEMA = {"type": "object", "properties": {
    "news_impact": {"type": "string", "enum": ["bad", "good", "routine"]},
    "action": {"type": "string", "enum": ["hold", "trim", "add"]}}}

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    d = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "auditor", "resigns", "profit", "up", "notice"]
    (d / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, num_labels=3, id2label={0: "bearish", 1: "bullish", 2: "neutral"},
                     label2id={"bearish": 0, "bullish": 1, "neutral": 2})
    cfg.shishya_temperature = 2.0
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(d)
    return str(d)


MAPPING = {"news_impact": {"bad": "bearish", "good": "bullish", "routine": "neutral"}}


def test_classifier_scores_mapped_options_and_reads_temperature(tiny_model):
    from shishya.backends import ClassifierBackend
    c = ClassifierBackend(tiny_model, fields=MAPPING, device="cpu")
    assert c.temperature == 2.0
    f = compile_schema(SCHEMA).field("news_impact")
    got = c.score("auditor resigns", [f], {"news_impact": f.options})["news_impact"]
    assert set(got) == {"bad", "good", "routine"} and abs(sum(got.values()) - 1) < 1e-5


def test_classifier_ignores_the_facts_block(tiny_model):
    from shishya.backends import ClassifierBackend
    c = ClassifierBackend(tiny_model, fields=MAPPING, device="cpu")
    f = compile_schema(SCHEMA).field("news_impact")
    bare = c.score("auditor resigns", [f], {"news_impact": f.options})
    with_facts = c.score(render("auditor resigns", {"owned_qty": 40}), [f], {"news_impact": f.options})
    assert bare == with_facts


def test_classifier_rejects_bad_mappings(tiny_model):
    from shishya.backends import ClassifierBackend
    with pytest.raises(BackendError, match="not in the model"):
        ClassifierBackend(tiny_model, fields={"news_impact": {"bad": "terrible"}}, device="cpu")
    with pytest.raises(BackendError, match="its own"):
        ClassifierBackend(tiny_model, fields={"news_impact": {"bad": "bearish", "good": "bearish"}}, device="cpu")


def test_cli_route_sends_the_field_to_the_classifier(tiny_model, tmp_path, capsys):
    import json

    from shishya.cli import main
    schema = tmp_path / "s.json"
    schema.write_text(json.dumps({"type": "object", "properties": {
        "news_impact": {"type": "string", "enum": ["bearish", "bullish", "neutral"]},
        "action": {"type": "string", "enum": ["hold", "trim"]}}}))
    code = main(["decide", str(schema), "--input", "auditor resigns", "--backend", "fake", "--json",
                 "--route", f"news_impact={tiny_model}"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["receipt"]["backend"]["name"] == "router"
    assert "news_impact=" in out["receipt"]["backend"]["model"]
    with pytest.raises(SystemExit):
        main(["decide", str(schema), "--input", "x", "--route"])        # argparse: missing value
    assert main(["decide", str(schema), "--input", "x", "--backend", "fake", "--route", "nofield"]) == 1
