"""translate.analyze lines spaCy's tokens up with the words gloss.js reads (needs spaCy + en_core_web_sm)."""
import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from translate import analyze  # noqa: E402


def test_one_entry_per_word_gloss_js_reads():
    got = analyze("I don\u2019t think the children went home...")
    assert [x["word"] for x in got] == ["I", "DON'T", "THINK", "THE", "CHILDREN", "WENT", "HOME"]
    by = {x["word"]: x for x in got}
    assert by["DON'T"]["neg"] and not by["THINK"]["neg"]
    assert (by["CHILDREN"]["lemma"], by["CHILDREN"]["tag"]) == ("CHILD", "NNS")
    assert (by["WENT"]["lemma"], by["WENT"]["tag"]) == ("GO", "VBD")


def test_noun_or_verb_from_the_sentence():
    assert analyze("my books")[1]["pos"] == "NOUN"
    assert analyze("she books a table")[1]["pos"] == "VERB"
