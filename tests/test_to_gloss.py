"""Step 2's own examples (gloss.py's __main__), run as tests. Needs spaCy and en_core_web_sm."""
from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")
try:
    from gloss import to_gloss
except OSError:  # spaCy installed without its English model
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

EXAMPLES = [
    ("I am hungry.", ["I", "HUNGRY"]),
    ("Are you hungry?", ["YOU", "HUNGRY"]),
    ("Where is the bathroom?", ["BATHROOM", "WHERE"]),
    ("What do you want?", ["YOU", "WANT", "WHAT"]),
    ("I don't understand this.", ["I", "UNDERSTAND", "THIS", "NOT"]),
    ("Did you eat my sandwich?", ["YOU", "EAT", "MY", "SANDWICH"]),
    ("My friend will not call tonight.", ["TONIGHT", "MY", "FRIEND", "CALL", "NOT"]),
    ("Who is your teacher?", ["YOUR", "TEACHER", "WHO"]),
    ("Is the store open?", ["STORE", "OPEN"]),
    ("I like coffee.", ["I", "LIKE", "COFFEE"]),
    ("Where is my grandmother going?", ["MY", "GRANDMOTHER", "GO", "WHERE"]),
    ("Does your teacher help you?", ["YOUR", "TEACHER", "HELP", "YOU"]),
]


@pytest.mark.parametrize("sentence,expected", EXAMPLES)
def test_examples(sentence, expected):
    assert to_gloss(sentence) == expected


def test_dictionary_form_before_fingerspelling():
    assert to_gloss("I ate pizza") == ["I", "EAT", "PIZZA"]
    assert to_gloss("my friends called") == ["MY", "FRIEND", "CALL"]
    assert to_gloss("I like xylophones") == ["I", "LIKE", "FS:XYLOPHONES"]
