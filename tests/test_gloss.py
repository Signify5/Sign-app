"""English -> ASL order rules in assets/hand/gloss.js, run under node."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

GLOSS_JS = Path(__file__).resolve().parent.parent / "assets" / "hand" / "gloss.js"

SIGNS = ["I", "YOU", "GO", "STORE", "EAT", "PIZZA", "FINISH", "PAST", "WILL", "MUST", "WORK", "YESTERDAY",
         "TOMORROW", "NEXT", "WEEK", "LAST WEEK", "WHAT", "WHERE", "HOW", "MANY", "YOUR", "NAME", "LIVE", "CAT",
         "DOG", "THREE", "WANT", "SICK", "NOT", "LIKE", "BOOK", "HAVE", "TWO", "DAY", "AGO", "MONDAY", "ON",
         "TO", "SCHOOL", "THANK YOU", "MOTHER", "CHILD"]
LEX = {
    "signs": {g: 1 for g in SIGNS},
    "alias": {"WENT": ["GO"], "ATE": ["EAT"], "CATS": ["CAT"], "DOGS": ["DOG"], "BOOKS": ["BOOK"], "WANTS": ["WANT"],
              "DAYS": ["DAY"], "WORKED": ["WORK"], "DIDN'T": ["NOT"], "FINISHED": ["FINISH"], "CHILDREN": ["CHILD"], "GOING": ["GO"],
              "THANKS": ["THANK YOU"], "EATEN": ["EAT"]},
    "tags": {"WENT": "PAST", "ATE": "PAST", "CATS": "S", "DOGS": "S", "BOOKS": "S", "WANTS": "S", "DAYS": "S",
             "WORKED": "ED", "FINISHED": "ED", "CHILDREN": "PL", "GOING": "ING", "EATEN": "PAST"},
    "nouns": ["STORE", "PIZZA", "CAT", "DOG", "BOOK", "DAY", "WEEK", "NAME", "SCHOOL", "MOTHER", "CHILD"],
    "drop": ["A", "AN", "THE", "IS", "AM", "ARE", "BE", "WAS", "WERE", "DO", "DOES", "DID", "AT", "OF"],
}

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def gloss(text: str, nlp=None, lex=LEX) -> dict:
    js = f"const {{toGloss}}=require({json.dumps(str(GLOSS_JS))});" \
         f"console.log(JSON.stringify(toGloss({json.dumps(text)},{json.dumps(lex)},{json.dumps(nlp)})))"
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def order(text: str, nlp=None, lex=LEX) -> str:
    return " ".join((x["gloss"] + "+" if x.get("rep") else x["gloss"]) if "gloss" in x else "#" + x["spell"]
                    for x in gloss(text, nlp, lex)["items"])


def nlp(*rows):
    """Hand-written spaCy readings: (word, lemma, pos, tag[, neg])."""
    return [{"word": r[0], "lemma": r[1], "pos": r[2], "tag": r[3], "neg": len(r) > 4 and r[4]} for r in rows]


def test_drops_english_only_words():
    assert order("I went to the store yesterday.") == "YESTERDAY I GO STORE"
    assert order("the") == "#THE"  # a lone word is never dropped


def test_time_first():
    assert order("I will go to school tomorrow") == "TOMORROW I GO SCHOOL"
    assert order("I am going to work next week") == "NEXT WEEK I WORK"
    assert order("I worked on Monday") == "MONDAY I WORK"
    assert order("I ate two days ago") == "TWO DAY AGO I EAT"


def test_past_tense():
    assert order("I ate pizza") == "I EAT PIZZA FINISH"
    assert order("I was sick") == "PAST I SICK"
    assert order("I didn't go") == "I NOT GO"
    assert order("I finished") == "I FINISH"


def test_future_and_obligation():
    assert order("I am going to eat") == "I WILL EAT"
    assert order("I have to work") == "I MUST WORK"
    assert order("I have eaten") == "I EAT FINISH" and order("I have two dogs") == "I HAVE TWO DOG"


def test_wh_questions():
    assert order("What is your name?") == "YOUR NAME WHAT"
    assert order("Where do you live?") == "YOU LIVE WHERE"
    assert order("How many cats do you have?") == "CAT YOU HAVE HOW MANY"


def test_plurals():
    assert order("I like cats") == "I LIKE CAT+"
    assert order("I want three dogs") == "I WANT THREE DOG"
    assert order("I want many books") == "I WANT MANY BOOK"
    assert order("Mother wants pizza") == "MOTHER WANT PIZZA"  # a verb's -s is not a plural
    assert order("children") == "CHILD+"


def test_report():
    r = gloss("Thanks, I ate the xyzzy.")
    assert r["spelled"] == ["XYZZY"] and "THE" in r["dropped"] and r["added"] == ["FINISH"]
    assert "thanks = THANK YOU" in r["swapped"]


def test_spacy_reading_decides_noun_or_verb():
    lex = dict(LEX, signs=dict(LEX["signs"], ROOM=1, HE=1), alias=dict(LEX["alias"], BOOKED=["BOOK"]),
               tags=dict(LEX["tags"], BOOKED="ED"))
    # the word list says BOOK is a noun, so BOOKS alone would be a plural; spaCy reads the verb here
    he_books = nlp(("HE", "HE", "PRON", "PRP"), ("BOOKS", "BOOK", "VERB", "VBZ"), ("A", "A", "DET", "DT"),
                   ("ROOM", "ROOM", "NOUN", "NN"))
    assert order("he books a room", he_books, lex) == "HE BOOK ROOM"
    assert order("he books a room", None, lex) == "HE BOOK+ ROOM"  # without spaCy
    my_books = nlp(("I", "I", "PRON", "PRP"), ("LIKE", "LIKE", "VERB", "VBP"), ("BOOKS", "BOOK", "NOUN", "NNS"))
    assert order("I like books", my_books, lex) == "I LIKE BOOK+"
    booked = nlp(("I", "I", "PRON", "PRP"), ("BOOKED", "BOOK", "VERB", "VBD"), ("A", "A", "DET", "DT"),
                 ("ROOM", "ROOM", "NOUN", "NN"))
    assert order("I booked a room", booked, lex) == "I BOOK ROOM FINISH"


def test_spacy_dictionary_form_fills_gaps_but_not_with_a_noun_sign_for_a_verb():
    lex = dict(LEX, signs=dict(LEX["signs"], BIG=1, FIRE=1, THEY=1), nouns=LEX["nouns"] + ["FIRE"])
    bigger = nlp(("CAT", "CAT", "NOUN", "NN"), ("BIGGER", "BIG", "ADJ", "JJR"))
    assert order("cat bigger", bigger, lex) == "CAT BIG"
    fired = nlp(("THEY", "THEY", "PRON", "PRP"), ("FIRED", "FIRE", "VERB", "VBD"), ("ME", "I", "PRON", "PRP"))
    assert "#FIRED" in order("they fired me", fired, lex).split()


def test_spacy_reading_out_of_step_is_ignored():
    assert order("I ate pizza", nlp(("SOMETHING", "ELSE", "NOUN", "NN"))) == "I EAT PIZZA FINISH"
