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


def gloss(text: str) -> dict:
    js = f"const {{toGloss}}=require({json.dumps(str(GLOSS_JS))});" \
         f"console.log(JSON.stringify(toGloss({json.dumps(text)},{json.dumps(LEX)})))"
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def order(text: str) -> str:
    return " ".join((x["gloss"] + "+" if x.get("rep") else x["gloss"]) if "gloss" in x else "#" + x["spell"]
                    for x in gloss(text)["items"])


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
