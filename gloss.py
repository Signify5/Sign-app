"""English to ASL gloss translation"""

from future import annotations
import spacy

nlp = spacy.load("en_core_web_sm")

LEXICON: dict[str, str] = {
    "i":"I","me":"I","you":"YOU","he":"HE","she":"SHE","my":"MY","your": "YOUR", "friend": "FRIEND", "teacher": "TEACHER", "mom": "MOM","sister":"SISTER","brother":"BROTHER","grandmother":"GRANDMOTHER",
    "sandwich": "SANDWICH", "store": "STORE", "open": "OPEN", "party": "PARTY","go":"GO", "going":"GO", "call":"CALL","eat":"EAT","come":"COME","understand":"UNDERSTAND","like":"LIKE","want":"WANT", 
    "finish":"FINISH","help":"HELP","hello":"HELLO","hungry":"HUNGRY","name":"NAME","bathroom":"BATHROOM","home":"HOME","coffee":"COFFEE","pizza":"PIZZA","dinner":"DINNER","breakfast":"BREAKFAST","homework":"HOMEWORK",
    "school":"SCHOOL","this":"THIS",
}

TIME_WORDS = {"tomorrow", "today", "yesterday", "tonight", "now", "later", "soon", "week"}
def to_gloss(text: str) -> list[str]:
    """Translate a complete English sentence into the appropriate gloss"""
    doc = nlp(text)
    time_word: str | None = None
    wh_word: str | None = None
    has_negation = False
    kept_tokens: list[str] = []

    for token in doc:
        word = token.text.lower()

        if token.dep_ == "neg":
            has_negation = True
            continue

        if token.tag_ in ("WRB", "WDT", "WP"):
            wh_word = word
            continue

        if word in TIME_WORDS and time_word is None:
            time_word = word
            continue

        if token.pos_ == "AUX" or token.pos_ == "DET":
            continue

        if token.pos_ == "PUNCT":
            continue

        kept_tokens.append(word)

    glossed = [LEXICON.get(w, "FS:" + w.upper()) for w in kept_tokens]

    result: list[str] = []
    if time_word:
        result.append(time_word.upper())
    result.extend(glossed)
    if has_negation:
        result.append("NOT")
    if wh_word:
        result.append(wh_word.upper())

    return result

if __name__ == "__main__":
    tests = [
        ("I am hungry.",  ["I", "HUNGRY"]),
        ("Are you hungry?",  ["YOU", "HUNGRY"]),
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

    passed = 0
    for sentence, expected in tests:
        actual = to_gloss(sentence)
        ok = actual == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {sentence!r} -> {actual}")

    print(f"{passed}/{len(tests)} tests passed")
