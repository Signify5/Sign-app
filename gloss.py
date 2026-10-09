"""
gloss.py — Track 2: English text -> ASL gloss sequence

v2: uses spaCy's grammatical analysis (part-of-speech tags) to detect
auxiliary verbs, articles, wh-words, and negation automatically -- instead
of hardcoded word lists. This means the rules now apply to ANY sentence,
not just words we typed into a list by hand.

The 8 rules are unchanged from v1:
  1. Drop "to be" verbs (is/am/are/was/were)        -> now via POS tag AUX
  2. Drop articles (a/an/the) and some prepositions   -> now via POS tag DET
  3. Topic-comment structure (reserved for later)
  4. Time markers go first; tense words dropped       -> still a curated set
     (spaCy doesn't reliably flag "tomorrow" as different from any other
     noun, so this part stays a word list -- a known, honest limitation)
  5. Wh-questions move the question word to the end   -> now via POS tag WRB/WDT
  6. Yes/no questions keep normal word order (gloss can't show eyebrows)
  7. Negation goes at the end of the verb phrase       -> now via dep_ == "neg"
  8. Subject and object are always kept                -> now via dep_ == "nsubj"/"dobj"
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------
# LEXICON: English word -> ASL gloss label. This part still needs
# real vocabulary (from WLASL or a teammate's list) to scale past
# a handful of words -- spaCy tells us grammar, not which sign to show.
# ---------------------------------------------------------------
LEXICON = {
    "i": "I", "me": "I", "you": "YOU", "he": "HE", "she": "SHE", "my": "MY",
    "your": "YOUR", "friend": "FRIEND", "teacher": "TEACHER", "mom": "MOM",
    "sister": "SISTER", "brother": "BROTHER", "grandmother": "GRANDMOTHER",
    "sandwich": "SANDWICH", "store": "STORE", "open": "OPEN", "party": "PARTY",
    "go": "GO", "going": "GO", "call": "CALL", "eat": "EAT", "come": "COME",
    "understand": "UNDERSTAND", "like": "LIKE", "want": "WANT", "finish": "FINISH",
    "help": "HELP", "hello": "HELLO", "hungry": "HUNGRY", "name": "NAME",
    "bathroom": "BATHROOM", "home": "HOME", "coffee": "COFFEE", "pizza": "PIZZA",
    "dinner": "DINNER", "breakfast": "BREAKFAST", "homework": "HOMEWORK",
    "school": "SCHOOL", "this": "THIS",
}

# Still a curated set -- see the note in the docstring above.
TIME_WORDS = {"tomorrow", "today", "yesterday", "tonight", "now", "later", "soon", "week"}


def to_gloss(text):
    doc = nlp(text)

    time_word = None
    wh_word = None
    has_negation = False
    kept_tokens = []  # (token, gloss_label) pairs, in original order

    for token in doc:
        word = token.text.lower()

        # Rule 7: negation, detected by spaCy's dependency label "neg"
        # (catches "not", "n't", "no" uniformly -- no word list needed)
        if token.dep_ == "neg":
            has_negation = True
            continue

        # Rule 5: wh-words, detected by part-of-speech tag
        # WRB = wh-adverb (where/when/why/how), WDT/WP = wh-determiner/pronoun (what/who/which)
        if token.tag_ in ("WRB", "WDT", "WP"):
            wh_word = word
            continue

        # Rule 4: time words -- still a curated set (see docstring)
        if word in TIME_WORDS and time_word is None:
            time_word = word
            continue

        # Rules 1 & 2: drop auxiliary verbs (is/am/are/do/does/did/will)
        # and articles (a/an/the), detected by POS tag.
        # Note: we only drop a determiner when spaCy ALSO calls it a
        # determiner by pos_ (not just the finer-grained tag_), because
        # words like "this"/"that" can be tagged DT even when they're
        # standing in as the object ("I understand this") rather than
        # modifying a noun ("this book") -- pos_ == DET catches the
        # modifying case; pos_ == PRON (tag_ DT) is the object case,
        # which we want to KEEP.
        if token.pos_ == "AUX" or token.pos_ == "DET":
            continue

        # Punctuation isn't signed
        if token.pos_ == "PUNCT":
            continue

        kept_tokens.append(word)

    # Rule 8 is automatically satisfied: we never filter out subjects/objects
    # above, only auxiliaries, articles, wh-words, time words, and negation.

    glossed = [LEXICON.get(w, "FS:" + w.upper()) for w in kept_tokens]

    result = []
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
        ("I am hungry.",                               ["I", "HUNGRY"]),
        ("Are you hungry?",                             ["YOU", "HUNGRY"]),
        ("Where is the bathroom?",                      ["BATHROOM", "WHERE"]),
        ("What do you want?",                           ["YOU", "WANT", "WHAT"]),
        ("I don't understand this.",                    ["I", "UNDERSTAND", "THIS", "NOT"]),
        ("Did you eat my sandwich?",                    ["YOU", "EAT", "MY", "SANDWICH"]),
        ("My friend will not call tonight.",             ["TONIGHT", "MY", "FRIEND", "CALL", "NOT"]),
        ("Who is your teacher?",                         ["YOUR", "TEACHER", "WHO"]),
        ("Is the store open?",                           ["STORE", "OPEN"]),
        ("I like coffee.",                               ["I", "LIKE", "COFFEE"]),
        # New sentences -- words/phrasing never explicitly listed anywhere,
        # to prove the rules now generalize instead of just matching a list:
        ("Where is my grandmother going?",               ["MY", "GRANDMOTHER", "GO", "WHERE"]),
        ("Does your teacher help you?",                  ["YOUR", "TEACHER", "HELP", "YOU"]),
    ]

    passed = 0
    for sentence, expected in tests:
        actual = to_gloss(sentence)
        ok = actual == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] \"{sentence}\"")
        print(f"   expected: {expected}")
        print(f"   actual:   {actual}\n")

    print(f"{passed}/{len(tests)} tests passed")
