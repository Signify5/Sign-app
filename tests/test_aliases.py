from aliases import DROP, inflections, lookup, resolve_aliases

SIGNS = {"GO", "MAKE", "STOP", "HOPE", "STUDY", "VISIT", "DIE", "HAPPY", "SIMPLE", "MOTHER", "FATHER",
         "DRIVE", "AGENT", "HELLO", "SHE", "WILL", "NOT", "THANK YOU", "AIRPLANE", "WE", "ON", "EVEN"}


def test_inflections():
    assert {"GOES", "GOING"} <= set(inflections("GO"))
    assert {"MAKES", "MAKING"} <= set(inflections("MAKE"))
    assert {"STOPPING", "STOPPED"} <= set(inflections("STOP"))
    assert "STOPING" not in inflections("STOP")
    assert {"STUDIES", "STUDIED", "STUDYING"} <= set(inflections("STUDY"))
    assert {"VISITED", "VISITING"} <= set(inflections("VISIT"))
    assert {"DYING", "DIED"} <= set(inflections("DIE"))
    assert "HAPPILY" in inflections("HAPPY") and "SIMPLY" in inflections("SIMPLE")


def test_false_forms_are_left_out():
    assert "WING" not in inflections("WE") and "TOES" not in inflections("TO")
    assert "ONLY" not in inflections("ON")
    assert "EVENING" not in inflections("EVEN")


def test_resolve():
    alias, notes, tags = resolve_aliases(SIGNS)
    assert alias["PARENTS"] == ["MOTHER", "FATHER"]
    assert alias["DRIVER"] == ["DRIVE", "AGENT"]
    assert alias["DRIVERS"] == ["DRIVE", "AGENT"]  # forms of an alias
    assert alias["WENT"] == ["GO"] and alias["HI"] == ["HELLO"] and alias["THANKS"] == ["THANK YOU"]
    assert alias["HE'S"] == ["SHE"]  # through HE
    assert alias["DIDN'T"] == ["NOT"] and "WON'T" not in alias  # WON'T needs REFUSE
    assert alias["HOPING"] == ["HOPE"] and alias["PLANES"] == ["AIRPLANE"]
    assert "GO" not in alias and "MOTHER" not in alias  # a sign keeps its own clip
    assert not set(DROP) & set(alias)
    assert any(n.startswith("TEACHER") for n in notes)  # TEACH is not in SIGNS
    assert tags["WENT"] == "PAST" and tags["HOPING"] == "ING" and tags["PLANES"] == "S" and tags["VISITED"] == "ED"


def test_lookup():
    alias, _, _ = resolve_aliases(SIGNS)
    assert lookup("GO", SIGNS, alias) == ["GO"]
    assert lookup("MOTHER'S", SIGNS, alias) == ["MOTHER"]
    assert lookup("GOING", SIGNS, alias) == ["GO"]
    assert lookup("XYZZY", SIGNS, alias) is None


def test_noun_signs_only_get_verb_forms_when_asl_signs_the_action_alike():
    alias, _, tags = resolve_aliases(SIGNS | {"FIRE", "BOOK"}, nouns={"FIRE", "BOOK", "STUDY", "AIRPLANE"})
    assert alias["FIRES"] == ["FIRE"] and alias["BOOKS"] == ["BOOK"]
    assert "FIRED" not in alias and "BOOKING" not in alias and "PLANING" not in alias
    assert alias["STUDIED"] == ["STUDY"]  # STUDY is listed in VERB_TOO
