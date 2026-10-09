"""English words that play signs the library already has.

Three tables, checked against the vocabulary when the library page is built:
- SAME_SIGN: groups of words ASL signs the same way; a missing word plays the first member that has a sign.
- ALIASES: a word or contraction to one or more glosses (irregular verbs and plurals, contractions,
  compounds such as PARENTS = MOTHER FATHER and person nouns such as DRIVER = DRIVE AGENT).
- DROP: English words ASL leaves out when they have no sign of their own.
Every sign and alias also gets its regular forms (-s, -ing, -ed, -ly) from `inflections`; the page
strips a possessive 's itself.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SAME_SIGN: List[Tuple[str, ...]] = [
    # greetings and answers
    ("HELLO", "HI", "HEY", "HOWDY"),
    ("YES", "YEAH", "YEP", "YUP"),
    ("NO", "NOPE", "NAH"),
    ("OK", "OKAY", "ALRIGHT"),
    ("BYE", "GOODBYE"),
    # family
    ("MOM", "MOMMY", "MUM", "MA"),
    ("DAD", "DADDY", "PAPA", "PA"),
    ("GRANDMA", "GRANDMOTHER", "GRANNY", "NANA"),
    ("GRANDPA", "GRANDFATHER", "GRANDDAD"),
    ("KID", "CHILD"),
    ("BABY", "INFANT"),
    # pronouns (pointing signs)
    ("SHE", "HE", "HIM", "IT"),
    ("HIS", "HER", "ITS", "HERS"),
    ("WE", "US"),
    ("THEY", "THEM"),
    ("THEIR", "THEIRS"),
    ("MY", "MINE"),
    ("YOUR", "YOURS"),
    # people
    ("MAN", "GUY"),
    ("WOMAN", "LADY"),
    ("DOCTOR", "PHYSICIAN"),
    ("LAWYER", "ATTORNEY"),
    ("POLICE", "COP", "COPS"),
    ("BOSS", "CHIEF"),
    ("FRIEND", "BUDDY", "PAL"),
    ("STUDENT", "LEARNER", "PUPIL"),
    ("WRITER", "AUTHOR"),
    # things and places
    ("AIRPLANE", "PLANE", "AEROPLANE"),
    ("CAR", "AUTO", "AUTOMOBILE"),
    ("BICYCLE", "BIKE"),
    ("TELEVISION", "TV"),
    ("PHONE", "TELEPHONE"),
    ("BATHROOM", "RESTROOM", "WASHROOM", "TOILET", "LAVATORY"),
    ("STORE", "SHOP"),
    ("CITY", "TOWN", "COMMUNITY"),
    ("ROAD", "STREET", "WAY", "PATH"),
    ("PICTURE", "PHOTO", "PHOTOGRAPH", "IMAGE"),
    ("MOVIE", "FILM", "CINEMA"),
    ("COUCH", "SOFA"),
    ("TRASH", "GARBAGE", "RUBBISH"),
    ("MONEY", "CASH"),
    # verbs
    ("START", "BEGIN"),
    ("BUY", "PURCHASE"),
    ("HELP", "ASSIST"),
    ("FIX", "REPAIR"),
    ("ANSWER", "REPLY", "RESPOND"),
    ("YELL", "SHOUT"),
    ("UNDERSTAND", "COMPREHEND"),
    ("FINISH", "DONE"),
    ("FAVORITE", "FAVOURITE", "PREFER"),  # the same sign; a spelling variant plays its own spelling first
    # describing words
    ("HAPPY", "GLAD"),
    ("BIG", "LARGE"),
    ("SMART", "INTELLIGENT", "CLEVER"),
    ("SICK", "ILL"),
    ("BEAUTIFUL", "PRETTY", "LOVELY"),
    ("FAST", "QUICK"),
    ("TRUE", "REAL", "REALLY", "SURE", "TRULY"),
    ("RIGHT", "CORRECT"),
    ("COLOR", "COLOUR"),
    ("GRAY", "GREY"),
    ("CENTER", "CENTRE"),
    ("THEATER", "THEATRE"),
]

# Irregular past forms; the page marks them as past tense (WENT -> GO, with FINISH or a time sign).
PAST: Dict[str, Tuple[str, ...]] = {
    "WENT": ("GO",), "GONE": ("GO",),
    "HAD": ("HAVE",),
    "MADE": ("MAKE",), "SAW": ("SEE",), "SEEN": ("SEE",),
    "ATE": ("EAT",), "EATEN": ("EAT",), "CAME": ("COME",),
    "GAVE": ("GIVE",), "GIVEN": ("GIVE",), "TOOK": ("TAKE",), "TAKEN": ("TAKE",),
    "GOT": ("GET",), "GOTTEN": ("GET",), "SAID": ("SAY",), "TOLD": ("TELL",),
    "KNEW": ("KNOW",), "KNOWN": ("KNOW",), "THOUGHT": ("THINK",),
    "BOUGHT": ("BUY",), "BROUGHT": ("BRING",), "TAUGHT": ("TEACH",), "FOUND": ("FIND",),
    "FELT": ("FEEL",), "KEPT": ("KEEP",), "SLEPT": ("SLEEP",), "MET": ("MEET",),
    "PAID": ("PAY",), "SOLD": ("SELL",), "SENT": ("SEND",), "SPENT": ("SPEND",),
    "BUILT": ("BUILD",), "WROTE": ("WRITE",), "WRITTEN": ("WRITE",),
    "DROVE": ("DRIVE",), "DRIVEN": ("DRIVE",), "RAN": ("RUN",), "SAT": ("SIT",),
    "STOOD": ("STAND",), "SPOKE": ("SPEAK",), "SPOKEN": ("SPEAK",),
    "BEGAN": ("BEGIN",), "BEGUN": ("BEGIN",),
    "FORGOT": ("FORGET",), "FORGOTTEN": ("FORGET",), "CHOSE": ("CHOOSE",), "CHOSEN": ("CHOOSE",),
    "WON": ("WIN",), "FELL": ("FALL",), "FALLEN": ("FALL",), "FLEW": ("FLY",), "FLOWN": ("FLY",),
    "GREW": ("GROW",), "GROWN": ("GROW",), "THREW": ("THROW",), "THROWN": ("THROW",),
    "DRANK": ("DRINK",), "SANG": ("SING",), "SUNG": ("SING",),
    "SWAM": ("SWIM",), "WORE": ("WEAR",), "WORN": ("WEAR",), "BROKEN": ("BREAK",),
    "HELD": ("HOLD",), "HEARD": ("HEAR",), "UNDERSTOOD": ("UNDERSTAND",),
    "CAUGHT": ("CATCH",), "FOUGHT": ("FIGHT",), "WOKE": ("WAKE UP",),
    "DRAWN": ("DRAW",), "DREW": ("DRAW",), "RODE": ("RIDE",), "RIDDEN": ("RIDE",),
    "STOLE": ("STEAL",), "STOLEN": ("STEAL",), "HID": ("HIDE",), "HIDDEN": ("HIDE",),
    "BITTEN": ("BITE",), "SHOOK": ("SHAKE",), "FED": ("FEED",),
    "LED": ("LEAD",), "LENT": ("LEND",), "LOST": ("LOSE",), "TORE": ("TEAR",),
    "BLEW": ("BLOW",), "DUG": ("DIG",), "FROZE": ("FREEZE",), "FROZEN": ("FREEZE",),
}

# Irregular plurals; the page treats them like CATS (MEN -> MAN, repeated unless a number or MANY says how many).
# HAS and PERSONS are plain aliases below: HAS is not past, PEOPLE is already plural.
PLURAL: Dict[str, Tuple[str, ...]] = {
    "CHILDREN": ("CHILD",), "MEN": ("MAN",), "WOMEN": ("WOMAN",), "FEET": ("FOOT",),
    "TEETH": ("TOOTH",), "MICE": ("MOUSE",), "GEESE": ("GOOSE",),
}

ALIASES: Dict[str, Tuple[str, ...]] = {
    "THANKS": ("THANK YOU",),
    "THX": ("THANK YOU",),
    "GOODNIGHT": ("GOOD", "NIGHT"),
    "GOOD NIGHT": ("GOOD", "NIGHT"),
    "FRIES": ("FRENCH FRIES",),
    # contractions
    "I'M": ("I",), "YOU'RE": ("YOU",), "WE'RE": ("WE",), "THEY'RE": ("THEY",),
    "HE'S": ("HE",), "SHE'S": ("SHE",), "IT'S": ("IT",), "THAT'S": ("THAT",),
    "WHAT'S": ("WHAT",), "WHERE'S": ("WHERE",), "WHO'S": ("WHO",), "HOW'S": ("HOW",),
    "WHEN'S": ("WHEN",), "WHY'S": ("WHY",), "THERE'S": ("THERE",), "HERE'S": ("HERE",),
    "I'VE": ("I",), "YOU'VE": ("YOU",), "WE'VE": ("WE",), "THEY'VE": ("THEY",),
    "I'D": ("I",), "YOU'D": ("YOU",), "WE'D": ("WE",), "THEY'D": ("THEY",),
    "I'LL": ("I", "WILL"), "YOU'LL": ("YOU", "WILL"), "WE'LL": ("WE", "WILL"),
    "THEY'LL": ("THEY", "WILL"), "HE'LL": ("HE", "WILL"), "SHE'LL": ("SHE", "WILL"),
    "HAS": ("HAVE",), "PERSONS": ("PEOPLE",),
    "LATELY": ("RECENT",), "NEARLY": ("ALMOST",),
    "DON'T": ("NOT",), "DOESN'T": ("NOT",), "DIDN'T": ("NOT",), "ISN'T": ("NOT",),
    "AREN'T": ("NOT",), "WASN'T": ("NOT",), "WEREN'T": ("NOT",), "AIN'T": ("NOT",),
    "HAVEN'T": ("NOT",), "HASN'T": ("NOT",), "HADN'T": ("NOT",),
    "CAN'T": ("CANNOT",), "COULDN'T": ("CANNOT",), "CANT": ("CANNOT",),
    "WON'T": ("REFUSE",), "WOULDN'T": ("NOT",),
    "SHOULDN'T": ("SHOULD", "NOT"), "MUSTN'T": ("MUST", "NOT"),
    "DONT": ("NOT",), "DOESNT": ("NOT",), "DIDNT": ("NOT",), "ISNT": ("NOT",), "WONT": ("REFUSE",),
    "IM": ("I",),
    # compounds
    "HUSBAND": ("MAN", "MARRY"),
    "WIFE": ("WOMAN", "MARRY"),
    "SON": ("BOY", "BABY"),
    "DAUGHTER": ("GIRL", "BABY"),
    "PARENTS": ("MOTHER", "FATHER"),
    "GIRLFRIEND": ("GIRL", "FRIEND"),
    "BOYFRIEND": ("BOY", "FRIEND"),
    "BREAKFAST": ("EAT", "MORNING"),
    "LUNCH": ("EAT", "NOON"),
    "DINNER": ("EAT", "NIGHT"),
    "SUPPER": ("EAT", "NIGHT"),
    "TONIGHT": ("NOW", "NIGHT"),
    "HOMEWORK": ("HOME", "WORK"),
    "BEDROOM": ("BED", "ROOM"),
    "CLASSROOM": ("CLASS", "ROOM"),
    "RAINCOAT": ("RAIN", "COAT"),
    # person nouns: verb or subject + the agent ending (both flat hands moving down the sides; built by
    # library.py from the PERSON sign's movement, since WLASL has no clip of the ending alone)
    "TEACHER": ("TEACH", "AGENT"),
    "WORKER": ("WORK", "AGENT"),
    "DRIVER": ("DRIVE", "AGENT"),
    "WRITER": ("WRITE", "AGENT"),
    "READER": ("READ", "AGENT"),
    "SINGER": ("SING", "AGENT"),
    "DANCER": ("DANCE", "AGENT"),
    "PLAYER": ("PLAY", "AGENT"),
    "RUNNER": ("RUN", "AGENT"),
    "SWIMMER": ("SWIM", "AGENT"),
    "BAKER": ("BAKE", "AGENT"),
    "COOK": ("COOK", "AGENT"),
    "LISTENER": ("LISTEN", "AGENT"),
    "HUNTER": ("HUNT", "AGENT"),
    "FIGHTER": ("FIGHT", "AGENT"),
    "SPEAKER": ("SPEAK", "AGENT"),
    "HELPER": ("HELP", "AGENT"),
    "WINNER": ("WIN", "AGENT"),
    "LEADER": ("LEAD", "AGENT"),
    "FOLLOWER": ("FOLLOW", "AGENT"),
    "BUILDER": ("BUILD", "AGENT"),
    "CLEANER": ("CLEAN", "AGENT"),
    "SELLER": ("SELL", "AGENT"),
    "SHOPPER": ("SHOP", "AGENT"),
    "SMOKER": ("SMOKE", "AGENT"),
    "SKIER": ("SKI", "AGENT"),
    "SKATER": ("SKATE", "AGENT"),
    "RIDER": ("RIDE", "AGENT"),
    "TRAVELER": ("TRAVEL", "AGENT"),
    "TRAVELLER": ("TRAVEL", "AGENT"),
    "VISITOR": ("VISIT", "AGENT"),
    "VOTER": ("VOTE", "AGENT"),
    "USER": ("USE", "AGENT"),
    "PLANNER": ("PLAN", "AGENT"),
    "PROGRAMMER": ("PROGRAM", "AGENT"),
    "RESEARCHER": ("RESEARCH", "AGENT"),
    "SUPPORTER": ("SUPPORT", "AGENT"),
    "THINKER": ("THINK", "AGENT"),
    "TYPIST": ("TYPE", "AGENT"),
    "WAITER": ("SERVE", "AGENT"),
    "WAITRESS": ("SERVE", "AGENT"),
    "SERVER": ("SERVE", "AGENT"),
    "KILLER": ("KILL", "AGENT"),
    "CLIMBER": ("CLIMB", "AGENT"),
    "DESIGNER": ("DESIGN", "AGENT"),
    "ORGANIZER": ("ORGANIZE", "AGENT"),
    "MANAGER": ("MANAGE", "AGENT"),
    "PAINTER": ("PAINT", "AGENT"),
    "ACTOR": ("ACT", "AGENT"),
    "ACTRESS": ("ACT", "AGENT"),
    "INTERPRETER": ("INTERPRET", "AGENT"),
    "PHOTOGRAPHER": ("CAMERA", "AGENT"),
    "FARMER": ("FARM", "AGENT"),
    "GARDENER": ("GARDEN", "AGENT"),
    "BANKER": ("BANK", "AGENT"),
    "ARTIST": ("ART", "AGENT"),
    "MUSICIAN": ("MUSIC", "AGENT"),
    "PIANIST": ("PIANO", "AGENT"),
    "LIBRARIAN": ("LIBRARY", "AGENT"),
    "SCIENTIST": ("SCIENCE", "AGENT"),
    "PSYCHOLOGIST": ("PSYCHOLOGY", "AGENT"),
    "AMERICAN": ("AMERICA", "AGENT"),
    "CANADIAN": ("CANADA", "AGENT"),
    "MEXICAN": ("MEXICO", "AGENT"),
    "AFRICAN": ("AFRICA", "AGENT"),
    "EUROPEAN": ("EUROPE", "AGENT"),
    "ITALIAN": ("ITALY", "AGENT"),
    "AUSTRALIAN": ("AUSTRALIA", "AGENT"),
}

ALIASES.update(PAST)
ALIASES.update(PLURAL)

DROP = ["A", "AN", "THE", "IS", "AM", "ARE", "BE", "WAS", "WERE", "BEEN", "BEING",
        "OF", "AT", "DO", "DOES", "DID", "LET'S", "LETS"]

# Generated forms that are really other words (EVENING is not EVEN + ING, ONLY is not ON + LY, TOES is not
# TO + ES) or mean something else (LATELY is not LATE, HEARTED is not HEART; WLASL's MEAN may be the cruel one,
# so not MEANING). Found by checking every generated form that is a common English word against WordNet's
# lemmas, then by hand.
NOT_INFLECTED = {
    "ALLS", "ALLY", "APPLY", "ARMED", "AWAYS", "BALLY", "BARELY", "BELLY", "BESIDES", "BIGS", "BLUES", "BODILY",
    "BROTHERLY", "BUTS", "BUTTED", "BUTTING", "CALLY", "CARLY", "COMMONS", "COSTLY", "COURTLY", "DEADLY", "DEATHLY",
    "DOLLY", "DOWNS", "EARED", "EARLY", "EARTHLY", "EASTERLY", "ELSES", "ENGINED", "ENVELOPED", "ENVELOPING",
    "EVENING", "FATHERLY", "FILMED", "FILMING", "FOOTBALLING", "FORMERLY", "FOUNDED", "FOUNDER", "FOUNDING",
    "FRIENDLY", "GHOSTLY", "GODLY", "GOLDING", "GOODING", "GOODS", "GOTS", "HAIRED", "HARDING", "HARDLY", "HEARTED",
    "HEAVENLY", "HED", "HERRING", "HILLY", "HING", "HISSED", "HISSING", "HOMELY", "HOTS", "HOURLY", "HOUSING",
    "HOWS", "IFS", "IMS", "ITALY", "KIDDING", "LARGELY", "LATELY", "LESSING", "LIKELY", "LIPPED", "LIVELY", "MADS",
    "MAJORLY", "MANLY", "MEANING", "MECHANICALLY", "MED", "METS", "MINES", "MING", "MINING", "MOTHERLY", "NAMELY",
    "NEARLY", "NEWLY", "NEWS", "NIGHTLY", "NOTS", "NOTTING", "ODDS", "OLDS", "ONLY", "ONS", "ORDERLY", "OVERLY",
    "PARTLY", "PASTED", "PASTING", "PEARLY", "PIED", "PRIESTLY", "PRINCELY", "PRINCIPLED", "PURPOSELY", "QUARTERLY",
    "READILY", "REDDING", "SALARIED", "SECRETED", "SEED", "SHAPELY", "SHED", "SHING", "SHORTLY", "SHORTS", "SICKLY",
    "SINGLY", "SMALLING", "SOMETHINGS", "STORIED", "TALLY", "TEETHING", "TELLY", "THESES", "THROATED", "TIMELY",
    "TIMING", "TIRED", "TOED", "TOES", "TOOTHED", "TOS", "TRAINING", "VERILY", "WALLY", "WED", "WEDDING", "WES",
    "WILLY", "WING", "WOKING", "WOMANLY", "WOODED", "WORLDLY", "YEARLY", "YOUS",
}
# Noun signs (assets/hand/nouns.txt) only get -ed and -ing forms when ASL signs the action the same way
# (STUDIED, RAINING). Otherwise the form usually means something else: FIRED is not flames, BOOKED is not a book,
# HEADED is not a head. Picked by hand from every such form that is a common English word.
VERB_TOO = {
    "AWARD", "BALANCE", "BARK", "BATH", "BATTLE", "BENEFIT", "BIKE", "BLOW", "BULLY", "BUTTER", "CAMP", "COACH",
    "COLOR", "COMB", "COMMAND", "COMMENT", "CONFLICT", "CONTACT", "CONTROL", "COPY", "COST", "COUNSEL", "DAMAGE",
    "DATE", "DEBATE", "DIVORCE", "DOUBT", "DREAM", "DRUM", "END", "EXCHANGE", "EXERCISE", "EXPERIENCE",
    "EXPERIMENT", "FEAR", "FISH", "FLOOD", "FOOL", "GOLF", "GOSSIP", "HAMMER", "INFLUENCE", "INTEREST", "IRON",
    "JOKE", "LEAK", "LECTURE", "LICENSE", "LIMIT", "LIST", "LOAN", "MESSAGE", "MILK", "MURDER", "NAME", "ORDER",
    "PARADE", "PARTY", "PHONE", "PLANT", "POOP", "PRACTICE", "PROCESS", "PROFIT", "PROGRESS", "QUESTION", "RACE",
    "RAIN", "RESEARCH", "RESPECT", "REST", "RUIN", "SALT", "SCORE", "SHAME", "SHOCK", "SHOVEL", "SHOWER", "SKATE",
    "SKETCH", "SLICE", "SNACK", "SNOW", "SPELL", "SPRAY", "STRESS", "STRUGGLE", "STUDY", "TASTE", "TELEPHONE",
    "TEXT", "THRILL", "TRADE", "TROUBLE", "TYPE", "VACATION", "VALUE", "WATER",
}
VOWELS = set("AEIOU")


def _syllables(w: str) -> int:
    return sum(1 for k, c in enumerate(w) if c in VOWELS and (k == 0 or w[k - 1] not in VOWELS))


def _cvc(w: str) -> bool:
    """Ends consonant-vowel-consonant, where English doubles the last letter (STOP -> STOPPED)."""
    return len(w) >= 3 and w[-1] not in VOWELS and w[-1] not in "WXY" and w[-2] in VOWELS and w[-3] not in VOWELS


def inflections(word: str) -> List[str]:
    """Regular English forms of a word: plural or -s, -ing, -ed and -ly. Over-generating is harmless
    (nobody types HELLOED); forms that are really other words are listed in NOT_INFLECTED."""
    w = word
    if len(w) < 2 or not w.isalpha():
        return []
    out: List[str] = []
    cons_y = w.endswith("Y") and w[-2] not in VOWELS
    # -s
    if cons_y:
        out.append(w[:-1] + "IES")
    elif w.endswith(("S", "X", "Z", "CH", "SH")):
        out.append(w + "ES")
    elif w.endswith("O"):
        out += [w + "S", w + "ES"]
    else:
        out.append(w + "S")
    # -ing and -ed
    if w.endswith("IE"):
        stems_ing, stems_ed = [w[:-2] + "Y"], [w[:-1]]
    elif w.endswith("E") and not w.endswith(("EE", "YE", "OE")):
        stems_ing, stems_ed = [w[:-1]], [w[:-1]]
    elif w.endswith("E"):
        stems_ing, stems_ed = [w], [w[:-1]]
    elif cons_y:
        stems_ing, stems_ed = [w], [w[:-1] + "I"]
    elif _cvc(w):
        doubled = w + w[-1]
        stems_ing = stems_ed = [doubled] if _syllables(w) == 1 else [w, doubled]
    else:
        stems_ing, stems_ed = [w], [w]
    out += [s + "ING" for s in stems_ing] + [s + "ED" for s in stems_ed]
    # -ly
    if cons_y:
        out.append(w[:-1] + "ILY")
    elif w.endswith("LE"):
        out.append(w[:-1] + "Y")
    elif w.endswith("IC"):
        out.append(w + "ALLY")
    elif w.endswith("LL"):
        out.append(w + "Y")
    else:
        out.append(w + "LY")
    return [f for f in out if f not in NOT_INFLECTED]


def _suffix_tag(form: str) -> str:
    return next((t for t in ("ING", "ED", "LY") if form.endswith(t)), "S")


def resolve_aliases(vocab: Iterable[str], nouns: Iterable[str] = ()
                    ) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
    """Words that play existing signs, as {word: [gloss, ...]}, a note per entry that was left out, and the
    grammar tag of each inflected word: S, ING, ED, LY (regular endings), PAST or PL (irregular forms).

    A word that already has its own sign keeps it, and a listed alias beats a generated form. Alias targets
    may name another alias (HE'S -> HE -> SHE). Signs in `nouns` get only the plural unless listed in VERB_TOO.
    """
    nouns = set(nouns)
    signs = set(vocab)
    table: Dict[str, Tuple[str, ...]] = {}
    for group in SAME_SIGN:
        base = next((g for g in group if g in signs), None)
        for g in group:
            if g not in signs and base:
                table.setdefault(g, (base,))
    for k, v in ALIASES.items():
        table[k] = v
    out: Dict[str, List[str]] = {}
    notes: List[str] = []

    def expand(k: str, seen: Tuple[str, ...]) -> Optional[List[str]]:
        if k in signs:
            return [k]
        if k in seen or k not in table:
            return None
        parts = [expand(g, seen + (k,)) for g in table[k]]
        return None if any(p is None for p in parts) else [g for p in parts for g in p]

    for k in table:
        if k in signs:
            continue
        r = expand(k, ())
        if r is None:
            notes.append(f"{k}: no sign for {' '.join(g for g in table[k] if expand(g, (k,)) is None)}")
        else:
            out[k] = r
    tags = {k: "PAST" for k in PAST if k in out}
    tags.update({k: "PL" for k in PLURAL if k in out})
    skip = signs | set(out) | set(DROP)
    for base in sorted(g for g in signs if " " not in g) + sorted(k for k in out if k.isalpha()):
        target = [base] if base in signs else out[base]
        noun_only = target[-1] in nouns and target[-1] not in VERB_TOO
        for form in inflections(base):
            if form not in skip and not (noun_only and _suffix_tag(form) != "S"):
                out[form] = target
                tags[form] = _suffix_tag(form)
                skip.add(form)
    return out, notes, tags


def lookup(word: str, signs: Sequence[str] | set, alias: Dict[str, List[str]]) -> Optional[List[str]]:
    """The glosses one English word plays, or None. Mirrors the page's tokenizer for a single word."""
    for w in (word, word[:-2] if word.endswith("'S") else None):
        if w in signs:
            return [w]
        if w in alias:
            return alias[w]
    return None
