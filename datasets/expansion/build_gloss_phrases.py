from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "gloss_to_asl" / "assets" / "hand" / "gloss_rules.json"
GLOSS_JS = ROOT / "gloss_to_asl" / "assets" / "hand" / "gloss.js"
OUT_JSON = ROOT / "gloss_to_asl" / "assets" / "hand" / "phrases.json"
OUT_PY = ROOT / "text_to_gloss" / "phrases.json"

MARKER_START = "  /* PHRASES_START */"
MARKER_END = "  /* PHRASES_END */"


def main() -> None:
    data = json.loads(RULES.read_text(encoding="ascii"))
    phrases = []
    for p in data["phrases"]:
        phrases.append({"en": p["en"].split(), "gloss": p["gloss"], "id": p["id"]})
    phrases.sort(key=lambda p: -len(p["en"]))
    payload = {
        "phrases": phrases,
        "grammar_tests": data["grammar"],
        "phrase_tests": [
            {"id": p["id"], "test": p["test"], "expect": p["expect"]} for p in data["phrases"]
        ],
    }
    text = json.dumps(payload, indent=2) + "\n"
    OUT_JSON.write_text(text, encoding="ascii")
    OUT_PY.write_text(text, encoding="ascii")

    js_arr = json.dumps(phrases, separators=(",", ":"))
    block = (
        f"{MARKER_START}\n"
        f"  const PHRASES = {js_arr};\n"
        f"{MARKER_END}"
    )
    src = GLOSS_JS.read_text(encoding="utf-8")
    if MARKER_START in src and MARKER_END in src:
        a = src.index(MARKER_START)
        b = src.index(MARKER_END) + len(MARKER_END)
        src = src[:a] + block + src[b:]
    else:
        needle = "  const PAST_AUX = new Set(['WAS', 'WERE', 'DID']);"
        if needle not in src:
            raise SystemExit("cannot find insertion point in gloss.js")
        src = src.replace(
            needle,
            needle + "\n\n" + block + "\n\n"
            "  function applyPhrases(w) {\n"
            "    const out = [];\n"
            "    let i = 0;\n"
            "    while (i < w.length) {\n"
            "      let hit = null;\n"
            "      for (const p of PHRASES) {\n"
            "        const n = p.en.length;\n"
            "        if (n && i + n <= w.length && w.slice(i, i + n).every((x, k) => x === p.en[k])) {\n"
            "          hit = p; break;\n"
            "        }\n"
            "      }\n"
            "      if (hit) { out.push(...hit.gloss); i += hit.en.length; }\n"
            "      else { out.push(w[i]); i++; }\n"
            "    }\n"
            "    return out.filter(Boolean);\n"
            "  }\n",
        )
        src = src.replace(
            "      const w = words(sentence), noPast = question || WH.has(words(sentence)[0]) || AUX.has(words(sentence)[0]), info = nlp ? nlp.slice(wi, wi + w.length) : [];\n"
            "      wi += w.length;",
            "      const raw = words(sentence), w = applyPhrases(raw), noPast = question || WH.has(raw[0]) || AUX.has(raw[0]), "
            "info = (nlp && w.length === raw.length) ? nlp.slice(wi, wi + raw.length) : [];\n"
            "      wi += raw.length;",
        )
    GLOSS_JS.write_text(src, encoding="utf-8")
    print("phrases", len(phrases), "total rules", len(phrases) + len(data["grammar"]))
    print("updated", GLOSS_JS)


if __name__ == "__main__":
    main()
