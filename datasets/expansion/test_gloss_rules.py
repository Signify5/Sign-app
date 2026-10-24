from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHRASES = ROOT / "gloss_to_asl" / "assets" / "hand" / "phrases.json"
GLOSS_JS = ROOT / "gloss_to_asl" / "assets" / "hand" / "gloss.js"
PACK = ROOT / "gloss_to_asl" / "assets" / "signs.json.gz"


def run_node_tests() -> tuple[int, int, list[str]]:
    data = json.loads(PHRASES.read_text(encoding="ascii"))
    pack = json.load(gzip.open(PACK, "rt", encoding="utf-8"))
    lex = {
        "signs": {k: True for k in pack["signs"]},
        "alias": pack["alias"],
        "tags": pack.get("tags", {}),
        "nouns": pack.get("nouns", []),
        "drop": pack.get("drop", []),
    }
    tests = data["phrase_tests"] + [
        {"id": g["id"], "test": g["test"], "expect": g["expect"]} for g in data["grammar_tests"]
    ]
    runner = f"""
const {{toGloss}} = require({json.dumps(str(GLOSS_JS).replace(chr(92), '/'))});
const lex = {json.dumps(lex)};
const tests = {json.dumps(tests)};
let pass = 0, fail = 0;
const fails = [];
for (const t of tests) {{
  const got = toGloss(t.test, lex, null).items.map(x => x.gloss || ('FS:' + x.spell));
  const ok = JSON.stringify(got) === JSON.stringify(t.expect);
  if (ok) pass++; else {{ fail++; fails.push(t.id + ' | ' + t.test + ' | got ' + JSON.stringify(got) + ' expect ' + JSON.stringify(t.expect)); }}
}}
console.log(JSON.stringify({{pass, fail, fails}}));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="ascii") as fh:
        fh.write(runner)
        path = fh.name
    try:
        out = subprocess.check_output(["node", path], text=True, cwd=str(ROOT))
    except FileNotFoundError:
        print("node not found; skipping JS tests")
        return 0, 0, ["node missing"]
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise
    finally:
        Path(path).unlink(missing_ok=True)
    result = json.loads(out.strip().splitlines()[-1])
    return result["pass"], result["fail"], result["fails"]


def run_python_phrase_tests() -> tuple[int, int, list[str]]:
    sys.path.insert(0, str(ROOT))
    from text_to_gloss.rules import apply_phrases
    data = json.loads(PHRASES.read_text(encoding="ascii"))
    pass_n = fail_n = 0
    fails = []
    for p in data["phrases"]:
        got = apply_phrases(list(p["en"]))
        exp = list(p["gloss"])
        if got == exp:
            pass_n += 1
        else:
            fail_n += 1
            fails.append(f"{p['id']} | {p['en']} | got {got} expect {exp}")
    return pass_n, fail_n, fails


def main() -> None:
    jp, jf, jfails = run_node_tests()
    print(f"JS gloss rules: {jp} passed, {jf} failed, {jp + jf} total")
    for line in jfails[:40]:
        print("  FAIL", line)
    if len(jfails) > 40:
        print(f"  ... {len(jfails) - 40} more")

    pp, pf, pfails = run_python_phrase_tests()
    print(f"Python phrase rules: {pp} passed, {pf} failed, {pp + pf} total")
    for line in pfails[:20]:
        print("  FAIL", line)

    print(f"RULE_COUNT phrases+grammar = {jp + jf}")


if __name__ == "__main__":
    main()
