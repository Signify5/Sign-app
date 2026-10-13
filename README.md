# Signify

Speak English, see it signed in ASL by 3D hands.

It works in three steps, each in its own folder:

| Step | Folder | What it does |
| --- | --- | --- |
| 1 | `speech_to_text/` | Listens to the microphone and writes down what was said (vosk, offline) |
| 2 | `text_to_gloss/` | Turns the English sentence into ASL gloss, the ASL word order (spaCy) |
| 3 | `gloss_to_asl/` | Plays each gloss word as a sign on 3D MakeHuman hands |

## What you need

- Windows, macOS or Linux, with Python 3.11 or newer
- Git
- Internet: for setup, and while running (the 3D hands load three.js from the web)
- A microphone

You do not need to download any sign videos. The hand motion for all 1,973 signs is already in the repo, in `gloss_to_asl/assets/signs.json.gz`.

## Setup and run

```bash
git clone https://github.com/prasada29-gif/Sign-app.git signify
```

```bash
cd signify
```

```bash
python setup.py
```

On macOS or Linux, use `python3 setup.py`.

That one command does everything:

1. makes a virtual environment in `venv/`, a private folder of packages just for this project
2. installs every package from `requirements.txt` into it
3. downloads the speech models into `models/`
4. builds the sign pages (`exports/library.html` for the app, and the website in `web/`)
5. starts the app

The first run takes a few minutes. Running it again is safe, because finished steps are skipped. Add `--no-run` to set everything up without starting the app.

To start the app later, without setting up again:

Windows:

```bash
venv\Scripts\python main.py
```

macOS / Linux:

```bash
venv/bin/python main.py
```

Press record, speak, and the 3D hands sign what you said.

## The website

Setup also builds a website in `web/`: a mic page, and a sign library page with every sign, the alphabet and some phrases. To try it on your computer:

```bash
venv\Scripts\python -m http.server 8000 --directory web
```

Then open http://localhost:8000 in **Chrome or Edge**.

- The mic button uses the browser's own speech recognition. It only works in Chrome and Edge, and it sends your voice to Google or Microsoft to turn it into text.
- Typing a sentence works in any browser.

## Rebuilding the signs from the videos (optional)

You only need this to add signs or to track them again. Everyone else can skip it.

```bash
python setup.py --rebuild-signs
```

1. It downloads the WLASL videos into `datasets/` (about 3 GB) and tracks the hands into `clips/`. This takes hours for all ~2,000 words. You can stop and re-run it, because finished words are kept.
2. It checks each sign against ASL-LEX, and signs that look wrong get fingerspelled instead.
3. It packs every sign into a new `gloss_to_asl/assets/signs.json.gz`, and builds the pages again. Commit that file so everyone gets the new signs.

Some videos are gone from the internet. They are skipped and listed in `clips/setup_report.json`.

## Optional tools

These are in `gloss_to_asl/tools/`, and they run with the venv's Python. You only need them to rebuild things that are already in the repo:

- `fill_hands.py` and `reextract.py`: re-track hands for signs that came out badly
- `build_nouns.py`: rebuild the noun list (`pip install nltk` first)
- `build_hand.py`: rebuild the 3D hand from the MakeHuman source files (kept on one computer, not in git)

## Licenses and data: read before sharing

- **3D hands:** MakeHuman base mesh, skeleton and weights, CC0, free to use.
- **Sign motion:** tracked by this project from WLASL sign videos.
  - WLASL is shared under the Computational Use of Data Agreement (C-UDA 1.0), which puts no limits on Results, meaning things made from the data that don't contain it.
  - `signs.json.gz` holds only joint angles and positions, with no video frames, which is why it is in the repo.
  - WLASL asks for academic and non-commercial use only. Do not sell Signify or use it commercially.
  - The WLASL videos themselves (`datasets/`) and the full tracking output (`clips/`) stay out of git.
  - Credit: Li et al., "Word-level Deep Sign Language Recognition from Video", WACV 2020 (https://github.com/dxli94/WLASL).
- **Sign checking:** ASL-LEX, used for research.
- Translations are made by a program and have not been checked by a Deaf signer.
